#include <stdio.h>
#include <assert.h>
#include <time.h>

#include "../bnf-parser/expression.h"
#include "../bnf-parser/Parser.h"
#include "../bnf-parser/Lexer.h"

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

int yyparse(struct expression ** expr, yyscan_t scanner);

struct expression * parse_file(const char * file_name){
    struct expression * expression;
    yyscan_t scanner;
    YY_BUFFER_STATE state;
    FILE * in;

    assert(yylex_init(&scanner) == 0);

    in = fopen(file_name, "r");

    yyset_in(in, scanner);

    assert(yyparse(&expression, scanner) == 0);
    //yy_delete_buffer(state, scanner);
    yylex_destroy(scanner);

    fclose(in);

    assert(expression != NULL);

    return expression;
}

__device__ void send_message_cuda(Edge_t edge, double * message) {
    unsigned int i, j, num_src, num_dest;
    double sum;

    num_src = edge->x_dim;
    num_dest = edge->y_dim;

    sum = 0.0;
    for(i = 0; i < num_src; ++i){
        edge->message[i] = 0.0;
        for(j = 0; j < num_dest; ++j){
            edge->message[i] += edge->joint_probabilities[i][j] * message[j];
        }
        sum += edge->message[i];
    }
    for(i = 0; i < num_src; ++i){
        edge->message[i] = edge->message[i] / sum;
    }
}

__device__ void combine_message_cuda(double * dest, Edge_t src_edge, unsigned int length){
    unsigned int i;
    double * src;

    src = src_edge->message;
    for(i = 0; i < length; ++i){
        if(src[i] > 0) {
            dest[i] = dest[i] * src[i];
        }
    }
}

__device__ void marginalize_node_cuda(Node_t nodes, unsigned int * dest_nodes_to_edges, unsigned int node_index, Edge_t edges){
    unsigned int i, num_variables, start_index, end_index, edge_index;
    char has_incoming;
    Edge_t edge;
    Node_t node;
    double sum;


    dest_nodes_to_edges = g->dest_nodes_to_edges;

    has_incoming = 0;

    node = &nodes[node_index];
    num_variables = node->num_variables;

    double new_message[MAX_STATES];
    for(i = 0; i < num_variables; ++i){
        new_message[i] = 1.0;
    }

    has_incoming = 0;

    start_index = dest_nodes_to_edges[node_index];
    if(node_index + 1 == g->current_num_vertices){
        end_index = g->current_num_vertices + g->current_num_edges;
    }
    else {
        end_index = dest_nodes_to_edges[node_index + 1];
    }

    for(i = start_index; i < end_index; ++i){
        edge_index = dest_nodes_to_edges[i];
        edge = &edges[edge_index];

        combine_message_cuda(new_message, edge, num_variables);
        has_incoming = 1;

    }
    if(has_incoming == 1){
        for(i = 0; i < num_variables; ++i){
            node->states[i] = new_message[i];
        }
    }
    sum = 0.0;
    for(i = 0; i < num_variables; ++i){
        sum += node->states[i];
    }
    if(sum <= 0.0){
        sum = 1.0;
    }

    for(i = 0; i < num_variables; ++i){
        node->states[i] = node->states[i] / sum;
    }
}

__global__ void loopy_propagate_one_iteration_cuda(Node_t nodes, Edge_t previous, Edge_t current, unsigned int * dest_node_to_edges,
                                                   unsigned int * src_node_to_edges, unsigned int num_vertices,
                                                   unsigned int current_num_edges){
    unsigned int j, num_variables, start_index, end_index, edge_index;
    Node_t node;
    Edge_t edge;
    Edge_t * temp;

    double message_buffer[MAX_STATES];


    unsigned int thread_index = blockIdx.x*blockDim.x+threadIdx.x;

    if(thread_index < num_vertices){
        node = &nodes[thread_index];
        num_variables = node->num_variables;
        //clear buffer
        for(j = 0; j < num_variables; ++j){
            message_buffer[j] = node->states[j];
        }

        //read incoming messages
        start_index = dest_node_to_edges[i];
        if(i + 1 >= num_vertices){
            end_index = num_vertices + current_num_edges;
        }
        else{
            end_index = dest_node_to_edges[i + 1];
        }

        for(j = start_index; j < end_index; ++j){
            edge_index = dest_node_to_edges[j];
            edge = &previous[edge_index];

            combine_message_cuda(message_buffer, edge, num_variables);
        }
/*
		printf("Message at node\n");
		print_node(graph, i);
		printf("[\t");
		for(j = 0; j < num_variables; ++j){
			printf("%.6lf\t", message_buffer[j]);
		}
		printf("\t]\n");*/


        //send message
        start_index = src_node_to_edges[i];
        if(thread_index + 1 >= num_vertices){
            end_index = num_vertices + current_num_edges;
        }
        else {
            end_index = src_node_to_edges[i + 1];
        }
        for(j = start_index; j < end_index; ++j){
            edge_index = src_node_to_edges[j];
            edge = &current[edge_index];
            /*printf("Sending on edge\n");
            print_edge(graph, edge_index);*/
            send_message_cuda(edge, message_buffer);
        }

    }

    __syncthreads();
    marginalize_node_cuda(graph, i, current);
    __syncthreads();

}


void loopy_propagate_until_batch_cuda(Graph_t graph, double convergence, unsigned int max_iterations){
    unsigned int i, j, k, num_nodes, grid_size;
    Edge_t previous_edges, previous, current, current_edges;
    double delta, diff, previous_delta;
    Edge_t * temp;

    struct graph * nv_graph;
    unsigned int * nv_src_nodes_to_graph;
    unsigned int * nv_dest_nodes_to_graph;

    previous_edges = *(graph->previous);
    current_edges = *(graph->current);

    num_nodes = graph->current_num_vertices;

    previous_delta = -1.0;

    grid_size = (graph->current_num_vertices+BLOCK_SIZE-1)/BLOCK_SIZE;

    for(i = 0; i < max_iterations; i+=BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            //printf("Current iteration: %d\n", i+1);
            loopy_propagate_one_iteration_cuda(graph);
            //swap previous and current
            temp = previous;
            graph->previous = graph->current;
            graph->current = temp;
        }

        delta = 0.0;

        for(j = 0; j < num_nodes; ++j){
            previous = &previous_edges[j];
            current = &current_edges[j];

            for(k = 0; k < previous->x_dim; ++k){
                diff = previous->message[k] - current->message[k];
                delta += fabs(diff);
            }
        }

        //printf("Current delta: %.6lf vs Previous delta: %.6lf\n", delta, previous_delta);
        if(delta < convergence || delta == previous_delta){
            break;
        }
        previous_delta = delta;
    }
}


void run_test_belief_propagation(struct expression * expression, const char * file_name){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int i;

    graph = build_graph(expression);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    calculate_diameter(graph);

    start = clock();
    init_levels_to_nodes(graph);
    //print_levels_to_nodes(graph);

    propagate_using_levels_start(graph);
    for(i = 1; i < graph->num_levels - 1; ++i){
        propagate_using_levels(graph, i);
    }
    reset_visited(graph);
    for(i = graph->num_levels - 1; i > 0; --i){
        propagate_using_levels(graph, i);
    }

    marginalize(graph);
    end = clock();

    time_elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%s,regular,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, time_elapsed);

    graph_destroy(graph);
}

void run_test_loopy_belief_propagation(struct expression * expression, const char * file_name){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;

    graph = build_graph(expression);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    loopy_propagate_until_batch_cuda(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    printf("%s,loopy,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, time_elapsed);

    graph_destroy(graph);
}

void run_tests_with_file(const char * file_name, unsigned int num_iterations){
    unsigned int i;
    struct expression * expr;

    expr = parse_file(file_name);
    for(i = 0; i < num_iterations; ++i){
        run_test_belief_propagation(expr, file_name);
    }

    for(i = 0; i < num_iterations; ++i){
        run_test_loopy_belief_propagation(expr, file_name);
    }

    delete_expression(expr);
}

int main(void)
{
/*
	extern int yydebug;
	yydebug = 1;
/*
	struct expression * expression = NULL;
	const char test[] = "// Bayesian Network in the Interchange Format\n// Produced by BayesianNetworks package in JavaBayes\n// Output created Sun Nov 02 17:49:49 GMT+00:00 1997\n// Bayesian network \nnetwork \"Dog-Problem\" { //5 variables and 5 probability distributions\nproperty \"credal-set constant-density-bounded 1.1\" ;\n}variable  \"light-on\" { //2 values\ntype discrete[2] {  \"true\"  \"false\" };\nproperty \"position = (218, 195)\" ;\n}\nvariable  \"bowel-problem\" { //2 values\ntype discrete[2] {  \"true\"  \"false\" };\nproperty \"position = (335, 99)\" ;\n}";
	test_ast(test);

  	test_parse_file("dog.bif");
	test_parse_file("alarm.bif");

	test_parse_file("very_large/andes.bif");
	test_loopy_belief_propagation("very_large/andes.bif");

	test_parse_file("Diabetes.bif");
	test_loopy_belief_propagation("Diabetes.bif");

	test_loopy_belief_propagation("dog.bif");
	test_loopy_belief_propagation("alarm.bif");
*/
    //test_file("dog.bif");
    //test_file("alarm.bif");

    /*expression = read_file("alarm.bif");

    assert(expression != NULL);

    delete_expression(expression);*/

    run_tests_with_file("../benchmark_files/small/asia.bif", 1);
    run_tests_with_file("../benchmark_files/small/cancer.bif", 1);
    run_tests_with_file("../benchmark_files/small/earthquake.bif", 1);
    run_tests_with_file("../benchmark_files/small/sachs.bif", 1);
    run_tests_with_file("../benchmark_files/small/survey.bif", 1);

    run_tests_with_file("../benchmark_files/medium/alarm.bif", 1);
    run_tests_with_file("../benchmark_files/medium/barley.bif", 1);
    //run_tests_with_file("../benchmark_files/medium/child.bif", 1);
    run_tests_with_file("../benchmark_files/medium/hailfinder.bif", 1);
    run_tests_with_file("../benchmark_files/medium/insurance.bif", 1);
    run_tests_with_file("../benchmark_files/medium/mildew.bif", 1);
    run_tests_with_file("../benchmark_files/medium/water.bif", 1);

    run_tests_with_file("../benchmark_files/large/hepar2.bif", 1);
    run_tests_with_file("../benchmark_files/large/win95pts.bif", 1);

    run_tests_with_file("../benchmark_files/very_large/andes.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/diabetes.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/link.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/munin1.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/munin2.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/munin3.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/munin4.bif", 1);
    //run_tests_with_file("very_large/pathfinder.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/pigs.bif", 1);

    return 0;
}
