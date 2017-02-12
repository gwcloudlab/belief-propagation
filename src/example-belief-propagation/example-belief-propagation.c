#include <stdlib.h>
#include <assert.h>

#include "../graph/graph.h"

static const int NUM_NODES = 4;
static const int NUM_EDGES = 5;
static const int NUM_VARIABLES = 2;

void add_nodes(Graph_t graph){
	double y2[NUM_VARIABLES];
	y2[0] = 1.0;
	y2[1] = 0.0;

	graph_add_node(graph, NUM_VARIABLES, "x1");
	graph_add_node(graph, NUM_VARIABLES, "x2");
	graph_add_node(graph, NUM_VARIABLES, "x3");
	graph_add_and_set_node_state(graph, NUM_VARIABLES, "y2", y2);
}

void add_edges(Graph_t graph){
	int i;

	double ** phi_1_2 = (double **)malloc(sizeof(double*) * NUM_VARIABLES);
	double ** phi_2_3 = (double **)malloc(sizeof(double*) * NUM_VARIABLES);
	double ** phi_2_4 = (double **)malloc(sizeof(double*) * NUM_VARIABLES);
	for(i = 0; i < NUM_VARIABLES; ++i){
		phi_1_2[i] = (double *)malloc(sizeof(double) * NUM_VARIABLES);
		phi_2_3[i] = (double *)malloc(sizeof(double) * NUM_VARIABLES);
		phi_2_4[i] = (double *)malloc(sizeof(double) * NUM_VARIABLES);
	}

	phi_1_2[0][0] = 1.0;
	phi_1_2[0][1] = 0.9;
	phi_1_2[1][0] = 0.9;
	phi_1_2[1][1] = 1.0;

	phi_2_3[0][0] = 0.1;
	phi_2_3[0][1] = 1.0;
	phi_2_3[1][0] = 1.0;
	phi_2_3[1][1] = 0.1;

	phi_2_4[0][0] = 1.0;
	phi_2_4[0][1] = 0.1;
	phi_2_4[1][0] = 0.1;
	phi_2_4[1][1] = 1.0;

	graph_add_edge(graph, 0, 1, 2, 2, phi_1_2);
	graph_add_edge(graph, 1, 0, 2, 2, phi_1_2);
	graph_add_edge(graph, 1, 2, 2, 2, phi_2_3);
	graph_add_edge(graph, 2, 1, 2, 2, phi_2_3);
	graph_add_edge(graph, 3, 1, 2, 2, phi_2_4);

	for(i = 0; i < NUM_VARIABLES; i++){
		free(phi_1_2[i]);
		free(phi_2_3[i]);
		free(phi_2_4[i]);
	}
	free(phi_1_2);
	free(phi_2_3);
	free(phi_2_4);
}

void assert_value(double diff){
	assert(diff > -0.0001);
	assert(diff < 0.0001);
}

void validate_nodes(Graph_t graph){
	Node_t node;
	double value;

	//x1
	node = &graph->nodes[0];
	value = node->states[0] - 0.521531100478469;
	assert_value(value);
	value = node->states[1] - 0.47846889952153115;
	assert_value(value);

	//x2
	node = &graph->nodes[1];
	value = node->states[0] - 0.9090909090909091;
	assert_value(value);
	value = node->states[1] - 0.09090909090909091;
	assert_value(value);

	//x3
	node = &graph->nodes[2];
	value = node->states[0] - 0.1652892561983471;
	assert_value(value);
	value = node->states[1] - 0.8347107438016529;
	assert_value(value);

	//y2
	node = &graph->nodes[3];
	value = node->states[0] - 1.0;
	assert_value(value);
	value = node->states[1] - 0.0;
	assert_value(value);
}

int main() {

	Graph_t graph;

	graph = create_graph(NUM_NODES, NUM_EDGES);

	add_nodes(graph);
	add_edges(graph);

	set_up_src_nodes_to_edges(graph);
	set_up_dest_nodes_to_edges(graph);

	print_nodes(graph);
	print_edges(graph);
	//print_src_nodes_to_edges(graph);
	//print_dest_nodes_to_edges(graph);

	send_from_leaf_nodes(graph);
	propagate(graph, graph->forward_queue, &graph->forward_queue_size, graph->backward_queue, &graph->backward_queue_size);

	reset_visited(graph);

	propagate(graph, graph->backward_queue, &graph->backward_queue_size, graph->forward_queue, &graph->forward_queue_size);
	marginalize(graph);

	print_nodes(graph);

	validate_nodes(graph);

	graph_destroy(graph);

	return 0;
}
