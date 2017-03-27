#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {
#include "../bnf-parser/expression.h"
#include "../bnf-parser/Parser.h"
#include "../bnf-parser/Lexer.h"
#include "../bnf-xml-parser/xml-expression.h"
}

int yyparse(struct expression ** expr, yyscan_t scanner);

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

texture<unsigned int, cudaTextureType1D, cudaReadModeElementType> tex_dest_node_to_edges;
texture<unsigned int, cudaTextureType1D, cudaReadModeElementType> tex_src_node_to_edges;

__device__
void init_message_buffer_cuda(double * buffer, Node_t node, unsigned int num_variables){
    unsigned int j;

    for(j = 0; j < num_variables; ++j){
        buffer[j] = node->states[j];
    }

}

__device__
void combine_message_cuda(double * dest, Edge_t edge, unsigned int length){
    unsigned int i;
    double * src;

    src = edge->message;

    for(i = 0; i < length; ++i){
        if(src[i] == src[i]){
            dest[i] = dest[i] * src[i];
        }
    }
}

__device__
void read_incoming_messages_cuda(double * message_buffer, Edge_t previous, unsigned int current_num_edges,
                            unsigned int num_vertices, unsigned int num_variables, unsigned int idx){
    unsigned int start_index, end_index, j, edge_index;
    Edge_t edge;

    start_index = tex1D(tex_dest_node_to_edges, idx);
    if(idx + 1 >= num_vertices){
        end_index = num_vertices + current_num_edges;
    }
    else{
        edge_index = tex1D(tex_dest_node_to_edges, idx + 1);
    }
    for(j = start_index; j < end_index; ++j){
        edge_index = tex1D(tex_dest_node_to_edges, j);
        edge = &previous[edge_index];

        combine_message_cuda(message_buffer, edge, num_variables);
    }
}

__device__
void send_message_for_edge_cuda(Edge_t edge, double * message){
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
    if(sum <= 0.0){
        sum = 1.0;
    }
    for(i = 0; i < num_src; ++i){
        edge->message[i] = edge->message[i] / sum;
    }
}

__device__
void send_message_for_node_cuda(double * message_buffer, unsigned int current_num_edges, Edge_t current,
                                unsigned int num_vertices, unsigned int idx){
    unsigned int start_index, end_index, j, edge_index;
    Edge_t edge;

    start_index = tex1D(tex_src_node_to_edges, idx);
    if(idx + 1 >= num_vertices){
        end_index = num_vertices + current_num_edges;
    }
    else{
        end_index = tex1D(tex_src_node_to_edges, idx + 1);
    }

    for(j = start_index; j < end_index; ++j){
        edge_index = tex1D(tex_src_node_to_edges, j);
        edge = &current[edge_index];
        send_message_for_edge_cuda(edge, message_buffer);
    }
}

__device__
void marginalize_node(Node_t nodes, unsigned int idx,
                        Edge_t current_edges, unsigned int num_vertices,
                      unsigned int num_edges){
    unsigned int i, num_variables, start_index, end_index, edge_index;
    char has_incoming;
    Edge_t edge;
    Node_t node;
    double sum;

    has_incoming = 0;

    node = &nodes[idx];
    num_variables = node->num_variables;

    double new_message[MAX_STATES];

    for(i = 0; i < num_variables; ++i){
        new_message[i] = 1.0;
    }

    start_index = tex1D(tex_dest_node_to_edges, idx);
    if(idx + 1 >= num_vertices){
        end_index = num_vertices + num_edges;
    }
    else{
        end_index = tex1D(tex_dest_node_to_edges, idx + 1);
    }

    for(i = start_index; i < end_index; ++i){
        edge_index = tex1D(tex_dest_node_to_edges, i);
        edge = &current_edges[edge_index];

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

__global__
void loopy_propagate_main_loop(unsigned int num_vertices, unsigned int num_edges,
                                Node_t nodes, Edge_t previous_edge, Edge_t current_edge){
    unsigned int idx, num_variables;
    double message_buffer[MAX_STATES];
    Node_t node;

    idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < num_vertices){
        node = &nodes[idx];
        num_variables = node->num_variables;

        init_message_buffer_cuda(message_buffer, node, num_variables);
        __syncthreads();

        read_incoming_messages_cuda(message_buffer, previous_edge, num_edges, num_vertices, num_variables, idx);
        __syncthreads();

        //send_message_for_node_cuda(message_buffer, num_edges, current_edge, num_vertices, idx);
        //__syncthreads();

        marginalize_node(nodes, idx, current_edge, num_vertices, num_edges);
    }

    __syncthreads();
}

__device__
double calculate_local_delta(unsigned int i, Edge_t previous_edges, Edge_t current_edges){
    Edge_t previous, current;
    double delta, diff;
    unsigned int k;

    delta = 0.0;

    previous = &previous_edges[i];
    current = &current_edges[i];

    for(k = 0; k < previous->x_dim; ++k){
        diff = previous->message[k] - current->message[k];
        if(diff != diff){
            diff = 0.0;
        }
        delta += fabs(diff);
    }

    return delta;
}

__global__
void calculate_delta(Edge_t previous_edges, Edge_t current_edges, double * delta, double * delta_array, unsigned int num_vertices){
    extern __shared__ double shared_delta[];
    unsigned int tid, idx, i, s;

    tid = threadIdx.x;
    idx = blockIdx.x*blockDim.x + threadIdx.x;
    i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    if(idx < num_vertices){
        delta_array[idx] = calculate_local_delta(idx, previous_edges, current_edges);
    }
    __syncthreads();

    double my_delta = (i < num_vertices) ? delta_array[i] : 0;

    if(i + BLOCK_SIZE < num_vertices){
        my_delta += delta_array[i + BLOCK_SIZE];
    }

    shared_delta[tid] = my_delta;
    __syncthreads();

    // do reduction in shared mememory
    for(s= blockDim.x / 2; s > 32; s>>=1){
        if(tid < s){
            shared_delta[tid] = my_delta = my_delta + shared_delta[tid + s];
        }

        __syncthreads();
    }

#if (__CUDA_ARCH__ >= 300)
    if(tid < 32){
        //fetch final intermediate sum from second warp
        if(BLOCK_SIZE >= 64){
            my_delta += shared_delta[tid + 32];
        }
        for(s = WARP_SIZE/2; s > 0; s /= 2){
            my_delta += __shfl_down(my_delta, s);
        }
    }
#else
    if((BLOCK_SIZE >= 64) && (tid < 32)){
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 32];
    }
    __syncthreads();
    if((BLOCK_SIZE >= 32) && (tid < 16)){
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 16];
    }
    __syncthreads();
    if((BLOCK_SIZE >= 16) && (tid < 8)){
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 8];
    }
    __syncthreads();
    if((BLOCK_SIZE >= 8) && (tid < 4)){
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 4];
    }
    __syncthreads();
    if((BLOCK_SIZE >= 4) && (tid < 2)){
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 2];
    }
    __syncthreads();
    if((BLOCK_SIZE >= 2) && (tid < 1)){
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 1];
    }
    __syncthreads();
#endif
    if(tid == 0) {
        *delta = my_delta;
    }
}

__global__
void calculate_delta_6(Edge_t previous_edges, Edge_t current_edges, double * delta, double * delta_array,
                       unsigned int num_vertices, char n_is_pow_2, unsigned int warp_size) {
    extern __shared__ double shared_delta[];

    unsigned int offset;
    // perform first level of reduce
    // reading from global memory, writing to shared memory
    unsigned int idx =  blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int grid_size = blockDim.x * 2 * gridDim.x;

    if(idx < num_vertices){
        delta_array[i] = calculate_local_delta(idx, previous_edges, current_edges);
    }
    __syncthreads();

    double my_delta = 0.0;

    while (i < num_vertices) {
        my_delta = delta_array[i];

        // ensure we don't read out of bounds
        if (n_is_pow_2 || i + blockDim.x < num_vertices) {
            my_delta += delta_array[i];
        }

        i += grid_size;
    }

    //each thread puts its local sum into shared memory
    shared_delta[tid] = my_delta;
    __syncthreads();

    // do reduction in shared mem
    if ((blockDim.x >= 512) && (tid < 256)) {
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 256];
    }
    __syncthreads();
    if ((blockDim.x >= 256) && (tid < 128)) {
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 128];
    }
    __syncthreads();
    if ((blockDim.x >= 128) && (tid < 64)) {
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 64];
    }
    __syncthreads();

#if (__CUDA_ARCH__ >= 300)
    if( tid < 32){
        // fetch final intermediate sum from 2nd warp
        if(blockDim.x >= 64){
            my_delta += shared_delta[tid + 32];
        }
        for(offset = warp_size/2; offset > 0; offset /= 2 ){
            my_delta += __shfl_down(my_delta, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockDim.x >= 64) && (tid < 32)) {
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 32];
    }
    __syncthreads();

    if ((blockDim.x >= 32) && (tid < 16)) {
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 16];
    }
    __syncthreads();

    if((blockDim.x >= 16) && (tid < 8)){
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 8];
    }
    __syncthreads();

    if((blockDim.x >= 8) && (tid < 4)){
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 4];
    }
    __syncthreads();

    if((blockDim.x >= 4) && (tid < 2)){
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 2];
    }
    __syncthreads();

    if((blockDim.x >= 2) && (tid < 1)){
        shared_delta[tid] = my_delta = my_delta + shared_delta[tid + 1];
    }
    __syncthreads();

#endif
    //write result for this block to global mem
    if(tid == 0){
        *delta = my_delta;
    }
}

__global__
void calculate_delta_simple(Edge_t previous_edges, Edge_t current_edges, double * delta, double * delta_array, unsigned int num_vertices) {
    extern __shared__ double shared_delta[];
    unsigned int tid, idx, i, s;

    tid = threadIdx.x;
    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_vertices) {
        delta_array[idx] = calculate_local_delta(idx, previous_edges, current_edges);
    }
    __syncthreads();

    shared_delta[tid] = (idx < num_vertices) ? delta_array[idx] : 0;

    __syncthreads();

    // do reduction in shared mem
    for(s = 1; s < blockDim.x; s *= 2){
        i = 2 * s * tid;
        if( i < blockDim.x ) {
            shared_delta[i] += shared_delta[i + s];
        }

        __syncthreads();
    }

    //write result for this block to global mem
    if(tid == 0){
        *delta = shared_delta[0];
    }
}

unsigned int loopy_propagate_until_cuda(Graph_t graph, double convergence, unsigned int max_iterations){
    unsigned int i, j, num_iter, num_vertices, num_edges;
    double * delta;
    double * delta_array;
    double previous_delta, host_delta;
    double * host_delta_ptr;
    char is_pow_2;
    cudaError_t err;

    host_delta = 0.0;

    struct cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    struct cudaArray * dest_node_to_edges;
    struct cudaArray * src_node_to_edges;
    Node_t nodes;
    Edge_t previous, current, temp, host_previous, host_current;

    num_vertices = graph->current_num_vertices;
    num_edges = graph->current_num_edges;
    host_previous = *(graph->previous);
    host_current = *(graph->current);

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMallocArray(&dest_node_to_edges, &channel_desc, graph->current_num_edges + graph->current_num_vertices, 1, cudaArrayDefault));
    CUDA_CHECK_RETURN(cudaMallocArray(&src_node_to_edges, &channel_desc, graph->current_num_edges + graph->current_num_vertices, 1, cudaArrayDefault));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&nodes, sizeof(struct node) * num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&previous, sizeof(struct edge) * num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current, sizeof(struct edge) * num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(double) * num_vertices));

    // copy data
    CUDA_CHECK_RETURN(cudaMemcpyToArray(dest_node_to_edges, 0, 0, graph->dest_nodes_to_edges, sizeof(int) * (num_edges + num_vertices), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToArray(src_node_to_edges, 0, 0, graph->src_nodes_to_edges, sizeof(int) * (num_edges + num_vertices), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(nodes, graph->nodes, sizeof(struct node) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(previous, host_previous, sizeof(struct edge) * num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current, host_current, sizeof(struct edge) * num_edges, cudaMemcpyHostToDevice));

    //bind textures
    tex_src_node_to_edges.addressMode[0] = cudaAddressModeWrap;
    tex_src_node_to_edges.addressMode[1] = cudaAddressModeWrap;
    tex_src_node_to_edges.filterMode = cudaFilterModePoint;
    tex_src_node_to_edges.normalized = 1;

    tex_dest_node_to_edges.addressMode[0] = cudaAddressModeWrap;
    tex_dest_node_to_edges.addressMode[1] = cudaAddressModeWrap;
    tex_dest_node_to_edges.filterMode = cudaFilterModePoint;
    tex_dest_node_to_edges.normalized = 1;

    CUDA_CHECK_RETURN(cudaBindTextureToArray(&tex_src_node_to_edges, src_node_to_edges, &channel_desc));
    CUDA_CHECK_RETURN(cudaBindTextureToArray(&tex_dest_node_to_edges, dest_node_to_edges, &channel_desc));

    const int blockCount = (num_vertices + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    num_iter = 0;

    dim3 dimReduceBlock(num_vertices, 1, 1);
    dim3 dimReduceGrid(blockCount, 1, 1);
    int reduceSmemSize = (num_vertices <= 32) ? 2 * num_vertices * sizeof(double) : num_vertices * sizeof(double);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            loopy_propagate_main_loop<<<blockCount, BLOCK_SIZE >>>(num_vertices, num_edges, nodes, previous, current);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
                exit(-1);
            }
            //swap pointers
            temp = current;
            current = previous;
            previous = temp;
            num_iter++;
        }
        //calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(previous, current, delta, delta_array, num_vertices, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(previous, current, delta, delta_array, num_vertices);
        calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(previous, current, delta, delta_array, num_vertices);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
        }
        CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(double), cudaMemcpyDeviceToHost));
        //printf("Current delta: %lf\n", host_delta);

        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            break;
        }
        previous_delta = host_delta;
    }

    // copy data back
    CUDA_CHECK_RETURN(cudaMemcpy(graph->nodes, nodes, sizeof(struct node) * num_vertices, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(host_current, current, sizeof(struct edge) * num_edges, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(host_previous, previous, sizeof(struct edge) * num_edges, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFreeArray(dest_node_to_edges));
    CUDA_CHECK_RETURN(cudaFreeArray(src_node_to_edges));
    CUDA_CHECK_RETURN(cudaFree(nodes));
    CUDA_CHECK_RETURN(cudaFree(previous));
    CUDA_CHECK_RETURN(cudaFree(current));
    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));

    /*printf("After=====");
    print_nodes(graph);
    print_edges(graph);*/

    return num_iter;
}

void test_ast(const char * expr)
{
    struct expression * expression;
    yyscan_t scanner;
    YY_BUFFER_STATE state;

    assert(yylex_init(&scanner) == 0);

    assert(scanner != NULL);
    assert(strlen(expr) > 0);

    state = yy_scan_string(expr, scanner);

    assert(yyparse(&expression, scanner) == 0);
    yy_delete_buffer(state, scanner);
    yylex_destroy(scanner);

    assert(expression != NULL);

    delete_expression(expression);
}

void test_file(const char * file_path)
{
    struct expression * expression;
    yyscan_t scanner;
    YY_BUFFER_STATE state;
    FILE * in;

    assert(yylex_init(&scanner) == 0);

    in = fopen(file_path, "r");

    yyset_in(in, scanner);

    assert(yyparse(&expression, scanner) == 0);
    //yy_delete_buffer(state, scanner);
    yylex_destroy(scanner);

    fclose(in);

    assert(expression != NULL);

    delete_expression(expression);
}

void test_parse_file(char * file_name){
    unsigned int i;
    struct expression * expression;
    yyscan_t scanner;
    YY_BUFFER_STATE state;
    FILE * in;
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;

    assert(yylex_init(&scanner) == 0);

    in = fopen(file_name, "r");

    yyset_in(in, scanner);

    assert(yyparse(&expression, scanner) == 0);
    //yy_delete_buffer(state, scanner);
    yylex_destroy(scanner);

    fclose(in);

    assert(expression != NULL);

    graph = build_graph(expression);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);

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
    printf("%s,regular,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, time_elapsed);

    //print_nodes(graph);

    assert(graph != NULL);

    delete_expression(expression);

    graph_destroy(graph);
}

void test_loopy_belief_propagation(char * file_name){
    struct expression * expression;
    yyscan_t scanner;
    YY_BUFFER_STATE state;
    FILE * in;
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;

    assert(yylex_init(&scanner) == 0);

    in = fopen(file_name, "r");

    yyset_in(in, scanner);

    assert(yyparse(&expression, scanner) == 0);
    //yy_delete_buffer(state, scanner);
    yylex_destroy(scanner);

    fclose(in);

    assert(expression != NULL);

    graph = build_graph(expression);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);

    start = clock();
    init_previous_edge(graph);

    loopy_propagate_until_cuda(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    printf("%s,loopy,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, time_elapsed);

    delete_expression(expression);

    graph_destroy(graph);
}

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
    printf("%s,regular,%d,%d,%d,2,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, time_elapsed);

    graph_destroy(graph);
}

void run_test_loopy_belief_propagation(struct expression * expression, const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    graph = build_graph(expression);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_cuda(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy,%d,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

void run_tests_with_file(const char * file_name, unsigned int num_iterations, FILE * out){
    unsigned int i;
    struct expression * expr;

    expr = parse_file(file_name);
    for(i = 0; i < num_iterations; ++i){
        run_test_belief_propagation(expr, file_name);
    }

    for(i = 0; i < num_iterations; ++i){
        run_test_loopy_belief_propagation(expr, file_name, out);
    }

    delete_expression(expr);
}

void run_tests_with_xml_file(const char * file_name, unsigned int num_iterations, FILE * out){
    unsigned int i;
    struct expression * expr;

    expr = parse_xml_file(file_name);
    /*for(i = 0; i < num_iterations; ++i){
        run_test_belief_propagation(expr, file_name);
    }*/

    for(i = 0; i < num_iterations; ++i){
        run_test_loopy_belief_propagation(expr, file_name, out);
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
*/
	//test_loopy_belief_propagation("../benchmark_files/dog.bif");
	//test_loopy_belief_propagation("../benchmark_files/alarm.bif");

    //test_file("dog.bif");
    //test_file("alarm.bif");

    /*expression = read_file("alarm.bif");

    assert(expression != NULL);

    delete_expression(expression);*/

    FILE * out = fopen("openmp_benchmark.csv", "w");
    fprintf(out, "File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)\n");
    fflush(out);

	/*run_tests_with_file("../benchmark_files/small/asia.bif", 1);
	run_tests_with_file("../benchmark_files/small/cancer.bif", 1);
	run_tests_with_file("../benchmark_files/small/earthquake.bif", 1);
	run_tests_with_file("../benchmark_files/small/sachs.bif", 1);
	run_tests_with_file("../benchmark_files/small/survey.bif", 1);
/*
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
	//run_tests_with_file("../benchmark_files/very_large/munin.bif", 1);
	run_tests_with_file("../benchmark_files/very_large/pathfinder.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/pigs.bif", 1);

    run_tests_with_xml_file("../benchmark_files/xml/bf_1000_2000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_1000_2000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_1000_2000_3.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_2000_4000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_2000_4000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_2000_4000_3.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_5000_10000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_5000_10000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_5000_10000_3.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_10000_20000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_10000_20000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_10000_20000_3.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_12000_24000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_12000_24000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_12000_24000_3.xml", 1);*/

    /*run_tests_with_xml_file("../benchmark_files/xml/bf_15000_30000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_15000_30000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_15000_30000_3.xml", 1);

    run_tests_with_xml_file("../benchmark_files/xml/bf_20000_40000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_20000_40000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_20000_40000_3.xml", 1);

    run_tests_with_xml_file("../benchmark_files/xml/bf_25000_50000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_25000_50000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_25000_50000_3.xml", 1);

    run_tests_with_xml_file("../benchmark_files/xml/bf_30000_60000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_30000_60000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_30000_60000_3.xml", 1);*/

    /*run_tests_with_xml_file("../benchmark_files/xml/bf_40000_80000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_40000_80000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_40000_80000_3.xml", 1);

    run_tests_with_xml_file("../benchmark_files/xml/bf_80000_160000_2.xml", 1);*/

    run_tests_with_xml_file("../benchmark_files/xml2/10_20.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/100_200.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/1000_2000.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/10000_20000.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/100000_200000.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/200000_400000.xml", 1, out);
    //run_tests_with_xml_file("../benchmark_files/xml2/300000_600000.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/400000_800000.xml", 1, out);
    //run_tests_with_xml_file("../benchmark_files/xml2/500000_1000000.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/600000_1200000.xml", 1, out);
    //run_tests_with_xml_file("../benchmark_files/xml2/700000_1400000.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/800000_1600000.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/1000000_2000000.xml", 1, out);
    //run_tests_with_xml_file("../benchmark_files/xml2/10000000_20000000.xml", 1, out);

    return 0;
}


/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned int line, const char *statement, cudaError_t err)
{
    if (err == cudaSuccess)
        return;
    printf("%s returned %s (%d) at %s:%d\n", statement, cudaGetErrorString(err), err, file, line);
    exit (1);
}

