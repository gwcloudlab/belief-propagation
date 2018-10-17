
#ifndef BELIEF_PROPAGATION_HPP
#define BELIEF_PROPAGATION_HPP

#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <cooperative_groups_helpers.h>
#include <mpi.h>

extern "C" {
    #include "../bnf-parser/expression.h"
    #include "../bnf-parser/Parser.h"
    #include "../bnf-parser/Lexer.h"
    #include "../bnf-xml-parser/xml-expression.h"
    #include "../snap-parser/snap-parser.h"
    #include "../csr-parser/csr-parser.h"
}

struct node_stream_data {
    int begin_index;
    int end_index;
    int streamNodeCount;
    cudaStream_t stream;

    int num_vertices;
    int num_edges;
    struct belief *buffers;
    struct belief *node_messages;
    struct joint_probability *joint_probabilities;
    struct belief *current_edge_messages;
    int *work_queue_nodes;
    int *num_work_items;
    int *work_queue_scratch;
    int * src_nodes_to_edges_nodes;
    int * src_nodes_to_edges_edges;
    int * dest_nodes_to_edges_nodes;
    int * dest_nodes_to_edges_edges;
};

struct edge_stream_data {
    int begin_index;
    int end_index;

    int streamEdgeCount;
    cudaStream_t stream;

    int num_vertices;
    int num_edges;

    struct belief *node_states;
    struct joint_probability *joint_probabilities;
    struct belief *current_edge_messages;
    int *work_queue_edges;
    int *num_work_items;
    int *work_queue_scratch;
    int * dest_nodes_to_edges_nodes;
    int * dest_nodes_to_edges_edges;
    int * edges_src_index;
    int * edges_dest_index;
};

void CheckCudaErrorAux (const char *, int, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

__device__
int atomic_add_inc(int *);

__device__
void update_work_queue_nodes_cuda(int *, int *, int *, struct belief *, int, float);

__device__
void update_work_queue_edges_cuda(int *, int *, int *, struct belief *, int, float);

__device__
void init_message_buffer_cuda(struct belief *, struct belief *, int, int);

__global__
void init_and_read_message_buffer_cuda_streaming(int, int, struct belief *, struct belief *,
        struct belief *, int *, int *, int, int, int *, int*);

__device__
void combine_message_cuda(struct belief *, struct belief *, int, int);

__device__
void combine_message_cuda_node_streaming(struct belief *, struct belief *, int, int);

__device__
void combine_message_cuda_edge_streaming(struct belief *, struct belief *, int, int);

__device__
void combine_page_rank_message_cuda(struct belief *, struct belief *, int, int);

__device__
void combine_viterbi_message_cuda(struct belief *, struct belief *, int, int);

__device__
void read_incoming_messages_cuda(struct belief *, struct belief *, int *,
                                 int *, int, int,
                                 int, int);

__device__
void send_message_for_edge_cuda(struct belief *, int, struct joint_probability *,
                                struct belief *);

__device__
void send_message_for_edge_cuda_streaming(struct belief *, int, struct joint_probability *,
                                struct belief *);
__device__
void send_message_for_node_cuda(struct belief *, int, struct joint_probability *,
                                struct belief *, int *, int *,
                                int, int);

__device__
void send_message_for_node_cuda_streaming(struct belief *, int, struct joint_probability *,
                                struct belief *, int *, int *,
                                int, int);

__device__
void marginalize_node(struct belief *, int,
                      struct belief *,
                      int *, int *,
                      int, int);

__device__
void marginalize_node_node_streaming(struct belief *, int,
                      struct belief *,
                      int *, int *,
                      int, int);

__device__
void marginalize_node_edge_streaming(struct belief *, int,
                      struct belief *,
                      int *, int *,
                      int, int);

__device__
void marginalize_page_rank_node(struct belief *, int,
                      struct belief *,
                      int *, int *,
                      int, int);

void argmax_node(struct belief *, int,
                 struct belief *,
                 int *, int *,
                 int, int);

__global__
void marginalize_nodes(struct belief *, struct belief *,
                       int *, int *,
                       int, int);

__global__
void marginalize_nodes_streaming(int, int,
                        struct belief *, struct belief *,
                       int *, int *,
                       int, int);

__global__
void marginalize_page_rank_nodes(struct belief *, struct belief *,
                       int *, int *,
                       int, int);

__global__
void argmax_nodes(struct belief *, struct belief *,
                                 int *, int *,
                                 int, int);

__global__
void loopy_propagate_main_loop(int, int,
                               struct belief *,
                               struct joint_probability *,
                               struct belief *,
                               int *, int *,
                               int *,
                               int *, int *,
                               int *, int *);

__global__
void loopy_propagate_init_read_buffer(struct belief *, int, int);

__global__
void __launch_bounds__(BLOCK_SIZE_NODE_STREAMING, MIN_BLOCKS_PER_MP)
send_message_for_node_cuda_streaming_kernel(int, int,
                                          int *, int *,
                                          struct belief *, int,
                                          struct joint_probability *,
                                          struct belief *,
                                          int *, int *,
                                          int);

__global__
void marginalize_node_cuda_streaming( int, int,
                                      int *, int *,
                                      struct belief *,
                                      struct belief *,
                                      int *, int *,
                                      int, int);

__global__
void page_rank_main_loop(int, int,
                               struct belief *,
                               struct joint_probability *,
                               struct belief *,
                               int *, int *,
                               int *, int *);

__global__
void viterbi_main_loop(int, int,
                         struct belief *,
                         struct joint_probability *,
                         struct belief *,
                         int *, int *,
                         int *, int *);

__device__
static void send_message_for_edge_iteration_cuda(struct belief *, int, int,
                                                 struct joint_probability *, struct belief *);

__global__
void send_message_for_edge_iteration_cuda_kernel(int, int *,
                                                 struct belief *, struct joint_probability *,
                                                 struct belief *);

__global__
void send_message_for_edge_iteration_cuda_work_queue_kernel(int, int *,
                                                            struct belief *, struct joint_probability *,
                                                            struct belief *,
                                                            int *, int *);

__global__
void send_message_for_edge_iteration_cuda_work_queue_kernel_streaming(
        int begin_index, int end_index,
        int * edges_src_index,
        struct belief *node_states,
        struct joint_probability *joint_probabilities,
        struct belief *current_edge_messages,
        int * work_queue_edges, int * num_work_queue_items);

__device__
void combine_loopy_edge_cuda(int, struct belief *, int, struct belief *);

__global__
void combine_loopy_edge_cuda_kernel(int, int *, struct belief *, struct belief *);

__global__
void combine_loopy_edge_cuda_work_queue_kernel(int, int *, struct belief *, struct belief *,
                                               int *, int *, int *);

__global__
void marginalize_loop_node_edge_kernel(struct belief *, int);

__global__
void marginalize_viterbi_beliefs(struct belief *, int);

__device__
float calculate_local_delta(int, struct belief *);

__global__
void calculate_delta(struct belief *, float *, float *, int);

__global__
void calculate_delta_6(struct belief *, float *, float *,
                       int, char, int);

__global__
void calculate_delta_simple(struct belief *, float *, float *,
                            int);

__global__
void update_work_queue_cuda_kernel(int *, int *, int*,
                                   struct belief *, int);
__global__
void update_work_queue_edges_cuda_kernel(int *, int *, int *, struct belief *, int);

void test_error();

int loopy_propagate_until_cuda_streaming(Graph_t, float, int);
int loopy_propagate_until_cuda_openmpi(Graph_t, float, int, int, int, int);
int loopy_propagate_until_cuda(Graph_t, float, int);
int loopy_propagate_until_cuda_edge(Graph_t, float, int);
int loopy_propagate_until_cuda_edge_streaming(Graph_t, float, int);
int loopy_propagate_until_cuda_edge_openmpi(Graph_t, float, int, int, int, int);


int page_rank_until_cuda(Graph_t, float, int);
int page_rank_until_cuda_edge(Graph_t, float, int);

int viterbi_until_cuda(Graph_t, float, int);
int viterbi_until_cuda_edge(Graph_t, float, int);

void run_test_loopy_belief_propagation_cuda(struct expression *, const char *, FILE *);
void run_test_loopy_belief_propagation_xml_file_cuda(const char *, FILE *);
void run_test_loopy_belief_propagation_xml_file_cuda_streaming(const char *, FILE *);
void run_test_loopy_belief_propagation_xml_file_edge_cuda(const char *, FILE *);
void run_test_loopy_belief_propagation_xml_file_edge_cuda_streaming(const char *, FILE *);

void run_test_loopy_belief_propagation_snap_file_cuda(const char *, const char *, FILE *);
void run_test_loopy_belief_propagation_snap_file_edge_cuda(const char *, const char *, FILE *);

void run_test_loopy_belief_propagation_mtx_files_cuda(const char *, const char *, FILE *);
void run_test_loopy_belief_propagation_mtx_files_cuda_streaming(const char *, const char *, FILE *);
void run_test_loopy_belief_propagation_mtx_files_cuda_openmpi(const char *, const char *, FILE *,
        int, int, int);
void run_test_loopy_belief_propagation_mtx_files_edge_cuda(const char *, const char *, FILE *);
void run_test_loopy_belief_propagation_mtx_files_edge_cuda_streaming(const char *, const char *, FILE *);
void run_test_loopy_belief_propagagtion_mtx_file_edge_openmpi_cuda(const char *, const char *,
                                                                   FILE *, int , int , int );


#endif //BELIEF_PROPAGATION_HPP
