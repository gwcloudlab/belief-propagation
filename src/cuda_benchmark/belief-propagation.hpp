
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

#define FULL_MASK 0xffffffff

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
    int * node_messages_size;
    struct joint_probability *joint_probabilities;
    int * joint_probabilities_dim_x;
    int * joint_probabilities_dim_y;
    struct belief *current_edge_messages;
    float *current_edge_messages_previous;
    float *current_edge_messages_current;
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
    int *joint_probabilities_dim_x;
    int *joint_probabilities_dim_y;

    struct belief *current_edge_messages;
    float *current_edge_messages_previous;
    float *current_edge_messages_current;
    int *current_edge_messages_size;

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
void update_work_queue_nodes_cuda(int *, int *, int *, const float *, const float *, int, float);

__device__
void update_work_queue_edges_cuda(int *, int *, int *, const float *, const float *, int, float);

__device__
void init_message_buffer_cuda(struct belief *, const struct belief *, int, int);

__global__
void init_and_read_message_buffer_cuda_streaming(int, int, struct belief *, const struct belief *, const int *,
        const struct belief *, const int *, const int *, int, int, const int *, const int*);

__device__
void combine_message_cuda(struct belief *, const struct belief *, int, int);

__device__
void combine_message_cuda_node_streaming(struct belief *, const struct belief *, int, int);

__device__
void combine_message_cuda_edge_streaming(struct belief *, const struct belief *, int, int);

__device__
void combine_page_rank_message_cuda(struct belief *, const struct belief *, int, int);

__device__
void combine_viterbi_message_cuda(struct belief *, const struct belief *, int, int);

__device__
void read_incoming_messages_cuda(struct belief *, const struct belief *,
                                 const int *, const int *, int, int,
                                 int, int);

__device__
void send_message_for_edge_cuda(const struct belief *, int, const struct joint_probability *, const int *, const int *,
                                struct belief *, float *, float *);

__device__
void send_message_for_edge_cuda_streaming(const struct belief *, int, const struct joint_probability *, const int *, const int *,
                                struct belief *, float *, float *);
__device__
void send_message_for_node_cuda(const struct belief *, int, const struct joint_probability *, const int *, const int*,
                                struct belief *, float *, float *, const int *, const int *,
                                int, int);

__device__
void send_message_for_node_cuda_streaming(const struct belief *, int, const struct joint_probability *, const int *,
                                const int *, struct belief *, float *, float *, const int *, const int *,
                                int, int);

__device__
void marginalize_node(struct belief *, int *, int,
                      const struct belief *,
                      const int *, const int *,
                      int, int);

__device__
void marginalize_node_node_streaming(struct belief *, int *, int,
                      const struct belief *,
                      const int *, const int *,
                      int, int);

__device__
void marginalize_node_edge_streaming(struct belief *, int *, int,
                      const struct belief *,
                      const int *, const int *,
                      int, int);

__device__
void marginalize_page_rank_node(struct belief *, int *, int,
                      const struct belief *,
                      const int *, const int *,
                      int, int);

__device__
void argmax_node(struct belief *, int *, int,
                 const struct belief *,
                 const int *, const int *,
                 int, int);

__global__
void marginalize_nodes(struct belief *, int *, const struct belief *,
                       const int *, const int *,
                       int, int);

__global__
void marginalize_nodes_streaming(int, int,
                        struct belief *, int *, const struct belief *,
                       const int *, const int *,
                       int, int);

__global__
void marginalize_page_rank_nodes(struct belief *, int *, const struct belief *,
                       const int *, const int *,
                       int, int);

__global__
void argmax_nodes(struct belief *, int *, const struct belief *,
                                 const int *, const int *,
                                 int, int);

__global__
void loopy_propagate_main_loop(int, int,
                               struct belief *,
                               int *,
                               float *, float *,
                               const struct joint_probability *,
                               const int *, const int *,
                               struct belief *,
                               float *, float *,
                               int *, int *,
                               int *,
                               const int *, const int *,
                               const int *, const int *);

__global__
void loopy_propagate_init_read_buffer(struct belief *, int *, int, int);

__global__
void __launch_bounds__(BLOCK_SIZE_NODE_STREAMING, MIN_BLOCKS_PER_MP)
send_message_for_node_cuda_streaming_kernel(int, int,
                                          const int *, const int *,
                                          const struct belief *, int,
                                          const struct joint_probability *,
                                          const int *, const int *,
                                          struct belief *,
                                          float *, float *,
                                          const int *, const int *,
                                          int);

__global__
void marginalize_node_cuda_streaming( int, int,
                                      const int *, const int *,
                                      struct belief *,
                                      int *,
                                      const struct belief *,
                                      const int *, const int *,
                                      int, int);

__global__
void page_rank_main_loop(int, int,
                               struct belief *,
                               int *,
                               float *, float *,
                               const struct joint_probability *,
                               const int *, const int *,
                               struct belief *,
                               float *, float *,
                               const int *, const int *,
                               const int *, const int *);

__global__
void viterbi_main_loop(int, int,
                         struct belief *,
                         int *,
                         float *, float *,
                         const struct joint_probability *,
                         const int *, const int *,
                         struct belief *,
                         float *, float *,
                         const int *, const int *,
                         const int *, const int *);

__device__
void send_message_for_edge_iteration_cuda(const struct belief *, int, int,
                                                 const struct joint_probability *, const int *, const int *,
                                                         struct belief *, float *, float *);

__global__
void send_message_for_edge_iteration_cuda_kernel(int, const int *,
                                                 const struct belief *, const struct joint_probability *, const int *,
                                                 const int *,
                                                 struct belief *, float *, float *);

__global__
void send_message_for_edge_iteration_cuda_work_queue_kernel(int, const int *,
                                                            const struct belief *, const struct joint_probability *,
                                                            const int *, const int *,
                                                            struct belief *,
                                                            float *, float *,
                                                            int *, int *);

__global__
void send_message_for_edge_iteration_cuda_work_queue_kernel_streaming(
        int, int,
        const int *,
        const struct belief *,
        const struct joint_probability *,
        const int *, const int *,
        struct belief *,
        float *, float *,
        const int *, const int *);

__device__
void combine_loopy_edge_cuda(int, const struct belief *, const int *, int, struct belief *);

__global__
void combine_loopy_edge_cuda_kernel(int, const int *, const struct belief *, const int *, struct belief *);

__global__
void combine_loopy_edge_cuda_work_queue_kernel(int, const int *, const struct belief *, const float *, const float *, const int *, struct belief *,
                                               int *, int *, int *);

__global__
void combine_loopy_edge_cuda_work_queue_kernel_streaming(int, int, const int *, const struct belief *, const int *, struct belief *,
                                               const int *, const int *, const int *);

__global__
void marginalize_loop_node_edge_kernel(struct belief *, const int *, int);

__global__
void marginalize_viterbi_beliefs(struct belief *, int *, int);

__device__
float calculate_local_delta(int, const float *, const float *);

__global__
void calculate_delta(const float *, const float *, float *, float *, int);

__global__
void calculate_delta_6(const float *, const float *, float *, float *,
                       int, char, int);

__global__
void calculate_delta_simple(const float *, const float *, float *, float *,
                            int);

__global__
void update_work_queue_cuda_kernel(int *, int *, int*,
                                   float *, float *, int);
__global__
void update_work_queue_edges_cuda_kernel(int *, int *, int *, float *, float *, int);

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
