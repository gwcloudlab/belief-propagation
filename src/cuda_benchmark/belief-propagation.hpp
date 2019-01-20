
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
    size_t begin_index;
    size_t end_index;
    size_t streamNodeCount;
    cudaStream_t stream;

    size_t num_vertices;
    size_t num_edges;
    struct belief *buffers;
    struct belief *node_messages;
    size_t * node_messages_size;
    size_t edge_joint_probability_dim_x;
    size_t edge_joint_probability_dim_y;
    struct belief *current_edge_messages;
    float *current_edge_messages_previous;
    float *current_edge_messages_current;
    size_t *work_queue_nodes;
    unsigned long long int *num_work_items;
    size_t *work_queue_scratch;
    size_t * src_nodes_to_edges_nodes;
    size_t * src_nodes_to_edges_edges;
    size_t * dest_nodes_to_edges_nodes;
    size_t * dest_nodes_to_edges_edges;
};

struct edge_stream_data {
    size_t begin_index;
    size_t end_index;

    size_t streamEdgeCount;
    cudaStream_t stream;

    size_t num_vertices;
    size_t num_edges;

    struct belief *node_states;

    size_t edge_joint_probability_dim_x;
    size_t edge_joint_probability_dim_y;

    struct belief *current_edge_messages;
    float *current_edge_messages_previous;
    float *current_edge_messages_current;
    size_t *current_edge_messages_size;

    size_t *work_queue_edges;
    unsigned long long int *num_work_items;
    size_t *work_queue_scratch;
    size_t * dest_nodes_to_edges_nodes;
    size_t * dest_nodes_to_edges_edges;
    size_t * edges_src_index;
    size_t * edges_dest_index;
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
size_t atomic_add_inc(size_t *);

__device__
void update_work_queue_nodes_cuda(size_t *, unsigned long long int *, size_t *, const float *, const float *, unsigned long long int, float);

__device__
void update_work_queue_edges_cuda(size_t *, unsigned long long int *, size_t *, const float *, const float *, unsigned long long int, float);

__device__
void init_message_buffer_cuda(struct belief *, const struct belief *, size_t, size_t);

__global__
void init_and_read_message_buffer_cuda_streaming(size_t, size_t, struct belief *, const struct belief *, const size_t *,
        const struct belief *, const size_t *, const size_t *, size_t, size_t, const size_t *, const unsigned long long int*);

__device__
void combine_message_cuda(struct belief *, const struct belief *, size_t, size_t);

__device__
void combine_message_cuda_node_streaming(struct belief *, const struct belief *, size_t, size_t);

__device__
void combine_message_cuda_edge_streaming(struct belief *, const struct belief *, size_t, size_t);

__device__
void combine_page_rank_message_cuda(struct belief *, const struct belief *, size_t, size_t);

__device__
void combine_viterbi_message_cuda(struct belief *, const struct belief *, size_t, size_t);

__device__
void read_incoming_messages_cuda(struct belief *, const struct belief *,
                                 const size_t *, const size_t *, size_t, size_t,
                                 size_t, size_t);

__device__
void send_message_for_edge_cuda(const struct belief *, size_t, const struct joint_probability *, const size_t *, const size_t *,
                                struct belief *, float *, float *);

__device__
void send_message_for_edge_cuda_streaming(const struct belief *, size_t, const struct joint_probability *, const size_t *, const size_t *,
                                struct belief *, float *, float *);
__device__
void send_message_for_node_cuda(const struct belief *, size_t, const struct joint_probability *, const size_t *, const size_t*,
                                struct belief *, float *, float *, const size_t *, const size_t *,
                                size_t, size_t);

__device__
void send_message_for_node_cuda_streaming(const struct belief *, size_t, const struct joint_probability *, const size_t *,
                                const size_t *, struct belief *, float *, float *, const size_t *, const size_t *,
                                size_t, size_t);

__device__
void marginalize_node(struct belief *, size_t *, size_t,
                      const struct belief *,
                      const size_t *, const size_t *,
                      size_t, size_t);

__device__
void marginalize_node_node_streaming(struct belief *, size_t *, size_t,
                      const struct belief *,
                      const size_t *, const size_t *,
                      size_t, size_t);

__device__
void marginalize_node_edge_streaming(struct belief *, size_t *, size_t,
                      const struct belief *,
                      const size_t *, const size_t *,
                      size_t, size_t);

__device__
void marginalize_page_rank_node(struct belief *, size_t *, size_t,
                      const struct belief *,
                      const size_t *, const size_t *,
                      size_t, size_t);

__device__
void argmax_node(struct belief *, size_t *, size_t,
                 const struct belief *,
                 const size_t *, const size_t *,
                 size_t, size_t);

__global__
void marginalize_nodes(struct belief *, size_t *, const struct belief *,
                       const size_t *, const size_t *,
                       size_t, size_t);

__global__
void marginalize_nodes_streaming(size_t, size_t,
                        struct belief *, size_t *, const struct belief *,
                       const size_t *, const size_t *,
                       size_t, size_t);

__global__
void marginalize_page_rank_nodes(struct belief *, size_t *, const struct belief *,
                       const size_t *, const size_t *,
                       size_t, size_t);

__global__
void argmax_nodes(struct belief *, size_t *, const struct belief *,
                                 const size_t *, const size_t *,
                                 size_t, size_t);

__global__
void loopy_propagate_main_loop(size_t, size_t,
                               struct belief *,
                               size_t *,
                               float *, float *,
                               const struct joint_probability *,
                               const size_t *, const size_t *,
                               struct belief *,
                               float *, float *,
                               size_t *, size_t *,
                               size_t *,
                               const size_t *, const size_t *,
                               const size_t *, const size_t *);

__global__
void loopy_propagate_init_read_buffer(struct belief *, size_t *, size_t, size_t);

__global__
void __launch_bounds__(BLOCK_SIZE_NODE_STREAMING, MIN_BLOCKS_PER_MP)
send_message_for_node_cuda_streaming_kernel(size_t, size_t,
                                          const size_t *, const size_t *,
                                          const struct belief *, size_t,
                                          const struct joint_probability *,
                                          const size_t *, const size_t *,
                                          struct belief *,
                                          float *, float *,
                                          const size_t *, const size_t *,
                                          size_t);

__global__
void marginalize_node_cuda_streaming( size_t, size_t,
                                      const size_t *, const size_t *,
                                      struct belief *,
                                      size_t *,
                                      const struct belief *,
                                      const size_t *, const size_t *,
                                      size_t, size_t);

__global__
void page_rank_main_loop(size_t, size_t,
                               struct belief *,
                               size_t *,
                               float *, float *,
                               const struct joint_probability *,
                               const size_t *, const size_t *,
                               struct belief *,
                               float *, float *,
                               const size_t *, const size_t *,
                               const size_t *, const size_t *);

__global__
void viterbi_main_loop(size_t, size_t,
                         struct belief *,
                         size_t *,
                         float *, float *,
                         const struct joint_probability *,
                         const size_t *, const size_t *,
                         struct belief *,
                         float *, float *,
                         const size_t *, const size_t *,
                         const size_t *, const size_t *);

__device__
void send_message_for_edge_iteration_cuda(const struct belief *, size_t, size_t,
                                                 size_t, size_t,
                                                         struct belief *, float *, float *);

__global__
void send_message_for_edge_iteration_cuda_kernel(size_t, const size_t *,
                                                 const struct belief *, size_t, size_t,
                                                 struct belief *, float *, float *);

__global__
void send_message_for_edge_iteration_cuda_work_queue_kernel(size_t, const size_t *,
                                                            size_t, size_t,
                                                            struct belief *,
                                                            float *, float *,
                                                            size_t *, size_t *);

__global__
void send_message_for_edge_iteration_cuda_work_queue_kernel_streaming(
        size_t, size_t,
        const size_t *,
        const struct belief *,
        size_t, size_t,
        struct belief *,
        float *, float *,
        const size_t *, const size_t *);

__device__
void combine_loopy_edge_cuda(size_t, const struct belief *, const size_t *, size_t, struct belief *);

__global__
void combine_loopy_edge_cuda_kernel(size_t, const size_t *, const struct belief *, const size_t *, struct belief *);

__global__
void combine_loopy_edge_cuda_work_queue_kernel(size_t, const size_t *, const struct belief *, const float *, const float *, const size_t *, struct belief *,
                                               size_t *, size_t *, size_t *);

__global__
void combine_loopy_edge_cuda_work_queue_kernel_streaming(size_t, size_t, const size_t *, const struct belief *, const size_t *, struct belief *,
                                               const size_t *, const size_t *, const size_t *);

__global__
void marginalize_loop_node_edge_kernel(struct belief *, const size_t *, size_t);

__global__
void marginalize_viterbi_beliefs(struct belief *, size_t *, size_t);

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
void update_work_queue_cuda_kernel(size_t *, unsigned long long int *, size_t*,
                                   float *, float *, size_t);
__global__
void update_work_queue_edges_cuda_kernel(size_t *, unsigned long long int *, size_t *, float *, float *, size_t);

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

void run_test_loopy_belief_propagation_mtx_files_cuda(const char *, const char *, const struct joint_probability *, int, int, FILE *);
void run_test_loopy_belief_propagation_mtx_files_cuda_streaming(const char *, const char *, const struct joint_probability *, int, int, FILE *);
void run_test_loopy_belief_propagation_mtx_files_cuda_openmpi(const char *, const char *, const struct joint_probability *, int, int, FILE *,
        int, int, int);
void run_test_loopy_belief_propagation_mtx_files_edge_cuda(const char *, const char *, const struct joint_probability *, int, int, FILE *);
void run_test_loopy_belief_propagation_mtx_files_edge_cuda_streaming(const char *, const char *, const struct joint_probability *, int, int, FILE *);
void run_test_loopy_belief_propagation_mtx_files_edge_cuda_openmpi(const char *, const char *, const struct joint_probability *, int, int,
                                                                   FILE *, int , int , int );


#endif //BELIEF_PROPAGATION_HPP
