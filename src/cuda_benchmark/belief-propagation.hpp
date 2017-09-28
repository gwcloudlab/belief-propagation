
#ifndef BELIEF_PROPAGATION_HPP
#define BELIEF_PROPAGATION_HPP

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
    #include "../snap-parser/snap-parser.h"
}

void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__device__
void init_message_buffer_cuda(struct belief *, struct belief *, unsigned int, unsigned int);

__device__
void combine_message_cuda(struct belief *, struct belief *, unsigned int, unsigned int);

__device__
void combine_page_rank_message_cuda(struct belief *, struct belief *, unsigned int, unsigned int);

__device__
void combine_viterbi_message_cuda(struct belief *, struct belief *, unsigned int, unsigned int);

__device__
void read_incoming_messages_cuda(struct belief *, struct belief *, unsigned int *,
                                 unsigned int *, unsigned int, unsigned int,
                                 unsigned int, unsigned int);

__device__
void send_message_for_edge_cuda(struct belief *, unsigned int, struct joint_probability *,
                                struct belief *);
__device__
void send_message_for_node_cuda(struct belief *, unsigned int, struct joint_probability *,
                                struct belief *, unsigned int *, unsigned int *,
                                unsigned int, unsigned int);

__device__
void marginalize_node(struct belief *, unsigned int,
                      struct belief *,
                      unsigned int *, unsigned int *,
                      unsigned int, unsigned int);

__device__
void marginalize_page_rank_node(struct belief *, unsigned int,
                      struct belief *,
                      unsigned int *, unsigned int *,
                      unsigned int, unsigned int);

void argmax_node(struct belief *, unsigned int,
                 struct belief *,
                 unsigned int *, unsigned int *,
                 unsigned int, unsigned int);

__global__
void marginalize_nodes(struct belief *, struct belief *,
                       unsigned int *, unsigned int *,
                       unsigned int, unsigned int);

__global__
void marginalize_page_rank_nodes(struct belief *, struct belief *,
                       unsigned int *, unsigned int *,
                       unsigned int, unsigned int);

__global__
void argmax_nodes(struct belief *, struct belief *,
                                 unsigned int *, unsigned int *,
                                 unsigned int, unsigned int);

__global__
void loopy_propagate_main_loop(unsigned int, unsigned int,
                               struct belief *,
                               struct joint_probability *,
                               struct belief *,
                               unsigned int *, unsigned int *,
                               unsigned int *, unsigned int *);

__global__
void page_rank_main_loop(unsigned int, unsigned int,
                               struct belief *,
                               struct joint_probability *,
                               struct belief *,
                               unsigned int *, unsigned int *,
                               unsigned int *, unsigned int *);

__global__
void viterbi_main_loop(unsigned int, unsigned int,
                         struct belief *,
                         struct joint_probability *,
                         struct belief *,
                         unsigned int *, unsigned int *,
                         unsigned int *, unsigned int *);

__device__
static void send_message_for_edge_iteration_cuda(struct belief *, unsigned int, unsigned int,
                                                 struct joint_probability *, struct belief *);

__global__
void send_message_for_edge_iteration_cuda_kernel(unsigned int, unsigned int *,
                                                 struct belief *, struct joint_probability *,
                                                 struct belief *);

__device__
void combine_loopy_edge_cuda(unsigned int, struct belief *, unsigned int, struct belief *);

__global__
void combine_loopy_edge_cuda_kernel(unsigned int, unsigned int *, struct belief *, struct belief *);

__global__
void marginalize_loop_node_edge_kernel(struct belief *, unsigned int);

__device__
float calculate_local_delta(unsigned int, struct belief *);

__global__
void calculate_delta(struct belief *, float *, float *, unsigned int);

__global__
void calculate_delta_6(struct belief *, float *, float *,
                       unsigned int, char, unsigned int);

__global__
void calculate_delta_simple(struct belief *, float *, float *,
                            unsigned int);

void test_error();

unsigned int loopy_propagate_until_cuda(Graph_t, float, unsigned int);
unsigned int loopy_propagate_until_cuda_edge(Graph_t, float, unsigned int);

unsigned int page_rank_until_cuda(Graph_t, float, unsigned int);
unsigned int page_rank_until_cuda_edge(Graph_t, float, unsigned int);

unsigned int viterbi_until_cuda(Graph_t, float, unsigned int);
unsigned int viterbi_until_cuda_edge(Graph_t, float, unsigned int);

void run_test_loopy_belief_propagation_cuda(struct expression *, const char *, FILE *);
void run_test_loopy_belief_propagation_xml_file_cuda(const char *, FILE *);
void run_test_loopy_belief_propagation_xml_file_edge_cuda(const char *, FILE *);

void run_test_loopy_belief_propagation_snap_file_cuda(const char *, const char *, FILE *);
void run_test_loopy_belief_propagation_snap_file_edge_cuda(const char *, const char *, FILE *);

#endif //BELIEF_PROPAGATION_HPP
