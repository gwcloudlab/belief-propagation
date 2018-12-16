#include "belief-propagation-kernels.hpp"
__constant__ struct joint_probability edge_joint_probability[1];
/**
 * Sets up the current buffer
 * @param message_buffer The message buffer to init
 * @param node_states The states to write
 * @param num_nodes The number of nodes in the graph
 */
__global__
void init_message_buffer_kernel(struct belief * __restrict__ message_buffer,
                                const struct belief * __restrict__ node_states, const int * __restrict__ node_states_size,
                                int num_nodes){
    int node_index, state_index, num_variables;

    for(node_index = blockIdx.x*blockDim.x + threadIdx.x; node_index < num_nodes; node_index += blockDim.x * gridDim.x){
        num_variables = node_states_size[node_index];

        for(state_index = blockIdx.y*blockDim.y + threadIdx.y; state_index < num_variables; state_index += blockDim.y * gridDim.y){
            message_buffer[node_index].data[state_index] = node_states[node_index].data[state_index];
        }
    }
}

/**
 * Combines the incoming messages with the given belief
 * @param dest The belief to update
 * @param edge_messages The buffered messages on the edge
 * @param num_vertices The number of nodes in the graph
 * @param node_index The index of the destination node
 * @param edge_offset The index offset for the edge
 * @param num_edges The number of edges in the graph
 * @param n_is_pow_2 Flag for adjusting shared memory
 * @param warp_size The warp size of the GPU
 */
__device__
void combine_message_cuda(struct belief * __restrict__ dest, const struct belief * __restrict__ edge_messages, int num_vertices, int node_index,
                          int edge_offset, int num_edges, char n_is_pow_2, int warp_size){
    __shared__ float shared_dest[BLOCK_SIZE_3_D_Z];
    __shared__ float shared_src[BLOCK_SIZE_3_D_Z];
    int index = threadIdx.z;

    if(index < num_vertices && edge_offset < num_edges){
        shared_dest[index] = dest[node_index].data[index];
        shared_src[index] = edge_messages[edge_offset].data[index];

        dest[node_index].data[index] = shared_dest[index] * shared_src[index];
    }
}

/**
 * Combines the incoming messages with the given PageRank
 * @param dest The belief to update
 * @param edge_messages The buffered messages on the edge
 * @param num_vertices The number of nodes in the graph
 * @param node_index The index of the destination node
 * @param edge_offset The index offset for the edge
 * @param num_edges The number of edges in the graph
 * @param n_is_pow_2 Flag for adjusting shared memory
 * @param warp_size The warp size of the GPU
 */
__device__
void combine_page_rank_message_cuda(struct belief * __restrict__ dest, const struct belief * __restrict__ edge_messages, int num_vertices, int node_index,
                          int edge_offset, int num_edges, char n_is_pow_2, int warp_size){
    __shared__ float shared_dest[BLOCK_SIZE_3_D_Z];
    __shared__ float shared_src[BLOCK_SIZE_3_D_Z];
    int index = threadIdx.z;

    if(index < num_vertices && edge_offset < num_edges){
        shared_dest[index] = dest[node_index].data[index];
        shared_src[index] = edge_messages[edge_offset].data[index];

        dest[node_index].data[index] = shared_dest[index] + shared_src[index];
    }
}

/**
 * Computes the argmax of the incoming messages with the given belief
 * @param dest The belief to update
 * @param edge_messages The buffered messages on the edge
 * @param num_vertices The number of nodes in the graph
 * @param node_index The index of the destination node
 * @param edge_offset The index offset for the edge
 * @param num_edges The number of edges in the graph
 * @param n_is_pow_2 Flag for adjusting shared memory
 * @param warp_size The warp size of the GPU
 */
__device__
void combine_viterbi_message_cuda(struct belief * __restrict__ dest, const struct belief * __restrict__ edge_messages, int num_vertices, int node_index,
                          int edge_offset, int num_edges, char n_is_pow_2, int warp_size){
    __shared__ float shared_dest[BLOCK_SIZE_3_D_Z];
    __shared__ float shared_src[BLOCK_SIZE_3_D_Z];
    int index = threadIdx.z;

    if(index < num_vertices && edge_offset < num_edges){
        shared_dest[index] = dest[node_index].data[index];
        shared_src[index] = edge_messages[edge_offset].data[index];

        dest[node_index].data[index] = fmaxf(shared_dest[index], shared_src[index]);
    }
}

/**
 * Reads the incoming messages and buffers them on the edge
 * @param message_buffer The message buffer of the edge
 * @param previous_messages The previous messages sent
 * @param dest_node_to_edges_nodes Parallel array; maps the nodes to the edges in which they are destination nodes; first half which maps nodes to their indices in dest_node_to_edges_edges
 * @param dest_node_to_edges_edges Parallel array; maps the nodes to the edges in which they are destination nodes; second half which maps the indices to edges
 * @param current_num_edges The number of edges in the graph
 * @param num_vertices The number of vertices in the graph
 * @param n_is_pow_2 Flag for determining how to adjust shared memory
 * @param warp_size The warp size of the GPU
 */
__global__
void read_incoming_messages_kernel(struct belief * __restrict__ message_buffer,
        const int * __restrict__ nodes_state_size,
        const struct belief * __restrict__ previous_messages,
                                   const int * __restrict__ dest_node_to_edges_nodes,
                                   const int * __restrict__ dest_node_to_edges_edges,
                                   int current_num_edges,
                                   int num_vertices,
                                   char n_is_pow_2, int warp_size){
    int node_index, edge_index, start_index, end_index, diff_index, tmp_index, num_variables;


    edge_index = blockIdx.y*blockDim.y + threadIdx.y;

    for(node_index = blockIdx.x*blockDim.x + threadIdx.x; node_index < num_vertices; node_index += blockDim.x * gridDim.x) {
        num_variables = nodes_state_size[node_index];

        start_index = dest_node_to_edges_nodes[node_index];
        if (node_index + 1 >= num_vertices) {
            end_index = current_num_edges;
        } else {
            end_index = dest_node_to_edges_nodes[node_index + 1];
        }
        diff_index = end_index - start_index;
        if (edge_index < diff_index) {
            tmp_index = dest_node_to_edges_edges[edge_index + start_index];
            combine_message_cuda(message_buffer, previous_messages, num_variables, node_index,
                                 tmp_index, current_num_edges, n_is_pow_2, warp_size);
        }
    }
}

/**
 * Combines the belief with the joint probability table and writes the result to the buffer
 * @param message_buffer The array of incoming beliefs
 * @param edge_index The index of the edge in which the belief is sent
 * @param node_index The incoming belief
 * @param joint_probabilities The joint probabilities of the edges
 * @param edge_messages The outbound buffer
 */
__device__
void send_message_for_edge_cuda(const struct belief * __restrict__ message_buffer,
                                int edge_index, int node_index,
                                int num_src, int num_dest,
                                struct belief * __restrict__ edge_messages,
                                float * __restrict__ edge_messages_previous,
                                float * __restrict__ edges_messages_current){
    int i, j;
    __shared__ float partial_sums[BLOCK_SIZE];
    __shared__ float sums[BLOCK_SIZE];
    __shared__ float s_belief[BLOCK_SIZE];


    sums[threadIdx.x] = 0.0;
    for(i = 0; i < num_src; ++i){
        partial_sums[threadIdx.x] = 0.0;
        for(j = 0; j < num_dest; ++j){
            s_belief[threadIdx.x] = message_buffer[node_index].data[j];
            partial_sums[threadIdx.x] += edge_joint_probability->data[i][j] * s_belief[threadIdx.x];
        }
        edge_messages[edge_index].data[i] = partial_sums[threadIdx.x];
        sums[threadIdx.x] += partial_sums[threadIdx.x];
    }
    if(sums[threadIdx.x] <= 0.0){
        sums[threadIdx.x] = 1.0;
    }
    edge_messages_previous[edge_index] = edges_messages_current[edge_index];
    edges_messages_current[edge_index] = sums[threadIdx.x];
    for(i = 0; i < num_src; ++i){
        edge_messages[edge_index].data[i] /= sums[threadIdx.x];
    }
}

/**
 * Sends the messages for all nodes in the graph
 * @param message_buffer The incoming beliefs
 * @param current_num_edges The number of edges in the graph
 * @param joint_probabilities The joint probability table of the graph
 * @param current_edge_messages The destination noe
 * @param src_node_to_edges_nodes Parallel array; maps nodes to the edges in which they are source nodes; first half; maps nodes to their index in src_node_to_edges_edges
 * @param src_node_to_edges_edges Parallel array; maps nodes to the edges in which they are source nodes; second half; maps the indices to the edges
 * @param num_vertices The number of vertices (nodes) in the graph
 */
__global__
void send_message_for_node_kernel(const struct belief * __restrict__ message_buffer, int current_num_edges,
                                          int joint_probabilities_dim_x,
                                  int joint_probabilities_dim_y,
                                          struct belief * __restrict__ current_edge_messages,
                                  float * __restrict__ current_edge_message_previous,
                                  float * __restrict__ current_edge_message_current,
                                  const int * __restrict__ src_node_to_edges_nodes,
                                  const int * __restrict__ src_node_to_edges_edges,
                                  int num_vertices){
    int node_index, edge_index, start_index, end_index, diff_index;

    edge_index = blockIdx.y*blockDim.y + threadIdx.y;

    for(node_index = blockIdx.x*blockDim.x + threadIdx.x; node_index < num_vertices; node_index += blockDim.x * gridDim.x){
        start_index = src_node_to_edges_nodes[node_index];
        if(node_index + 1 >= num_vertices){
            end_index = current_num_edges;
        }
        else{
            end_index = src_node_to_edges_nodes[node_index + 1];
        }
        diff_index = end_index - start_index;
        if (edge_index < diff_index) {
            edge_index = src_node_to_edges_edges[edge_index + start_index];
            send_message_for_edge_cuda(message_buffer, edge_index, node_index,
                    joint_probabilities_dim_x, joint_probabilities_dim_y, current_edge_messages,
                    current_edge_message_previous, current_edge_message_current);
        }
    }
}

/**
 * Marginalizes and normalizes the beliefs in the graph
 * @param message_buffer The source beliefs
 * @param node_states The current states of the nodes
 * @param current_edges_messages The current buffered messages on the graph
 * @param dest_node_to_edges_nodes Parallel array; maps nodes to the edges in which they are the destination nodes; first half; maps nodes to their indices in dest_node_to_edges_edges
 * @param dest_node_to_edges_edges Parallel array; maps nodes to the edges in which they are the destination nodes; second half; maps the indices to the edges
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 * @param n_is_pow_2 Flag for determining if padding needed for shared memory
 * @param warp_size The size of the warp of the GPU
 */
__global__
void marginalize_node_combine_kernel(struct belief * __restrict__ message_buffer,
        const struct belief * __restrict__ node_states, const int * __restrict__ node_states_size,
                             const struct belief * __restrict__ current_edges_messages,
                             const int * __restrict__ dest_node_to_edges_nodes,
                             const int * __restrict__ dest_node_to_edges_edges,
                             int num_vertices,
                             int num_edges, char n_is_pow_2, int warp_size){
    int node_index, edge_index, temp_edge_index, num_variables, start_index, end_index, diff_index;

    edge_index =  blockIdx.y*blockDim.y + threadIdx.y;


    for(node_index = blockIdx.x*blockDim.x + threadIdx.x; node_index < num_vertices; node_index += blockDim.x * gridDim.x) {
        num_variables = node_states_size[node_index];
        if(edge_index < num_variables){
            message_buffer[node_index].data[edge_index] = 1.0;
        }
        start_index = dest_node_to_edges_nodes[node_index];
        if(node_index + 1 >= num_vertices){
            end_index = num_edges;
        }
        else{
            end_index = dest_node_to_edges_nodes[node_index + 1];
        }
        diff_index = end_index - start_index;
        if(edge_index < diff_index){
            temp_edge_index = dest_node_to_edges_edges[edge_index + start_index];

            combine_message_cuda(message_buffer, current_edges_messages, num_variables, node_index, temp_edge_index, num_edges, n_is_pow_2, warp_size);
        }

    }
}

/**
 * Marginalizes and normalizes the beliefs in the graph
 * @param message_buffer The source beliefs
 * @param node_states The current states of the nodes
 * @param current_edges_messages The current buffered messages on the graph
 * @param dest_node_to_edges_nodes Parallel array; maps nodes to the edges in which they are the destination nodes; first half; maps nodes to their indices in dest_node_to_edges_edges
 * @param dest_node_to_edges_edges Parallel array; maps nodes to the edges in which they are the destination nodes; second half; maps the indices to the edges
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 * @param n_is_pow_2 Flag for determining if padding needed for shared memory
 * @param warp_size The size of the warp of the GPU
 */
__global__
void marginalize_page_rank_node_combine_kernel(struct belief * __restrict__ message_buffer,
        const struct belief * __restrict__ node_states, const int * __restrict__ node_states_size,
                                     const struct belief * __restrict__ current_edges_messages,
                                     const int * __restrict__ dest_node_to_edges_nodes,
                                     const int * __restrict__ dest_node_to_edges_edges,
                                     int num_vertices,
                                     int num_edges, char n_is_pow_2, int warp_size){
    int node_index, edge_index, temp_edge_index, num_variables, start_index, end_index, diff_index;

    edge_index =  blockIdx.y*blockDim.y + threadIdx.y;


    for(node_index = blockIdx.x*blockDim.x + threadIdx.x; node_index < num_vertices; node_index += blockDim.x * gridDim.x) {
        num_variables = node_states_size[node_index];
        if(edge_index < num_variables){
            message_buffer[node_index].data[edge_index] = 0.0;
        }
        start_index = dest_node_to_edges_nodes[node_index];
        if(node_index + 1 >= num_vertices){
            end_index = num_edges;
        }
        else{
            end_index = dest_node_to_edges_nodes[node_index + 1];
        }
        diff_index = end_index - start_index;
        if(edge_index < diff_index){
            temp_edge_index = dest_node_to_edges_edges[edge_index + start_index];

            combine_page_rank_message_cuda(message_buffer, current_edges_messages, num_variables, node_index, temp_edge_index, num_edges, n_is_pow_2, warp_size);
        }

    }
}

/**
 * Computes the argmax of the beliefs in the graph
 * @param message_buffer The source beliefs
 * @param node_states The current states of the nodes
 * @param current_edges_messages The current buffered messages on the graph
 * @param dest_node_to_edges_nodes Parallel array; maps nodes to the edges in which they are the destination nodes; first half; maps nodes to their indices in dest_node_to_edges_edges
 * @param dest_node_to_edges_edges Parallel array; maps nodes to the edges in which they are the destination nodes; second half; maps the indices to the edges
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 * @param n_is_pow_2 Flag for determining if padding needed for shared memory
 * @param warp_size The size of the warp of the GPU
 */
__global__
void argmax_node_combine_kernel(struct belief * __restrict__ message_buffer,
        const struct belief * __restrict__ node_states, const int * __restrict__ node_states_size,
                                     const struct belief * __restrict__ current_edges_messages,
                                     const int * __restrict__ dest_node_to_edges_nodes,
                                     const int * __restrict__  dest_node_to_edges_edges,
                                     int num_vertices,
                                     int num_edges, char n_is_pow_2, int warp_size){
    int node_index, edge_index, temp_edge_index, num_variables, start_index, end_index, diff_index;

    edge_index =  blockIdx.y*blockDim.y + threadIdx.y;


    for(node_index = blockIdx.x*blockDim.x + threadIdx.x; node_index < num_vertices; node_index += blockDim.x * gridDim.x) {
        num_variables = node_states_size[node_index];
        if(edge_index < num_variables){
            message_buffer[node_index].data[edge_index] = 1.0;
        }
        start_index = dest_node_to_edges_nodes[node_index];
        if(node_index + 1 >= num_vertices){
            end_index = num_edges;
        }
        else{
            end_index = dest_node_to_edges_nodes[node_index + 1];
        }
        diff_index = end_index - start_index;
        if(edge_index < diff_index){
            temp_edge_index = dest_node_to_edges_edges[edge_index + start_index];

            combine_viterbi_message_cuda(message_buffer, current_edges_messages, num_variables, node_index, temp_edge_index, num_edges, n_is_pow_2, warp_size);
        }

    }
}

/**
 * Marginalizes and normalizes nodes in the graph
 * @param message_buffer The incoming beliefs
 * @param node_states The destination belief to update
 * @param current_edges_messages The buffered beliefs on the edges
 * @param dest_node_to_edges_nodes Parallel array; maps nodes to the edges in which they are the destination node; first half; maps nodes to their indices in dest_node_to_edges_edges
 * @param dest_node_to_edges_edges Parallel array; maps nodes to the edges in which they are the destination node; second half; maps the indices to the edges
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 * @param n_is_pow_2 Flag for determining if padding is needed for shared memory
 * @param warp_size The size of the warp of the GPU
 */
__global__
void marginalize_sum_node_kernel(const struct belief * __restrict__  message_buffer,
        struct belief * __restrict__ node_states, const int * __restrict__ node_states_size,
                             const struct belief * __restrict__ current_edges_messages,
                             const int * __restrict__ dest_node_to_edges_nodes,
                             const int * __restrict__ dest_node_to_edges_edges,
                             int num_vertices,
                             int num_edges, char n_is_pow_2, int warp_size){
    int node_index, edge_index, num_variables;
    __shared__ float sum[BLOCK_SIZE_2_D_X];
    __shared__ float shared_message_buffer[BLOCK_SIZE_2_D_X][BLOCK_SIZE_2_D_Y];

    edge_index =  blockIdx.y*blockDim.y + threadIdx.y;

    for(node_index = blockIdx.x*blockDim.x + threadIdx.x; node_index < num_vertices; node_index += blockDim.x * gridDim.x) {
        num_variables = node_states_size[node_index];
        if(edge_index < num_variables) {
            if (edge_index == 0) {
                sum[threadIdx.x] = 0.0;
            }
            shared_message_buffer[threadIdx.x][threadIdx.y] *= message_buffer[node_index].data[edge_index];
            __syncthreads();

            atomicAdd(&sum[threadIdx.x], shared_message_buffer[threadIdx.x][threadIdx.y]);
            __syncthreads();
            if (threadIdx.y == 0 && sum[threadIdx.x] <= 0.0) {
                sum[threadIdx.x] = 1.0;
            }
            __syncthreads();
            node_states[node_index].data[edge_index] = shared_message_buffer[threadIdx.x][threadIdx.y] / sum[threadIdx.x];
        }
    }

}

__global__
void marginalize_dampening_factor_kernel(const struct belief * __restrict__ message_buffer,
        struct belief * __restrict__ node_states, const int * __restrict__ node_states_size,
                                          const struct belief * __restrict__ current_edges_messages,
                                          const int * __restrict__ dest_node_to_edges_nodes,
                                          const int * __restrict__ dest_node_to_edges_edges,
                                          int num_vertices,
                                          int num_edges, char n_is_pow_2, int warp_size){
    int node_index, edge_index, num_variables, end_index, start_index;
    __shared__ float factor[BLOCK_SIZE_2_D_X];
    __shared__ float shared_message_buffer[BLOCK_SIZE_2_D_X][BLOCK_SIZE_2_D_Y];

    edge_index =  blockIdx.y*blockDim.y + threadIdx.y;

    for(node_index = blockIdx.x*blockDim.x + threadIdx.x; node_index < num_vertices; node_index += blockDim.x * gridDim.x) {
        num_variables = node_states_size[node_index];
        start_index = dest_node_to_edges_nodes[node_index];
        if(node_index + 1 >= num_vertices){
            end_index = num_edges;
        }
        else{
            end_index = dest_node_to_edges_nodes[node_index + 1];
        }
        if(edge_index < num_variables) {
            if (edge_index == 0) {
                factor[threadIdx.x] = (1 - DAMPENING_FACTOR) / (end_index - start_index);
            }
            __syncthreads();
            shared_message_buffer[threadIdx.x][threadIdx.y] = factor[threadIdx.x] + DAMPENING_FACTOR * message_buffer[node_index].data[edge_index];
            __syncthreads();
            node_states[node_index].data[edge_index] = shared_message_buffer[threadIdx.x][threadIdx.y];
        }
    }

}

__global__
void marginalize_viterbi_beliefs(struct belief * __restrict__ nodes, const int * __restrict__ nodes_size, int num_nodes){
    int idx, i;
    float sum;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_nodes; idx += blockDim.x * gridDim.x){
        sum = 0.0;
        for(i = 0; i < nodes_size[idx]; ++i){
            sum += nodes[idx].data[i];
        }
        for(i = 0; i < nodes_size[idx]; ++i){
            nodes[idx].data[i] = nodes[idx].data[i] / sum;
        }
    }
}

/**
 * Marginalizes and normalizes nodes in the graph
 * @param message_buffer The incoming beliefs
 * @param node_states The destination belief to update
 * @param current_edges_messages The buffered beliefs on the edges
 * @param dest_node_to_edges_nodes Parallel array; maps nodes to the edges in which they are the destination node; first half; maps nodes to their indices in dest_node_to_edges_edges
 * @param dest_node_to_edges_edges Parallel array; maps nodes to the edges in which they are the destination node; second half; maps the indices to the edges
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 * @param n_is_pow_2 Flag for determining if padding is needed for shared memory
 * @param warp_size The size of the warp of the GPU
 */
__global__
void argmax_kernel(const struct belief * __restrict__ message_buffer,
        struct belief * __restrict__ node_states,
                                 const int * __restrict__ node_states_size,
                                 const struct belief * __restrict__ current_edges_messages,
                                 const int * __restrict__ dest_node_to_edges_nodes,
                                 const int * __restrict__ dest_node_to_edges_edges,
                                 int num_vertices,
                                 int num_edges, char n_is_pow_2, int warp_size){
    int node_index, edge_index, num_variables;
    __shared__ float shared_message_buffer[BLOCK_SIZE_2_D_X][BLOCK_SIZE_2_D_Y];

    edge_index =  blockIdx.y*blockDim.y + threadIdx.y;

    for(node_index = blockIdx.x*blockDim.x + threadIdx.x; node_index < num_vertices; node_index += blockDim.x * gridDim.x) {
        num_variables = node_states_size[node_index];
        if(edge_index < num_variables) {
            if (edge_index == 0) {
                shared_message_buffer[threadIdx.x][threadIdx.y] = -1.0f;
            }
            __syncthreads();
            shared_message_buffer[threadIdx.x][threadIdx.y] = fmaxf(shared_message_buffer[threadIdx.x][threadIdx.y], message_buffer[node_index].data[edge_index]);
            __syncthreads();

            node_states[node_index].data[edge_index] = shared_message_buffer[threadIdx.x][threadIdx.y];
        }
    }

}

/**
 * Calculates the delta for a given message
 * @param i The message's index
 * @param current_messages The current messages
 * @return The delta between the messages
 */
__device__
float calculate_local_delta(int i, const float * __restrict__ current_messages_previous, const float * __restrict__ current_messages_current){
    float delta, diff;

    diff = current_messages_previous[i] - current_messages_current[i];
    if(diff != diff){
        diff = 0.0;
    }
    delta = (float)fabs(diff);

    return delta;
}

/**
 * Calculates the delta used for testing for convergence via reduction
 * @param current_messages The current beliefs of the graph
 * @param delta The delta to write back
 * @param delta_array Temp array used to store partial deltas for reduction
 * @param num_edges The number of edges in the graph
 */
__global__
void calculate_delta(const float * __restrict__ current_messages_previous, const float * __restrict__ current_messages_current,
        float * __restrict__ delta, float * __restrict__ delta_array, int num_edges){
    extern __shared__ float shared_delta[];
    int tid, idx, i, s;

    tid = threadIdx.x;
    i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    for(idx = blockIdx.x*blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x){
        delta_array[idx] = calculate_local_delta(idx, current_messages_previous, current_messages_current);
    }
    __syncthreads();

    float my_delta = (i < num_edges) ? delta_array[i] : 0;

    if(i + BLOCK_SIZE < num_edges){
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

/**
 * Calculates the delta used for testing for convergence via reduction
 * @details Copied from the sample in the NVIDIA CUDA SDK
 * @param current_messages The current beliefs of the graph
 * @param delta The delta to write back
 * @param delta_array Temp array used to store partial deltas for reduction
 * @param num_edges The number of edges in the graph
 */
__global__
void calculate_delta_6(const float * __restrict__ current_messages_previous, const float * __restrict__ current_messages_current,
        float * __restrict__ delta, float * __restrict__ delta_array,
                       int num_edges, char n_is_pow_2, int warp_size) {
    extern __shared__ float shared_delta[];

    int offset;
    // perform first level of reduce
    // reading from global memory, writing to shared memory
    int idx;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int grid_size = blockDim.x * 2 * gridDim.x;

    for(idx = blockIdx.x*blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x){
        delta_array[idx] = calculate_local_delta(idx, current_messages_previous, current_messages_current);
    }
    __syncthreads();

    float my_delta = 0.0;

    while (i < num_edges) {
        my_delta = delta_array[i];

        // ensure we don't read out of bounds
        if (n_is_pow_2 || i + blockDim.x < num_edges) {
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

/**
 * Calculates the delta used for testing for convergence via reduction
 * @details Simple kernel used for testing
 * @param previous_messages The previous beliefs of the graph
 * @param current_messages The current beliefs of the graph
 * @param delta The delta to write back
 * @param delta_array Temp array used to store partial deltas for reduction
 * @param num_edges The number of edges in the graph
 */
__global__
void calculate_delta_simple(const float * __restrict__ current_messages_previous,
                            const float * __restrict__ current_messages_current,
                            float * __restrict__ delta, float * __restrict__ delta_array,
                            int num_edges) {
    extern __shared__ float shared_delta[];
    int tid, idx, i, s;

    tid = threadIdx.x;

    for(idx = blockIdx.x*blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x){
        delta_array[idx] = calculate_local_delta(idx, current_messages_previous, current_messages_current);
    }
    __syncthreads();

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_delta[tid] = (idx < num_edges) ? delta_array[idx] : 0;

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

/**
 * Helper function to test for errors for kernel calls
 */
void check_cuda_kernel_return_code(){
    cudaError_t err;

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

/**
 * Runs loopy BP on the graph
 * @param graph The graph to use
 * @param convergence The convergence threshold; if the delta falls below it, execution will stop
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations executed
 */
int loopy_propagate_until_cuda_kernels(Graph_t graph, float convergence, int max_iterations){
    int i, j, num_iter, num_vertices, num_edges, edge_joint_probability_dim_x, edge_joint_probability_dim_y;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct belief * message_buffer;

    struct belief * current_messages;
    int * current_messages_size;
    float * current_messages_previous;
    float * current_messages_current;

    int * src_nodes_to_edges_nodes;
    int * src_nodes_to_edges_edges;
    int * dest_nodes_to_edges_nodes;
    int * dest_nodes_to_edges_edges;

    struct belief * node_states;
    int * node_states_size;

    host_delta = 0.0;

    num_vertices = graph->current_num_vertices;
    num_edges = graph->current_num_edges;
    edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_nodes, sizeof(int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_edges, sizeof(int) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_nodes_to_edges_nodes, sizeof(int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_nodes_to_edges_edges, sizeof(int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_size, sizeof(int) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_current, sizeof(float) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_previous, sizeof(float) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states_size, sizeof(int) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&message_buffer, sizeof(struct belief) * num_vertices));

    // copy data

    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_size, graph->edges_messages_size, sizeof(int) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_previous, graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_current, graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states_size, graph->node_states_size, sizeof(int) * graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(int) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(int) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_nodes_to_edges_nodes, graph->src_nodes_to_edges_node_list, sizeof(int) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_nodes_to_edges_edges, graph->src_nodes_to_edges_edge_list, sizeof(int) * graph->current_num_edges, cudaMemcpyHostToDevice));

    const int blockEdge1dCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;

    const int blockNodeCount = (num_vertices + BLOCK_SIZE_2_D_X - 1)/BLOCK_SIZE_2_D_X;
    const int blockStateCount = (MAX_STATES + BLOCK_SIZE_2_D_Y - 1)/BLOCK_SIZE_2_D_Y;
    const int blockDegreeCount = (graph->max_degree + BLOCK_SIZE_2_D_Y - 1)/BLOCK_SIZE_2_D_Y;

    const int blockMessageNodeCount = (num_vertices + BLOCK_SIZE_3_D_X - 1)/BLOCK_SIZE_3_D_X;
    const int blockMessageDegreeCount = ( graph->max_degree + BLOCK_SIZE_3_D_Y - 1)/BLOCK_SIZE_3_D_Y;
    const int blockMessageStateCount = ( MAX_STATES + BLOCK_SIZE_3_D_Z - 1)/BLOCK_SIZE_3_D_Z;

    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(blockEdge1dCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    dim3 dimInitMessageBuffer(BLOCK_SIZE_2_D_X, BLOCK_SIZE_2_D_Y, 1);
    dim3 dimInitGrid(blockNodeCount, blockStateCount, 1);
    dim3 dimDegreeGrid(blockNodeCount, blockDegreeCount, 1);
//    int reduce2DSmemSize = (BLOCK_SIZE_2_D_Y <= 32) ? 2 * BLOCK_SIZE_2_D_Y * sizeof(float) : BLOCK_SIZE_2_D_Y * sizeof(float);

    dim3 dimMessagesBuffer(BLOCK_SIZE_3_D_X, BLOCK_SIZE_3_D_Y, BLOCK_SIZE_3_D_Z);
    dim3 dimMessagesGrid(blockMessageNodeCount, blockMessageDegreeCount, blockMessageStateCount);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            init_message_buffer_kernel<<<dimInitGrid, dimInitMessageBuffer>>>(message_buffer, node_states, node_states_size, num_vertices);
            check_cuda_kernel_return_code();
            //CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
            read_incoming_messages_kernel <<<dimMessagesGrid, dimMessagesBuffer>>>(message_buffer, node_states_size, current_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_edges, num_vertices, is_pow_2, WARP_SIZE);
            check_cuda_kernel_return_code();
            //CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
            send_message_for_node_kernel<<<dimInitGrid, dimInitMessageBuffer>>>(message_buffer, num_edges,
                    edge_joint_probability_dim_x, edge_joint_probability_dim_y,
                    current_messages, current_messages_previous, current_messages_current,
                    src_nodes_to_edges_nodes, src_nodes_to_edges_edges, num_vertices);
            check_cuda_kernel_return_code();
            //CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
            marginalize_node_combine_kernel<<<dimMessagesGrid, dimMessagesBuffer>>>(message_buffer,
                    node_states, node_states_size,
                    current_messages,
                    dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges, is_pow_2, WARP_SIZE);
            check_cuda_kernel_return_code();
            marginalize_sum_node_kernel<<<dimInitGrid, dimInitMessageBuffer>>>(message_buffer,
                    node_states, node_states_size,
                    current_messages,
                    dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges, is_pow_2, WARP_SIZE);
            check_cuda_kernel_return_code();
            //CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages_previous, current_messages_current, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        check_cuda_kernel_return_code();
        CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
     //   printf("Current delta: %f\n", host_delta);

        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            break;
        }
        previous_delta = host_delta;
    }

    // copy data back
    CUDA_CHECK_RETURN(cudaMemcpy(graph->node_states, node_states, sizeof(struct belief) * num_vertices, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(graph->edges_messages, current_messages, sizeof(struct belief) * num_edges, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_edges));
    CUDA_CHECK_RETURN(cudaFree(src_nodes_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(src_nodes_to_edges_edges));

    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(current_messages_size));
    CUDA_CHECK_RETURN(cudaFree(current_messages_previous));
    CUDA_CHECK_RETURN(cudaFree(current_messages_current));

    CUDA_CHECK_RETURN(cudaFree(message_buffer));

    CUDA_CHECK_RETURN(cudaFree(node_states));
    CUDA_CHECK_RETURN(cudaFree(node_states_size));

    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));

    /*printf("After=====");
    print_nodes(graph);
    print_edges(graph);*/

    return num_iter;
}


/**
 * Runs PageRank on the graph
 * @param graph The graph to use
 * @param convergence The convergence threshold; if the delta falls below it, execution will stop
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations executed
 */
int page_rank_until_cuda_kernels(Graph_t graph, float convergence, int max_iterations){
    int i, j, num_iter, num_vertices, num_edges, edge_joint_probability_dim_x, edge_joint_probability_dim_y;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct belief * message_buffer;

    struct belief * current_messages;
    int * current_messages_size;
    float * current_messages_previous;
    float * current_messages_current;

    int * src_nodes_to_edges_nodes;
    int * src_nodes_to_edges_edges;
    int * dest_nodes_to_edges_nodes;
    int * dest_nodes_to_edges_edges;

    struct belief * node_states;
    int * node_states_size;

    host_delta = 0.0;
    previous_delta = INFINITY;

    num_vertices = graph->current_num_vertices;
    num_edges = graph->current_num_edges;
    edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_nodes, sizeof(int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_edges, sizeof(int) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_nodes_to_edges_nodes, sizeof(int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_nodes_to_edges_edges, sizeof(int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_size, sizeof(int) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_previous, sizeof(float) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_current, sizeof(float) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states_size, sizeof(int) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&message_buffer, sizeof(struct belief) * num_vertices));

    // copy data
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages_size, sizeof(int) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states_size, sizeof(int) * graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(int) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(int) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_nodes_to_edges_nodes, graph->src_nodes_to_edges_node_list, sizeof(int) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_nodes_to_edges_edges, graph->src_nodes_to_edges_edge_list, sizeof(int) * graph->current_num_edges, cudaMemcpyHostToDevice));

    const int blockEdge1dCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;

    const int blockNodeCount = (num_vertices + BLOCK_SIZE_2_D_X - 1)/BLOCK_SIZE_2_D_X;
    const int blockStateCount = (MAX_STATES + BLOCK_SIZE_2_D_Y - 1)/BLOCK_SIZE_2_D_Y;
    const int blockDegreeCount = (graph->max_degree + BLOCK_SIZE_2_D_Y - 1)/BLOCK_SIZE_2_D_Y;

    const int blockMessageNodeCount = (num_vertices + BLOCK_SIZE_3_D_X - 1)/BLOCK_SIZE_3_D_X;
    const int blockMessageDegreeCount = ( graph->max_degree + BLOCK_SIZE_3_D_Y - 1)/BLOCK_SIZE_3_D_Y;
    const int blockMessageStateCount = ( MAX_STATES + BLOCK_SIZE_3_D_Z - 1)/BLOCK_SIZE_3_D_Z;

    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(blockEdge1dCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    dim3 dimInitMessageBuffer(BLOCK_SIZE_2_D_X, BLOCK_SIZE_2_D_Y, 1);
    dim3 dimInitGrid(blockNodeCount, blockStateCount, 1);
    dim3 dimDegreeGrid(blockNodeCount, blockDegreeCount, 1);
//    int reduce2DSmemSize = (BLOCK_SIZE_2_D_Y <= 32) ? 2 * BLOCK_SIZE_2_D_Y * sizeof(float) : BLOCK_SIZE_2_D_Y * sizeof(float);

    dim3 dimMessagesBuffer(BLOCK_SIZE_3_D_X, BLOCK_SIZE_3_D_Y, BLOCK_SIZE_3_D_Z);
    dim3 dimMessagesGrid(blockMessageNodeCount, blockMessageDegreeCount, blockMessageStateCount);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            init_message_buffer_kernel<<<dimInitGrid, dimInitMessageBuffer>>>(message_buffer, node_states, node_states_size, num_vertices);
            check_cuda_kernel_return_code();
            //CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
            read_incoming_messages_kernel <<<dimMessagesGrid, dimMessagesBuffer>>>(message_buffer, node_states_size,
                    current_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_edges, num_vertices, is_pow_2, WARP_SIZE);
            check_cuda_kernel_return_code();
            //CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
            send_message_for_node_kernel<<<dimInitGrid, dimInitMessageBuffer>>>(message_buffer, num_edges,
                    edge_joint_probability_dim_x, edge_joint_probability_dim_y,
                    current_messages, current_messages_previous, current_messages_current,
                    src_nodes_to_edges_nodes, src_nodes_to_edges_edges, num_vertices);
            check_cuda_kernel_return_code();
            //CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
            marginalize_page_rank_node_combine_kernel<<<dimMessagesGrid, dimMessagesBuffer>>>(message_buffer,
                    node_states, node_states_size,
                    current_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges, is_pow_2, WARP_SIZE);
            check_cuda_kernel_return_code();
            marginalize_dampening_factor_kernel<<<dimInitGrid, dimInitMessageBuffer>>>(message_buffer, node_states, node_states_size, current_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges, is_pow_2, WARP_SIZE);
            check_cuda_kernel_return_code();
            //CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages_previous, current_messages_current, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        check_cuda_kernel_return_code();
        CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
        //   printf("Current delta: %f\n", host_delta);

        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            break;
        }
        previous_delta = host_delta;
    }

    // copy data back
    CUDA_CHECK_RETURN(cudaMemcpy(graph->node_states, node_states, sizeof(struct belief) * num_vertices, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(graph->edges_messages, current_messages, sizeof(struct belief) * num_edges, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_edges));
    CUDA_CHECK_RETURN(cudaFree(src_nodes_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(src_nodes_to_edges_edges));

    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(current_messages_previous));
    CUDA_CHECK_RETURN(cudaFree(current_messages_current));
    CUDA_CHECK_RETURN(cudaFree(current_messages_size));

    CUDA_CHECK_RETURN(cudaFree(message_buffer));

    CUDA_CHECK_RETURN(cudaFree(node_states));
    CUDA_CHECK_RETURN(cudaFree(node_states_size));

    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));

    /*printf("After=====");
    print_nodes(graph);
    print_edges(graph);*/

    return num_iter;
}

/**
 * Runs Viterbi on the graph
 * @param graph The graph to use
 * @param convergence The convergence threshold; if the delta falls below it, execution will stop
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations executed
 */
int viterbi_until_cuda_kernels(Graph_t graph, float convergence, int max_iterations){
    int i, j, num_iter, num_vertices, num_edges, edge_joint_probability_dim_x, edge_joint_probability_dim_y;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct belief * message_buffer;

    struct belief * current_messages;
    int * current_messages_size;
    float * current_messages_previous;
    float * current_messages_current;

    int * src_nodes_to_edges_nodes;
    int * src_nodes_to_edges_edges;
    int * dest_nodes_to_edges_nodes;
    int * dest_nodes_to_edges_edges;

    struct belief * node_states;
    int * node_states_size;

    previous_delta = INFINITY;
    host_delta = 0.0;

    num_vertices = graph->current_num_vertices;
    num_edges = graph->current_num_edges;
    edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_nodes, sizeof(int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_edges, sizeof(int) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_nodes_to_edges_nodes, sizeof(int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_nodes_to_edges_edges, sizeof(int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_size, sizeof(int) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_previous, sizeof(float) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_current, sizeof(float) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states_size, sizeof(int) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&message_buffer, sizeof(struct belief) * num_vertices));

    // copy data
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_size, graph->edges_messages_size, sizeof(int) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_previous, graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_current, graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states_size, graph->node_states_size, sizeof(int) * graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(int) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(int) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_nodes_to_edges_nodes, graph->src_nodes_to_edges_node_list, sizeof(int) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_nodes_to_edges_edges, graph->src_nodes_to_edges_edge_list, sizeof(int) * graph->current_num_edges, cudaMemcpyHostToDevice));

    const int blockEdge1dCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;

    const int blockNodeCount = (num_vertices + BLOCK_SIZE_2_D_X - 1)/BLOCK_SIZE_2_D_X;
    const int blockStateCount = (MAX_STATES + BLOCK_SIZE_2_D_Y - 1)/BLOCK_SIZE_2_D_Y;
    const int blockDegreeCount = (graph->max_degree + BLOCK_SIZE_2_D_Y - 1)/BLOCK_SIZE_2_D_Y;

    const int blockMessageNodeCount = (num_vertices + BLOCK_SIZE_3_D_X - 1)/BLOCK_SIZE_3_D_X;
    const int blockMessageDegreeCount = ( graph->max_degree + BLOCK_SIZE_3_D_Y - 1)/BLOCK_SIZE_3_D_Y;
    const int blockMessageStateCount = ( MAX_STATES + BLOCK_SIZE_3_D_Z - 1)/BLOCK_SIZE_3_D_Z;

    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(blockEdge1dCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    dim3 dimInitMessageBuffer(BLOCK_SIZE_2_D_X, BLOCK_SIZE_2_D_Y, 1);
    dim3 dimInitGrid(blockNodeCount, blockStateCount, 1);
    dim3 dimDegreeGrid(blockNodeCount, blockDegreeCount, 1);
//    int reduce2DSmemSize = (BLOCK_SIZE_2_D_Y <= 32) ? 2 * BLOCK_SIZE_2_D_Y * sizeof(float) : BLOCK_SIZE_2_D_Y * sizeof(float);

    dim3 dimMessagesBuffer(BLOCK_SIZE_3_D_X, BLOCK_SIZE_3_D_Y, BLOCK_SIZE_3_D_Z);
    dim3 dimMessagesGrid(blockMessageNodeCount, blockMessageDegreeCount, blockMessageStateCount);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            init_message_buffer_kernel<<<dimInitGrid, dimInitMessageBuffer>>>(message_buffer, node_states, node_states_size, num_vertices);
            check_cuda_kernel_return_code();
            //CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
            read_incoming_messages_kernel <<<dimMessagesGrid, dimMessagesBuffer>>>(message_buffer, node_states_size,
                    current_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_edges, num_vertices, is_pow_2, WARP_SIZE);
            check_cuda_kernel_return_code();
            //CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
            send_message_for_node_kernel<<<dimInitGrid, dimInitMessageBuffer>>>(message_buffer, num_edges,
                    edge_joint_probability_dim_x, edge_joint_probability_dim_y,
                    current_messages, current_messages_previous, current_messages_current,
                    src_nodes_to_edges_nodes, src_nodes_to_edges_edges, num_vertices);
            check_cuda_kernel_return_code();
            //CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
            argmax_node_combine_kernel<<<dimMessagesGrid, dimMessagesBuffer>>>(message_buffer, node_states, node_states_size, current_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges, is_pow_2, WARP_SIZE);
            check_cuda_kernel_return_code();
            argmax_kernel<<<dimInitGrid, dimInitMessageBuffer>>>(message_buffer, node_states, node_states_size, current_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges, is_pow_2, WARP_SIZE);
            check_cuda_kernel_return_code();
            //CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages_previous, current_messages_current, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        check_cuda_kernel_return_code();
        CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
        //   printf("Current delta: %f\n", host_delta);

        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            marginalize_viterbi_beliefs<<<num_vertices, BLOCK_SIZE>>>(node_states, node_states_size, num_vertices);
            break;
        }
        previous_delta = host_delta;
    }

    // copy data back
    CUDA_CHECK_RETURN(cudaMemcpy(graph->node_states, node_states, sizeof(struct belief) * num_vertices, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(graph->edges_messages, current_messages, sizeof(struct belief) * num_edges, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_edges));
    CUDA_CHECK_RETURN(cudaFree(src_nodes_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(src_nodes_to_edges_edges));

    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(current_messages_size));
    CUDA_CHECK_RETURN(cudaFree(current_messages_previous));
    CUDA_CHECK_RETURN(cudaFree(current_messages_current));

    CUDA_CHECK_RETURN(cudaFree(message_buffer));

    CUDA_CHECK_RETURN(cudaFree(node_states));
    CUDA_CHECK_RETURN(cudaFree(node_states_size));

    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));

    /*printf("After=====");
    print_nodes(graph);
    print_edges(graph);*/

    return num_iter;
}

/**
 * Runs loopy BP on the file
 * @param file_name The path of the file to read
 */
void test_loopy_belief_propagation_kernels(char * file_name){
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

    loopy_propagate_until_cuda_kernels(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    printf("%s,loopy,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, time_elapsed);

    delete_expression(expression);

    graph_destroy(graph);
}

/**
 * Runs loopy BP on the AST root node
 * @param expression The BNF AST root node
 * @param file_name The input file path
 * @param out The file handle for the output CSV
 */
void run_test_loopy_belief_propagation_kernels(struct expression * expression, const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    int num_iterations;

    graph = build_graph(expression);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_cuda_kernels(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy,%d,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

/**
 * Reads the XML file and runs loopy BP on it
 * @param file_name The input XML file path
 * @param out The output CSV file handle
 */
void run_test_loopy_belief_propagation_xml_file_kernels(const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    int num_iterations;

    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_cuda_kernels(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy,%d,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}



/**
 * Reads the XML file and runs loopy BP on it
 * @param edge_file_name The file to read for the SNAP edges
 * @param node_file_name The file to read for the SNAP observed nodes
 * @param out The CSV file to output to
 */
void run_test_loopy_belief_propagation_snap_file_kernels(const char * edge_file_name, const char * node_file_name,
                                                         FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    int num_iterations;

    graph = parse_graph_from_snap_files(edge_file_name, node_file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_cuda_kernels(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy,%d,%d,%d,%d,%lf\n", edge_file_name, node_file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}



void run_test_loopy_belief_propagation_mtx_files_kernels(const char * edges_mtx, const char * nodes_mtx,
                                                         const struct joint_probability * edge_probability,
                                                                 int dim_x, int dim_y,
                                                         FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    int num_iterations;

    graph = build_graph_from_mtx(edges_mtx, nodes_mtx, edge_probability, dim_x, dim_y);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_cuda_kernels(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy,%d,%d,%d,%d,%lf\n", edges_mtx, nodes_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}


/**
 * Function for printing errors with kernel execution
 * @param file The source code file
 * @param line The line within file
 * @param statement The name of the kernel
 * @param err The error message
 */
void CheckCudaErrorAux (const char *file, int line, const char *statement, cudaError_t err)
{
    if (err == cudaSuccess) {
        return;
    }
    printf("%s returned %s (%d) at %s:%d\n", statement, cudaGetErrorString(err), err, file, line);
    exit (1);
}

