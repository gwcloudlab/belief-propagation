#include "belief-propagation.hpp"

__device__ __forceinline__ unsigned int LaneMaskLt()
{
    unsigned int ret;
    asm ("mov.u32 %0, %%lanemask_lt;" : "=r"(ret) );
    return ret;
}



__device__
unsigned int atomic_add_inc(unsigned int * ctr) {
    // from https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
    unsigned int active = __activemask();
    int leader = __ffs(active) - 1;
    int change = __popc(active);
    unsigned int rank = __popc(active & LaneMaskLt());
    unsigned int warp_res;
    if(rank == 0) {
        warp_res = atomicAdd(ctr, change);
    }
    warp_res = __shfl_sync(active, warp_res, leader);
    return warp_res + rank;
}


__device__
void update_work_queue_nodes_cuda(unsigned int * work_queue_nodes, unsigned int * num_work_items, unsigned int *work_queue_scratch, struct belief * node_states, unsigned int num_vertices, float precision) {
    unsigned int i;
    unsigned int ctr = 0;
    unsigned int orig_num_work_items = *num_work_items;

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < *num_work_items; i += blockDim.x * gridDim.x){
        if(fabs(node_states[work_queue_nodes[i]].current - node_states[work_queue_nodes[i]].previous) >= precision) {
            work_queue_scratch[ctr] = work_queue_nodes[i];
            atomic_add_inc(&ctr);
        }
    }

    __syncthreads();
    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vertices; i += blockDim.x * gridDim.x){
        work_queue_nodes[i] = work_queue_scratch[i];
    }
    atomicCAS(num_work_items, orig_num_work_items, ctr);
}

__device__
void update_work_queue_edges_cuda(unsigned int * work_queue_edge, unsigned int * num_work_items, unsigned int *work_queue_scratch, struct belief * edge_states, unsigned int num_edges, float precision) {
    unsigned int i;
    unsigned int ctr = 0;
    unsigned int orig_num_work_items = *num_work_items;

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < *num_work_items; i += blockDim.x * gridDim.x){
        if(fabs(edge_states[work_queue_edge[i]].current - edge_states[work_queue_edge[i]].previous) >= precision) {
            work_queue_scratch[ctr] = work_queue_edge[i];
            atomic_add_inc(&ctr);
        }
    }

    __syncthreads();
    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < num_edges; i += blockDim.x * gridDim.x){
        work_queue_edge[i] = work_queue_scratch[i];
    }
    atomicCAS(num_work_items, orig_num_work_items, ctr);
}

__device__

/**
 * Initialize the message buffer to what is stored in node_states
 * @param buffer The message buffer
 * @param node_states The states to init to
 * @param num_variables The size of the arrays
 * @param node_index The index of the current belief
  */
__device__
void init_message_buffer_cuda(struct belief *buffer, struct belief *node_states, unsigned int num_variables, unsigned int node_index){
    unsigned int j;

    buffer->size = num_variables;
    for(j = 0; j < num_variables; ++j){
        buffer->data[j] = node_states[node_index].data[j];
    }

}

__global__
void init_and_read_message_buffer_cuda_streaming(
        unsigned int begin_index, unsigned int end_index,
        struct belief *buffers, struct belief *node_states,
                                                 struct belief * previous_messages,
                                                 unsigned int * dest_nodes_to_edges_nodes,
                                                 unsigned int * dest_nodes_to_edges_edges,
                                                 unsigned int current_num_edges,
                                                 unsigned int num_vertices,
        unsigned int *work_queue, unsigned int *num_work_queue_items) {
    unsigned int i, node_index, num_variables;

    for(i = blockIdx.x * blockDim.x + threadIdx.x + begin_index; i < end_index && i < *num_work_queue_items; i += blockDim.x * gridDim.x){
        node_index = work_queue[i];

        num_variables = node_states[node_index].size;
        init_message_buffer_cuda(&(buffers[node_index]), node_states, num_variables, node_index);
        read_incoming_messages_cuda(&(buffers[node_index]), previous_messages, dest_nodes_to_edges_nodes,
                                    dest_nodes_to_edges_edges, current_num_edges, num_vertices, num_variables, node_index);
    }
}

/**
 * Combine the dest node with the incoming belief
 * @param dest The belief to update
 * @param edge_messages The incoming belief
 * @param length The number of probabilities to combine
 * @param offset The offset in the incoming messages
 */
__device__
void combine_message_cuda(struct belief * dest, struct belief * edge_messages, unsigned int length, unsigned int offset){
    unsigned int i;
    float message;
    __shared__ float buffer[BLOCK_SIZE];

    for(i = 0; i < length; ++i){
        buffer[threadIdx.x] = dest->data[i];
        message = edge_messages[offset].data[i];
        if(message == message){
            buffer[threadIdx.x] *= message;
            dest->data[i] = buffer[threadIdx.x];
        }
    }
}

__device__
void combine_message_cuda_node_streaming(struct belief * dest, struct belief * edge_messages, unsigned int length, unsigned int offset){
    unsigned int i;
    float message;
    __shared__ float buffer[BLOCK_SIZE_NODE_STREAMING];

    for(i = 0; i < length; ++i){
        buffer[threadIdx.x] = dest->data[i];
        message = edge_messages[offset].data[i];
        if(message == message){
            buffer[threadIdx.x] *= message;
            dest->data[i] = buffer[threadIdx.x];
        }
    }
}

__device__
void combine_message_cuda_edge_streaming(struct belief * dest, struct belief * edge_messages, unsigned int length, unsigned int offset){
    unsigned int i;
    float message;
    __shared__ float buffer[BLOCK_SIZE_NODE_EDGE_STREAMING];

    for(i = 0; i < length; ++i){
        buffer[threadIdx.x] = dest->data[i];
        message = edge_messages[offset].data[i];
        if(message == message){
            buffer[threadIdx.x] *= message;
            dest->data[i] = buffer[threadIdx.x];
        }
    }
}

__device__
void combine_page_rank_message_cuda(struct belief * dest, struct belief * edge_messages, unsigned int length, unsigned int offset){
    unsigned int i;
    float message;
    __shared__ float buffer[BLOCK_SIZE];

    for(i = 0; i < length; ++i){
        buffer[threadIdx.x] = dest->data[i];
        message = edge_messages[offset].data[i];
        if(message == message){
            buffer[threadIdx.x] += message;
            dest->data[i] = buffer[threadIdx.x];
        }
    }
}

__device__
void combine_viterbi_message_cuda(struct belief * dest, struct belief * edge_messages, unsigned int length, unsigned int offset){
    unsigned int i;
    float message;
    __shared__ float buffer[BLOCK_SIZE];

    for(i = 0; i < length; ++i){
        buffer[threadIdx.x] = dest->data[i];
        message = edge_messages[offset].data[i];
        if(message == message){
            buffer[threadIdx.x] = fmaxf(buffer[threadIdx.x], message);
            dest->data[i] = buffer[threadIdx.x];
        }
    }
}

/**
 * Combines the incoming messages for the given node
 * @param message_buffer The current belief
 * @param dest_nodes_to_edges_nodes The indices in dest_nodes_to_edges_edges by node index
 * @param dest_nodes_to_edges_edges The indices of the edges indexed by their dest node
 * @param current_num_edges The number of edges in the graph
 * @param num_vertices The number of vertices in the graph
 * @param num_variables The number of beliefs in the graph
 * @param idx The index of the current node
 */
__device__
void read_incoming_messages_cuda(struct belief * message_buffer,
                                 struct belief * previous_messages,
                                 unsigned int * dest_nodes_to_edges_nodes,
                                 unsigned int * dest_nodes_to_edges_edges,
                                 unsigned int current_num_edges,
                            unsigned int num_vertices, unsigned int num_variables, unsigned int idx){
    unsigned int start_index, end_index, j, edge_index;

    start_index = dest_nodes_to_edges_nodes[idx];
    if(idx + 1 >= num_vertices){
        end_index = current_num_edges;
    }
    else{
        end_index = dest_nodes_to_edges_nodes[idx + 1];
    }
    for(j = start_index; j < end_index; ++j){
        edge_index = dest_nodes_to_edges_edges[j];

        combine_message_cuda(message_buffer, previous_messages, num_variables, edge_index);
    }
}

/**
 * Send the current beliefs along the edge to the current node
 * @param buffer The current node
 * @param edge_index The index of the edge
 * @param joint_probabilities The joint probability table on the edge
 * @param edge_messages The current beliefs
 */
__device__
void send_message_for_edge_cuda(struct belief * buffer, unsigned int edge_index,
                                struct joint_probability * joint_probabilities,
                                struct belief * edge_messages){
    unsigned int i, j, num_src, num_dest;
    float sum;
    struct joint_probability joint_probability;
    __shared__ float partial_sums[BLOCK_SIZE * MAX_STATES];

    joint_probability = joint_probabilities[edge_index];

    num_src = joint_probability.dim_x;
    num_dest = joint_probability.dim_y;

    sum = 0.0;
    for(i = 0; i < num_src; ++i){
        partial_sums[threadIdx.x * MAX_STATES + i] = 0.0;
        for(j = 0; j < num_dest; ++j){
            partial_sums[threadIdx.x * MAX_STATES + i] += joint_probability.data[i][j] * buffer->data[j];
        }
        sum += partial_sums[threadIdx.x * MAX_STATES + i];
    }
    if(sum <= 0.0){
        sum = 1.0;
    }
    edge_messages[edge_index].previous = edge_messages[edge_index].current;
    edge_messages[edge_index].current = sum;
    for(i = 0; i < num_src; ++i){
        edge_messages[edge_index].data[i] /= sum;
    }
}

__device__
void send_message_for_edge_cuda_streaming(struct belief * buffer, unsigned int edge_index,
                                struct joint_probability * joint_probabilities,
                                struct belief * edge_messages){
    unsigned int i, j, num_src, num_dest;
    float sum;
    struct joint_probability joint_probability;
    __shared__ float partial_sums[BLOCK_SIZE_NODE_STREAMING * MAX_STATES];

    joint_probability = joint_probabilities[edge_index];

    num_src = joint_probability.dim_x;
    num_dest = joint_probability.dim_y;

    sum = 0.0;
    for(i = 0; i < num_src; ++i){
        partial_sums[threadIdx.x * MAX_STATES + i] = 0.0;
        for(j = 0; j < num_dest; ++j){
            partial_sums[threadIdx.x * MAX_STATES + i] += joint_probability.data[i][j] * buffer->data[j];
        }
        sum += partial_sums[threadIdx.x * MAX_STATES + i];
    }
    if(sum <= 0.0){
        sum = 1.0;
    }
    edge_messages[edge_index].previous = edge_messages[edge_index].current;
    edge_messages[edge_index].current = sum;
    for(i = 0; i < num_src; ++i){
        edge_messages[edge_index].data[i] /= sum;
    }
}

/**
 * Propagate the current beliefs to current node
 * @param message_buffer The current node
 * @param current_num_edges The number of edges in the graph
 * @param joint_probabilities The list of joint probabilities
 * @param current_edge_messages The incoming messages
 * @param src_nodes_to_edges_nodes The indices in src_nodes_to_edges_edges indexed by src node index
 * @param src_nodes_to_edges_edges The edges indexed by their source node
 * @param num_vertices The number of the vertices in the graph
 * @param idx The current node index
 */
__device__
void send_message_for_node_cuda(struct belief *message_buffer, unsigned int current_num_edges,
                                struct joint_probability *joint_probabilities,
                                struct belief *current_edge_messages,
                                unsigned int * src_nodes_to_edges_nodes, unsigned int * src_nodes_to_edges_edges,
                                unsigned int num_vertices, unsigned int idx){
    unsigned int start_index, end_index, j, edge_index;

    start_index = src_nodes_to_edges_nodes[idx];
    if(idx + 1 >= num_vertices){
        end_index = current_num_edges;
    }
    else{
        end_index = src_nodes_to_edges_nodes[idx + 1];
    }

    for(j = start_index; j < end_index; ++j){
        edge_index = src_nodes_to_edges_edges[j];
        send_message_for_edge_cuda(message_buffer, edge_index, joint_probabilities, current_edge_messages);
    }
}

__device__
void send_message_for_node_cuda_streaming(struct belief *message_buffer, unsigned int current_num_edges,
                                struct joint_probability *joint_probabilities,
                                struct belief *current_edge_messages,
                                unsigned int * src_nodes_to_edges_nodes, unsigned int * src_nodes_to_edges_edges,
                                unsigned int num_vertices, unsigned int idx){
    unsigned int start_index, end_index, j, edge_index;

    start_index = src_nodes_to_edges_nodes[idx];
    if(idx + 1 >= num_vertices){
        end_index = current_num_edges;
    }
    else{
        end_index = src_nodes_to_edges_nodes[idx + 1];
    }

    for(j = start_index; j < end_index; ++j){
        edge_index = src_nodes_to_edges_edges[j];
        send_message_for_edge_cuda_streaming(message_buffer, edge_index, joint_probabilities, current_edge_messages);
    }
}

__global__
void
__launch_bounds__(BLOCK_SIZE_NODE_STREAMING, MIN_BLOCKS_PER_MP)
send_message_for_node_cuda_streaming_kernel(unsigned int begin_index, unsigned int end_index,
                                          unsigned int *work_queue, unsigned int *num_work_queue_items,
                                          struct belief *message_buffers, unsigned int current_num_edges,
                                          struct joint_probability *joint_probabilities,
                                          struct belief *current_edge_messages,
                                          unsigned int * src_nodes_to_edges_nodes, unsigned int * src_nodes_to_edges_edges,
                                          unsigned int num_vertices) {
    unsigned int i, node_index;

    for(i = blockIdx.x * blockDim.x + threadIdx.x + begin_index; i < end_index && i < *num_work_queue_items; i += blockDim.x * gridDim.x) {
        node_index = work_queue[i];

        send_message_for_node_cuda_streaming(&(message_buffers[node_index]), current_num_edges, joint_probabilities,
                                   current_edge_messages, src_nodes_to_edges_nodes, src_nodes_to_edges_edges,
        num_vertices, node_index);
    }
}

/**
 * Marginalizes and normalizes the belief probabilities for a given node
 * @param node_num_vars The number of variables for a given node
 * @param node_states The states of the given node
 * @param idx The node's index
 * @param current_edges_messages The array holding the current beliefs on the ege
 * @param dest_nodes_to_edges_nodes The parallel array holding the mapping of nodes to their edges in dest_nodes_to_edges_edges
 * @param dest_nodes_to_edges_edges Array holding the mapping of nodes to their edges in which they are the destination
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 */
__device__
void marginalize_node(struct belief *node_states, unsigned int idx,
                      struct belief *current_edges_messages,
                      unsigned int * dest_nodes_to_edges_nodes, unsigned int * dest_nodes_to_edges_edges,
                      unsigned int num_vertices, unsigned int num_edges){
    unsigned int i, num_variables, start_index, end_index, edge_index;
    float sum;

    num_variables = node_states[idx].size;

    struct belief new_belief;

    new_belief.size = num_variables;
    for(i = 0; i < num_variables; ++i){
        new_belief.data[i] = 1.0;
    }

    start_index = dest_nodes_to_edges_nodes[idx];
    if(idx + 1 >= num_vertices){
        end_index = num_edges;
    }
    else{
        end_index = dest_nodes_to_edges_nodes[idx + 1];
    }

    for(i = start_index; i < end_index; ++i){
        edge_index = dest_nodes_to_edges_edges[i];

        combine_message_cuda(&new_belief, current_edges_messages, num_variables, edge_index);
    }
    if(start_index < end_index){
        for(i = 0; i < num_variables; ++i){
             new_belief.data[i] *= node_states[idx].data[i];
        }
    }
    sum = 0.0;
    for(i = 0; i < num_variables; ++i){
        sum += new_belief.data[i];
    }
    if(sum <= 0.0){
        sum = 1.0;
    }
    for(i = 0; i < num_variables; ++i){
        node_states[idx].data[i] /= sum;
    }
}

__device__
void marginalize_node_node_streaming(struct belief *node_states, unsigned int idx,
                      struct belief *current_edges_messages,
                      unsigned int * dest_nodes_to_edges_nodes, unsigned int * dest_nodes_to_edges_edges,
                      unsigned int num_vertices, unsigned int num_edges){
    unsigned int i, num_variables, start_index, end_index, edge_index;
    float sum;

    num_variables = node_states[idx].size;

    struct belief new_belief;

    new_belief.size = num_variables;
    for(i = 0; i < num_variables; ++i){
        new_belief.data[i] = 1.0;
    }

    start_index = dest_nodes_to_edges_nodes[idx];
    if(idx + 1 >= num_vertices){
        end_index = num_edges;
    }
    else{
        end_index = dest_nodes_to_edges_nodes[idx + 1];
    }

    for(i = start_index; i < end_index; ++i){
        edge_index = dest_nodes_to_edges_edges[i];

        combine_message_cuda_node_streaming(&new_belief, current_edges_messages, num_variables, edge_index);
    }
    if(start_index < end_index){
        for(i = 0; i < num_variables; ++i){
            new_belief.data[i] *= node_states[idx].data[i];
        }
    }
    sum = 0.0;
    for(i = 0; i < num_variables; ++i){
        sum += new_belief.data[i];
    }
    if(sum <= 0.0){
        sum = 1.0;
    }
    for(i = 0; i < num_variables; ++i){
        node_states[idx].data[i] /= sum;
    }
}

__device__
void marginalize_node_edge_streaming(struct belief *node_states, unsigned int idx,
                      struct belief *current_edges_messages,
                      unsigned int * dest_nodes_to_edges_nodes, unsigned int * dest_nodes_to_edges_edges,
                      unsigned int num_vertices, unsigned int num_edges){
    unsigned int i, num_variables, start_index, end_index, edge_index;
    float sum;

    num_variables = node_states[idx].size;

    struct belief new_belief;

    new_belief.size = num_variables;
    for(i = 0; i < num_variables; ++i){
        new_belief.data[i] = 1.0;
    }

    start_index = dest_nodes_to_edges_nodes[idx];
    if(idx + 1 >= num_vertices){
        end_index = num_edges;
    }
    else{
        end_index = dest_nodes_to_edges_nodes[idx + 1];
    }

    for(i = start_index; i < end_index; ++i){
        edge_index = dest_nodes_to_edges_edges[i];

        combine_message_cuda_edge_streaming(&new_belief, current_edges_messages, num_variables, edge_index);
    }
    if(start_index < end_index){
        for(i = 0; i < num_variables; ++i){
            new_belief.data[i] *= node_states[idx].data[i];
        }
    }
    sum = 0.0;
    for(i = 0; i < num_variables; ++i){
        sum += new_belief.data[i];
    }
    if(sum <= 0.0){
        sum = 1.0;
    }
    for(i = 0; i < num_variables; ++i){
        node_states[idx].data[i] /= sum;
    }
}

__global__
void marginalize_node_cuda_streaming( unsigned int begin_index, unsigned int end_index,
                                unsigned int *work_queue, unsigned int *num_work_queue_items,
                                struct belief *node_states,
                                struct belief *current_edges_messages,
                                unsigned int * dest_nodes_to_edges_nodes, unsigned int * dest_nodes_to_edges_edges,
                                unsigned int num_vertices, unsigned int num_edges) {
    unsigned int i, node_index;

    for(i = blockIdx.x * blockDim.x + threadIdx.x + begin_index; i < end_index && i < *num_work_queue_items; i += blockDim.x * gridDim.x) {
        node_index = work_queue[i];

        marginalize_node_node_streaming(node_states, node_index, current_edges_messages, dest_nodes_to_edges_nodes,
                         dest_nodes_to_edges_edges, num_vertices, num_edges);
    }
}

/**
 * Marginalizes and normalizes the PageRanks for a given node
 * @param node_num_vars The number of variables for a given node
 * @param node_states The states of the given node
 * @param idx The node's index
 * @param current_edges_messages The array holding the current beliefs on the ege
 * @param dest_nodes_to_edges_nodes The parallel array holding the mapping of nodes to their edges in dest_nodes_to_edges_edges
 * @param dest_nodes_to_edges_edges Array holding the mapping of nodes to their edges in which they are the destination
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 */
__device__
void marginalize_page_rank_node(struct belief *node_states, unsigned int idx,
                                struct belief *current_edges_messages,
                                unsigned int * dest_nodes_to_edges_nodes, unsigned int * dest_nodes_to_edges_edges,
                                unsigned int num_vertices, unsigned int num_edges) {
    unsigned int i, num_variables, start_index, end_index, edge_index;
    float factor;

    num_variables = node_states[idx].size;

    struct belief new_belief;

    new_belief.size = num_variables;
    for (i = 0; i < num_variables; ++i) {
        new_belief.data[i] = 0.0;
    }

    start_index = dest_nodes_to_edges_nodes[idx];
    if (idx + 1 >= num_vertices) {
        end_index = num_edges;
    } else {
        end_index = dest_nodes_to_edges_nodes[idx + 1];
    }

    for (i = start_index; i < end_index; ++i) {
        edge_index = dest_nodes_to_edges_edges[i];

        combine_page_rank_message_cuda(&new_belief, current_edges_messages, num_variables, edge_index);
    }

    if (start_index < end_index) {
        factor = (1 - DAMPENING_FACTOR) / (end_index - start_index);
        for (i = 0; i < num_variables; ++i) {
            new_belief.data[i] = factor + DAMPENING_FACTOR * new_belief.data[i];
        }
    }
}

/**
 * Computes the argmax the belief probabilities for a given node
 * @param node_num_vars The number of variables for a given node
 * @param node_states The states of the given node
 * @param idx The node's index
 * @param current_edges_messages The array holding the current beliefs on the ege
 * @param dest_nodes_to_edges_nodes The parallel array holding the mapping of nodes to their edges in dest_nodes_to_edges_edges
 * @param dest_nodes_to_edges_edges Array holding the mapping of nodes to their edges in which they are the destination
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 */
__device__
void argmax_node(struct belief *node_states, unsigned int idx,
                      struct belief *current_edges_messages,
                      unsigned int * dest_nodes_to_edges_nodes, unsigned int * dest_nodes_to_edges_edges,
                      unsigned int num_vertices, unsigned int num_edges){
    unsigned int i, num_variables, start_index, end_index, edge_index;

    num_variables = node_states[idx].size;

    struct belief new_belief;

    new_belief.size = num_variables;
    for(i = 0; i < num_variables; ++i){
        new_belief.data[i] = -1.0f;
    }

    start_index = dest_nodes_to_edges_nodes[idx];
    if(idx + 1 >= num_vertices){
        end_index = num_edges;
    }
    else{
        end_index = dest_nodes_to_edges_nodes[idx + 1];
    }

    for(i = start_index; i < end_index; ++i){
        edge_index = dest_nodes_to_edges_edges[i];

        combine_viterbi_message_cuda(&new_belief, current_edges_messages, num_variables, edge_index);
    }
    if(start_index < end_index){
        for(i = 0; i < num_variables; ++i){
            new_belief.data[i] = fmaxf(new_belief.data[i], node_states[idx].data[i]);
        }
    }
}

/**
 * Marginalizes and normalizes all nodes in the graph
 * @param node_states The current states of all nodes in the graph
 * @param current_edges_messages The current messages held in transit along the edge
 * @param dest_nodes_to_edges_nodes The mapping of nodes to their edges; parallel array which maps nodes to their edge indices in dest_nodes_to_edges_edges
 * @param dest_nodes_to_edges_edges The mapping nodes of nodes to the edges; consists of edge indices for which the node is the destination node
 * @param num_vertices The number of vertices in the graph
 * @param num_edges The number of edges in the graph
 */
__global__
void marginalize_nodes(struct belief *node_states,
                       struct belief *current_edges_messages,
                       unsigned int * dest_nodes_to_edges_nodes, unsigned int * dest_nodes_to_edges_edges,
                       unsigned int num_vertices, unsigned int num_edges) {
    unsigned int idx;
    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices; idx += blockDim.x * gridDim.x){
        marginalize_node(node_states, idx, current_edges_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
    }
}

__global__
void marginalize_nodes_streaming(unsigned int begin_index, unsigned int end_index,struct belief *node_states,
                                 struct belief *current_edges_messages,
                                 unsigned int * dest_nodes_to_edges_nodes, unsigned int * dest_nodes_to_edges_edges,
                                 unsigned int num_vertices, unsigned int num_edges) {
    unsigned int idx;
    for (idx = blockIdx.x * blockDim.x + threadIdx.x + begin_index; idx < end_index; idx += blockDim.x * gridDim.x) {
        marginalize_node_edge_streaming(node_states, idx, current_edges_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges,
                         num_vertices, num_edges);
    }
}

/**
 * Marginalizes and normalizes all PageRank nodes in the graph
 * @param node_states The current states of all nodes in the graph
 * @param current_edges_messages The current messages held in transit along the edge
 * @param dest_nodes_to_edges_nodes The mapping of nodes to their edges; parallel array which maps nodes to their edge indices in dest_nodes_to_edges_edges
 * @param dest_nodes_to_edges_edges The mapping nodes of nodes to the edges; consists of edge indices for which the node is the destination node
 * @param num_vertices The number of vertices in the graph
 * @param num_edges The number of edges in the graph
 */
__global__
void marginalize_page_rank_nodes(struct belief *node_states,
                       struct belief *current_edges_messages,
                       unsigned int * dest_nodes_to_edges_nodes, unsigned int * dest_nodes_to_edges_edges,
                       unsigned int num_vertices, unsigned int num_edges) {
    unsigned int idx;
    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices; idx += blockDim.x * gridDim.x){
        marginalize_page_rank_node(node_states, idx, current_edges_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
    }
}

/**
 * Computes the argmax all nodes in the graph
 * @param node_states The current states of all nodes in the graph
 * @param current_edges_messages The current messages held in transit along the edge
 * @param dest_nodes_to_edges_nodes The mapping of nodes to their edges; parallel array which maps nodes to their edge indices in dest_nodes_to_edges_edges
 * @param dest_nodes_to_edges_edges The mapping nodes of nodes to the edges; consists of edge indices for which the node is the destination node
 * @param num_vertices The number of vertices in the graph
 * @param num_edges The number of edges in the graph
 */
__global__
void argmax_nodes(struct belief *node_states,
                       struct belief *current_edges_messages,
                       unsigned int * dest_nodes_to_edges_nodes, unsigned int * dest_nodes_to_edges_edges,
                       unsigned int num_vertices, unsigned int num_edges) {
    unsigned int idx;
    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices; idx += blockDim.x * gridDim.x){
        argmax_node(node_states, idx, current_edges_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
    }
}

/**
 * Runs loopy BP on the GPU
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 * @param node_messages The current beliefs of each node
 * @param joint_probabilities The joint probability table for each edge
 * @param previous_edge_messages The previous messages sent on the edges
 * @param current_edge_messages The current messages sent on the edges
 * @param src_nodes_to_edges_nodes The mapping of source nodes to their edges; parallel array; mapping of nodes to their edges in src_nodes_to_edges_edges
 * @param src_nodes_to_edges_edges The mapping of source nodes to their edges; consists of edges indexed by their source nodes
 * @param dest_nodes_to_edges_nodes The mapping of dest nodes to their edges; parallel array; mapping of nodes to their edges in dest_nodes_to_edges_edges
 * @param dest_nodes_to_edges_edges THe mapping of dest nodes to their edges; consists of edges indexed by their dest nodes
 */
__global__
void loopy_propagate_main_loop(unsigned int num_vertices, unsigned int num_edges,
                               struct belief *node_messages,
                               struct joint_probability *joint_probabilities,
                               struct belief *current_edge_messages,
                               unsigned int *work_queue_nodes, unsigned int *num_work_items,
                               unsigned int *work_queue_scratch,
                               unsigned int * src_nodes_to_edges_nodes, unsigned int * src_nodes_to_edges_edges,
                               unsigned int * dest_nodes_to_edges_nodes, unsigned int * dest_nodes_to_edges_edges){
    unsigned int i, idx, num_variables;
    struct belief new_belief;

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < *num_work_items; i += blockDim.x * gridDim.x){
        idx = work_queue_nodes[i];

        num_variables = node_messages[idx].size;

        init_message_buffer_cuda(&new_belief, node_messages, num_variables, idx);
        __syncthreads();

        read_incoming_messages_cuda(&new_belief, current_edge_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_edges, num_vertices, num_variables, idx);
        __syncthreads();

        send_message_for_node_cuda(&new_belief, num_edges, joint_probabilities, current_edge_messages, src_nodes_to_edges_nodes, src_nodes_to_edges_edges, num_vertices, idx);
        __syncthreads();

        marginalize_node(node_messages, idx, current_edge_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
        __syncthreads();
    }


    update_work_queue_nodes_cuda(work_queue_nodes, num_work_items, work_queue_scratch, node_messages, num_vertices, PRECISION_ITERATION);

    __syncthreads();


}


/**
 * Runs PageRank on the GPU
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 * @param node_messages The current beliefs of each node
 * @param joint_probabilities The joint probability table for each edge
 * @param previous_edge_messages The previous messages sent on the edges
 * @param current_edge_messages The current messages sent on the edges
 * @param src_nodes_to_edges_nodes The mapping of source nodes to their edges; parallel array; mapping of nodes to their edges in src_nodes_to_edges_edges
 * @param src_nodes_to_edges_edges The mapping of source nodes to their edges; consists of edges indexed by their source nodes
 * @param dest_nodes_to_edges_nodes The mapping of dest nodes to their edges; parallel array; mapping of nodes to their edges in dest_nodes_to_edges_edges
 * @param dest_nodes_to_edges_edges THe mapping of dest nodes to their edges; consists of edges indexed by their dest nodes
 */
__global__
void page_rank_main_loop(unsigned int num_vertices, unsigned int num_edges,
                               struct belief *node_messages,
                               struct joint_probability *joint_probabilities,
                               struct belief *current_edge_messages,
                               unsigned int * src_nodes_to_edges_nodes, unsigned int * src_nodes_to_edges_edges,
                               unsigned int * dest_nodes_to_edges_nodes, unsigned int * dest_nodes_to_edges_edges){
    unsigned int idx, num_variables;
    struct belief new_belief;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices; idx += blockDim.x * gridDim.x){
        num_variables = node_messages[idx].size;

        init_message_buffer_cuda(&new_belief, node_messages, num_variables, idx);
        __syncthreads();

        read_incoming_messages_cuda(&new_belief, current_edge_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_edges, num_vertices, num_variables, idx);
        __syncthreads();

        send_message_for_node_cuda(&new_belief, num_edges, joint_probabilities, current_edge_messages, src_nodes_to_edges_nodes, src_nodes_to_edges_edges, num_vertices, idx);
        __syncthreads();

        marginalize_page_rank_node(node_messages, idx, current_edge_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
        __syncthreads();
    }
}

/**
 * Runs Viterbi on the GPU
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 * @param node_messages The current beliefs of each node
 * @param joint_probabilities The joint probability table for each edge
 * @param previous_edge_messages The previous messages sent on the edges
 * @param current_edge_messages The current messages sent on the edges
 * @param src_nodes_to_edges_nodes The mapping of source nodes to their edges; parallel array; mapping of nodes to their edges in src_nodes_to_edges_edges
 * @param src_nodes_to_edges_edges The mapping of source nodes to their edges; consists of edges indexed by their source nodes
 * @param dest_nodes_to_edges_nodes The mapping of dest nodes to their edges; parallel array; mapping of nodes to their edges in dest_nodes_to_edges_edges
 * @param dest_nodes_to_edges_edges THe mapping of dest nodes to their edges; consists of edges indexed by their dest nodes
 */
__global__
void viterbi_main_loop(unsigned int num_vertices, unsigned int num_edges,
                         struct belief *node_messages,
                         struct joint_probability *joint_probabilities,
                         struct belief *current_edge_messages,
                         unsigned int * src_nodes_to_edges_nodes, unsigned int * src_nodes_to_edges_edges,
                         unsigned int * dest_nodes_to_edges_nodes, unsigned int * dest_nodes_to_edges_edges){
    unsigned int idx, num_variables;
    struct belief new_belief;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices; idx += blockDim.x * gridDim.x){
        num_variables = node_messages[idx].size;

        init_message_buffer_cuda(&new_belief, node_messages, num_variables, idx);
        __syncthreads();

        read_incoming_messages_cuda(&new_belief, current_edge_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_edges, num_vertices, num_variables, idx);
        __syncthreads();

        send_message_for_node_cuda(&new_belief, num_edges, joint_probabilities, current_edge_messages, src_nodes_to_edges_nodes, src_nodes_to_edges_edges, num_vertices, idx);
        __syncthreads();

        argmax_node(node_messages, idx, current_edge_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
        __syncthreads();
    }
}

/**
 * Updates the current belief using the joint probability and the incoming messages
 * @param belief The belief being updated
 * @param src_index The index of the incoming belief message
 * @param edge_index The index of the edge carrying the belief
 * @param joint_probabilities The joint probability matrix
 * @param edge_messages The current message on the edge
 */
__device__
static void send_message_for_edge_iteration_cuda(struct belief *belief, unsigned int src_index, unsigned int edge_index,
                                                 struct joint_probability *joint_probabilities, struct belief *edge_messages){
    unsigned int i, j, num_src, num_dest;
    float sum;
    __shared__ float partial_sums[MAX_STATES * BLOCK_SIZE];

    num_src = joint_probabilities[edge_index].dim_x;
    num_dest = joint_probabilities[edge_index].dim_y;

    sum = 0.0;
    for(i = 0; i < num_src; ++i){
        partial_sums[MAX_STATES * threadIdx.x + i] = 0.0;
        for(j = 0; j < num_dest; ++j){
            partial_sums[MAX_STATES * threadIdx.x + i] += joint_probabilities[edge_index].data[i][j] * belief[src_index].data[j];
        }
        sum += partial_sums[MAX_STATES * threadIdx.x + i];
    }
    if(sum <= 0.0){
        sum = 1.0;
    }
    edge_messages[edge_index].previous = edge_messages[edge_index].current;
    edge_messages[edge_index].current = sum;
    for (i = 0; i < num_src; ++i) {
        edge_messages[edge_index].data[i] /= sum;
    }
}

/**
 * Sends a message along the edge
 * @param num_edges The number of edges in the graph
 * @param edges_src_index The index of the source node for the edge
 * @param node_states The beliefs of all nodes in the graph
 * @param joint_probabilities The joint probabilities of all edges in the graph
 * @param current_edge_messages The current belief held on the edge
 */
__global__
void send_message_for_edge_iteration_cuda_kernel(unsigned int num_edges, unsigned int * edges_src_index,
                                                 struct belief *node_states,
                                                 struct joint_probability *joint_probabilities,
                                                 struct belief *current_edge_messages){
    unsigned int idx, src_node_index;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x){
        src_node_index = edges_src_index[idx];

        send_message_for_edge_iteration_cuda(node_states, src_node_index, idx, joint_probabilities, current_edge_messages);
    }
}

__global__
void send_message_for_edge_iteration_cuda_work_queue_kernel(unsigned int num_edges, unsigned int * edges_src_index,
                                                            struct belief *node_states,
                                                            struct joint_probability *joint_probabilities,
                                                            struct belief *current_edge_messages,
                                                            unsigned int * work_queue_edges, unsigned int * num_work_queue_items) {
    unsigned int i, idx, src_node_index;
    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < *num_work_queue_items; i += blockDim.x * gridDim.x){
        idx = work_queue_edges[i];

        src_node_index = edges_src_index[idx];

        send_message_for_edge_iteration_cuda(node_states, src_node_index, idx, joint_probabilities, current_edge_messages);
    }
}

__global__
void send_message_for_edge_iteration_cuda_work_queue_kernel_streaming(
                                                            unsigned int begin_index, unsigned int end_index,
                                                            unsigned int * edges_src_index,
                                                            struct belief *node_states,
                                                            struct joint_probability *joint_probabilities,
                                                            struct belief *current_edge_messages,
                                                            unsigned int * work_queue_edges, unsigned int * num_work_queue_items) {
    unsigned int i, idx, src_node_index;
    for(i = blockIdx.x * blockDim.x + threadIdx.x + begin_index; i < end_index && i < *num_work_queue_items; i += blockDim.x * gridDim.x){
        idx = work_queue_edges[i];

        src_node_index = edges_src_index[idx];

        send_message_for_edge_iteration_cuda(node_states, src_node_index, idx, joint_probabilities, current_edge_messages);
    }
}

/**
 * Combines messages on the edge
 * @param edge_index The current edge's index
 * @param current_messages The current message to combine
 * @param dest_node_index The destination node's index in the graph
 * @param belief The current belief being buffered
 * @param num_variables The number of states within the belief
 */
__device__
void combine_loopy_edge_cuda(unsigned int edge_index, struct belief *current_messages, unsigned int dest_node_index,
                             struct belief *belief){
    unsigned int i, num_variables;
    unsigned int * address_as_uint;
    unsigned int old, assumed;
    __shared__ float current_message_value[BLOCK_SIZE], current_belief_value[BLOCK_SIZE];

    address_as_uint = (unsigned int *)current_messages;
    num_variables = current_messages[edge_index].size;

    for(i = 0; i < num_variables; ++i){
        current_message_value[threadIdx.x] = current_messages[edge_index].data[i];
        current_belief_value[threadIdx.x] = belief[dest_node_index].data[i];
        if(current_belief_value[threadIdx.x] > 0.0f){
            old = __float_as_uint(current_message_value[threadIdx.x]);
            do{
                assumed = old;
                old = atomicCAS(address_as_uint, assumed, __float_as_uint(current_belief_value[threadIdx.x] * __uint_as_float(assumed)));
            }while(assumed != old);
            belief[dest_node_index].data[i] = current_belief_value[threadIdx.x];
        }
        __syncthreads();
    }
}

/**
 * Combines incoming messages on the edge
 * @param num_edges The number of the edges in thr graph
 * @param edges_dest_index The index of the destination nodes in the graph
 * @param current_edge_messages The current edge message used for buffering
 * @param node_states The current beliefs of all nodes in the graph
 */
__global__
void combine_loopy_edge_cuda_kernel(unsigned int num_edges, unsigned int * edges_dest_index,
                                    struct belief *current_edge_messages, struct belief *node_states){
    unsigned idx, dest_node_index;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x){
        dest_node_index = edges_dest_index[idx];

        combine_loopy_edge_cuda(idx, current_edge_messages, dest_node_index, node_states);
    }
}

__global__
void combine_loopy_edge_cuda_work_queue_kernel(unsigned int num_edges, unsigned int * edges_dest_index,
                                    struct belief *current_edge_messages, struct belief *node_states,
                                               unsigned int * work_queue_edges, unsigned int * num_work_items,
                                               unsigned int * work_queue_scratch){
    unsigned i, idx, dest_node_index;

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < *num_work_items; i += blockDim.x * gridDim.x){
        idx = work_queue_edges[i];

        dest_node_index = edges_dest_index[idx];

        combine_loopy_edge_cuda(idx, current_edge_messages, dest_node_index, node_states);
    }

    __syncthreads();
    update_work_queue_edges_cuda(work_queue_edges, num_work_items, work_queue_scratch, current_edge_messages, num_edges, PRECISION_ITERATION);
}

__global__
void combine_loopy_edge_cuda_work_queue_kernel_streaming(unsigned int begin_index, unsigned int end_index,
                                                         unsigned int * edges_dest_index,
                                               struct belief *current_edge_messages, struct belief *node_states,
                                               unsigned int * work_queue_edges, unsigned int * num_work_items,
                                               unsigned int * work_queue_scratch){
    unsigned i, idx, dest_node_index;

    for(i = blockIdx.x * blockDim.x + threadIdx.x + begin_index; i < end_index && i < *num_work_items; i += blockDim.x * gridDim.x){
        idx = work_queue_edges[i];

        dest_node_index = edges_dest_index[idx];

        combine_loopy_edge_cuda(idx, current_edge_messages, dest_node_index, node_states);
    }
}

/**
 * Marginalizes and normalizes a belief in the graph
 * @param belief The current belief
 * @param num_vertices The number of nodes in the graph
 */
__global__
void marginalize_loop_node_edge_kernel(struct belief *belief, unsigned int num_vertices){
    unsigned int i, idx, num_variables;
    float sum;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices; idx += blockDim.x * gridDim.x){
        num_variables = belief->size;
        sum = 0.0f;
        for(i = 0; i < num_variables; ++i){
            sum += belief->data[i];
        }
        if(sum >= 0.0f){
            for(i = 0; i < num_variables; ++i){
                belief->data[i] /= sum;
            }
        }
    }
}

/**
 * Calculates the delta between the current beliefs and the previous ones
 * @param i The current index of the edge
 * @param current_messages The current messages
 * @return The summed delta
 */
__device__
float calculate_local_delta(unsigned int i, struct belief * current_messages){
    float delta, diff;

    diff = current_messages[i].previous - current_messages[i].current;
    if(diff != diff){
        diff = 0.0;
    }
    delta = (float)fabs(diff);

    return delta;
}

/**
 * Calculates the delta across all messages to test for convergence via parallel reduction
 * @param current_messages The current states
 * @param delta The delta to write
 * @param delta_array Temp array to hold the partial deltas so that they can be reduced
 * @param num_edges The number of edges in the graph
 */
__global__
void calculate_delta(struct belief *current_messages,
                     float * delta, float * delta_array,
                     unsigned int num_edges){
    extern __shared__ float shared_delta[];
    unsigned int tid, idx, i, s;

    tid = threadIdx.x;
    i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x){
        delta_array[idx] = calculate_local_delta(idx, current_messages);
    }
    __syncthreads();

    float my_delta = (i < num_edges) ? delta_array[i] : 0;

    if(i + BLOCK_SIZE < num_edges){
        my_delta += delta_array[i + BLOCK_SIZE];
    }

    shared_delta[tid] = my_delta;
    __syncthreads();

    // do reduction in shared memory
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
 * @brief Calculates the delta across all messages to test for convergence via parallel reduction
 * @details Optimized parallel reduction code borrowed from the CUDA toolkit samples
 * @param current_messages The current states
 * @param delta The delta to write
 * @param delta_array Temp array to hold partial deltas for reduction
 * @param num_edges The number of edges in the graph
 * @param n_is_pow_2 Flag to address padding for shared memory
 * @param warp_size The size of the warp of the GPU
 */
__global__
void calculate_delta_6( struct belief * current_messages,
                       float * delta, float * delta_array,
                       unsigned int num_edges, char n_is_pow_2, unsigned int warp_size) {
    extern __shared__ float shared_delta[];

    unsigned int offset;
    // perform first level of reduce
    // reading from global memory, writing to shared memory
    unsigned int idx;
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int grid_size = blockDim.x * 2 * gridDim.x;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x){
        delta_array[idx] = calculate_local_delta(idx, current_messages);
    }
    __syncthreads();

    float my_delta = 0.0;

    while (i < num_edges) {
        my_delta += delta_array[i];

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
 * Calculates the delta across all messages to test for convergence via parallel reduction
 * @details Simple implementation used for comparison against reduction code
 * @param current_messages The current messages
 * @param delta The delta to write
 * @param delta_array Temp array to hold partial deltas to be used for reduction
 * @param num_edges The number of the edges in the graph
 */
__global__
void calculate_delta_simple(struct belief * current_messages,
                            float * delta, float * delta_array,
                            unsigned int num_edges) {
    extern __shared__ float shared_delta[];
    unsigned int tid, idx, i, s;

    tid = threadIdx.x;
    idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x){
        delta_array[idx] = calculate_local_delta(idx, current_messages);
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

__global__
void marginalize_viterbi_beliefs(struct belief * nodes, unsigned int num_nodes){
    unsigned int idx, i;
    float sum;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_nodes; idx += blockDim.x * gridDim.x){
        sum = 0.0;
        for(i = 0; i < nodes[idx].size; ++i){
            sum += nodes[idx].data[i];
        }
        for(i = 0; i < nodes[idx].size; ++i){
            nodes[idx].data[i] = nodes[idx].data[i] / sum;
        }
    }
}

/**
 * Helper function to test for error with CUDA kernel execution
 */
void test_error(){
    cudaError_t err;

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

/**
 * Runs loopy BP on the GPU
 * @param graph The graph to run
 * @param convergence The convergence threshold; when the delta falls below this threshold, execution will halt
 * @param max_iterations The number of executions to stop at
 * @return The actual number of iterations ran
 */
unsigned int loopy_propagate_until_cuda(Graph_t graph, float convergence, unsigned int max_iterations){
    unsigned int i, j, num_iter, num_vertices, num_edges;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct joint_probability * edges_joint_probabilities;

    struct belief * current_messages;

    struct belief * node_states;

    host_delta = 0.0;

    init_work_queue_nodes(graph);

    struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    unsigned int * dest_node_to_edges_nodes;
    unsigned int * dest_node_to_edges_edges;
    unsigned int * src_node_to_edges_nodes;
    unsigned int * src_node_to_edges_edges;
    unsigned int * work_queue_nodes;
    unsigned int * work_queue_scratch;
    unsigned int * num_work_items;

    num_vertices = graph->current_num_vertices;
    num_edges = graph->current_num_edges;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_nodes, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_edges, sizeof(unsigned int) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_nodes, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_edges, sizeof(unsigned int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_nodes, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_scratch, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&num_work_items, sizeof(unsigned int)));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));


    // copy data
    CUDA_CHECK_RETURN(cudaMemcpy(edges_joint_probabilities, graph->edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges, cudaMemcpyHostToDevice ));

    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) *  graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(unsigned int) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(unsigned int) * num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_nodes, graph->src_nodes_to_edges_node_list, sizeof(unsigned int) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_edges, graph->src_nodes_to_edges_edge_list, sizeof(unsigned int) * num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(work_queue_nodes, graph->work_queue_nodes, sizeof(unsigned int) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(num_work_items, &graph->num_work_items_nodes, sizeof(unsigned int), cudaMemcpyHostToDevice));

    const int edgeCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    const int nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            loopy_propagate_main_loop<<<nodeCount, BLOCK_SIZE >>>(num_vertices, num_edges,
            node_states,
            edges_joint_probabilities,
            current_messages,
            work_queue_nodes, num_work_items,
            work_queue_scratch,
            src_node_to_edges_nodes, src_node_to_edges_edges,
            dest_node_to_edges_nodes, dest_node_to_edges_edges);
            test_error();
            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        test_error();
        CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
        //printf("Current delta: %f\n", host_delta);

        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            break;
        }
        previous_delta = host_delta;
    }

    // copy data back
    CUDA_CHECK_RETURN(cudaMemcpy(graph->node_states, node_states, sizeof(struct belief) * num_vertices, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(graph->edges_messages, current_messages, sizeof(struct belief) * num_edges, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(dest_node_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(dest_node_to_edges_edges));
    CUDA_CHECK_RETURN(cudaFree(src_node_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(src_node_to_edges_edges));

    CUDA_CHECK_RETURN(cudaFree(edges_joint_probabilities));

    CUDA_CHECK_RETURN(cudaFree(current_messages));

    CUDA_CHECK_RETURN(cudaFree(node_states));

    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));

    CUDA_CHECK_RETURN(cudaFree(work_queue_nodes));
    CUDA_CHECK_RETURN(cudaFree(work_queue_scratch));
    CUDA_CHECK_RETURN(cudaFree(num_work_items));

    /*printf("After=====");
    print_nodes(graph);
    print_edges(graph);*/

    return num_iter;
}

static void *launch_init_read_buffer_node_kernels(void *data) {
    struct node_stream_data *stream_data;

    stream_data = (struct node_stream_data *)data;

    int blockCount = stream_data->streamNodeCount;

    init_and_read_message_buffer_cuda_streaming<<<blockCount, BLOCK_SIZE_NODE_STREAMING, 0, stream_data->stream>>>(stream_data->begin_index,
                                                stream_data->end_index, stream_data->buffers, stream_data->node_messages,
    stream_data->current_edge_messages, stream_data->dest_nodes_to_edges_nodes, stream_data->dest_nodes_to_edges_edges,
            stream_data->num_edges, stream_data->num_vertices, stream_data->work_queue_nodes, stream_data->num_work_items);

    cudaStreamSynchronize(stream_data->stream);
    test_error();

    return NULL;
}

static void *launch_write_node_kernels(void *data) {
    struct node_stream_data *stream_data;

    stream_data = (struct node_stream_data *)data;

    int blockCount = stream_data->streamNodeCount;

    send_message_for_node_cuda_streaming_kernel<<<blockCount, BLOCK_SIZE_NODE_STREAMING, 0, stream_data->stream>>>(stream_data->begin_index, stream_data->end_index,
            stream_data->work_queue_nodes, stream_data->num_work_items, stream_data->buffers, stream_data->num_edges,
    stream_data->joint_probabilities, stream_data->current_edge_messages, stream_data->src_nodes_to_edges_nodes,
            stream_data->src_nodes_to_edges_edges, stream_data->num_vertices);

    cudaStreamSynchronize(stream_data->stream);
    test_error();

    return NULL;
}

static void *launch_marginalize_node_kernels(void *data) {
    struct node_stream_data *stream_data;

    stream_data = (struct node_stream_data *)data;

    int blockCount = stream_data->streamNodeCount;

    marginalize_node_cuda_streaming<<<blockCount, BLOCK_SIZE_NODE_STREAMING, 0, stream_data->stream>>>(stream_data->begin_index, stream_data->end_index,
    stream_data->work_queue_nodes, stream_data->num_work_items, stream_data->node_messages,
            stream_data->current_edge_messages, stream_data->dest_nodes_to_edges_nodes,
            stream_data->dest_nodes_to_edges_edges, stream_data->num_vertices, stream_data->num_edges);

    cudaStreamSynchronize(stream_data->stream);
    test_error();

    return NULL;
}

/**
 * Runs loopy BP on the GPU
 * @param graph The graph to run
 * @param convergence The convergence threshold; when the delta falls below this threshold, execution will halt
 * @param max_iterations The number of executions to stop at
 * @return The actual number of iterations ran
 */
unsigned int loopy_propagate_until_cuda_streaming(Graph_t graph, float convergence, unsigned int max_iterations){
    unsigned int i, j, k, num_iter, num_vertices, num_edges;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct joint_probability * edges_joint_probabilities;

    struct belief * current_messages;

    struct belief * node_states;

    struct belief * read_buffer;
    int retval;

    host_delta = 0.0;

    init_work_queue_nodes(graph);

    struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    pthread_t threads[NUM_THREAD_PARTITIONS];
    cudaStream_t streams[NUM_THREAD_PARTITIONS];
    struct node_stream_data thread_data[NUM_THREAD_PARTITIONS];

    unsigned int * dest_node_to_edges_nodes;
    unsigned int * dest_node_to_edges_edges;
    unsigned int * src_node_to_edges_nodes;
    unsigned int * src_node_to_edges_edges;
    unsigned int * work_queue_nodes;
    unsigned int * work_queue_scratch;
    unsigned int * num_work_items;

    num_vertices = graph->current_num_vertices;
    num_edges = graph->current_num_edges;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_nodes, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_edges, sizeof(unsigned int) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_nodes, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_edges, sizeof(unsigned int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_nodes, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_scratch, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&num_work_items, sizeof(unsigned int)));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&read_buffer, sizeof(struct belief) * graph->current_num_vertices));

    // copy data
    CUDA_CHECK_RETURN(cudaMemcpy(edges_joint_probabilities, graph->edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges, cudaMemcpyHostToDevice ));

    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) *  graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(unsigned int) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(unsigned int) * num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_nodes, graph->src_nodes_to_edges_node_list, sizeof(unsigned int) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_edges, graph->src_nodes_to_edges_edge_list, sizeof(unsigned int) * num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(work_queue_nodes, graph->work_queue_nodes, sizeof(unsigned int) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(num_work_items, &graph->num_work_items_nodes, sizeof(unsigned int), cudaMemcpyHostToDevice));

    const int edgeCount = (num_edges + BLOCK_SIZE_NODE_STREAMING - 1)/ BLOCK_SIZE_NODE_STREAMING;
    //const int nodeCount = (num_vertices + BLOCK_SIZE_NODE_STREAMING - 1) / BLOCK_SIZE_NODE_STREAMING;''
    const int partitionSize = (num_vertices + NUM_THREAD_PARTITIONS - 1) / NUM_THREAD_PARTITIONS;
    const int partitionCount = (partitionSize + BLOCK_SIZE_NODE_STREAMING - 1) / BLOCK_SIZE_NODE_STREAMING;

    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE_NODE_STREAMING, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE_NODE_STREAMING <= 32) ? 2 * BLOCK_SIZE_NODE_STREAMING * sizeof(float) : BLOCK_SIZE_NODE_STREAMING * sizeof(float);

    unsigned int curr_index = 0;
    //prepare streams and data
    for(i = 0; i < NUM_THREAD_PARTITIONS; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

        thread_data[i].begin_index = curr_index;
        thread_data[i].end_index = curr_index + partitionSize;
        if(thread_data[i].end_index > num_vertices) {
            thread_data[i].end_index = num_vertices;
        }
        curr_index += partitionSize;
        thread_data[i].streamNodeCount = partitionCount;

        thread_data[i].buffers = read_buffer;
        thread_data[i].num_vertices = num_vertices;
        thread_data[i].num_edges = num_edges;
        thread_data[i].node_messages = node_states;
        thread_data[i].current_edge_messages = current_messages;
        thread_data[i].work_queue_nodes = work_queue_nodes;
        thread_data[i].num_work_items = num_work_items;
        thread_data[i].joint_probabilities = edges_joint_probabilities;
        thread_data[i].work_queue_scratch = work_queue_scratch;
        thread_data[i].src_nodes_to_edges_nodes = src_node_to_edges_nodes;
        thread_data[i].src_nodes_to_edges_edges = src_node_to_edges_edges;
        thread_data[i].dest_nodes_to_edges_nodes = dest_node_to_edges_nodes;
        thread_data[i].dest_nodes_to_edges_edges = dest_node_to_edges_edges;
        thread_data[i].stream = streams[i];
    }

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {

            //init + read data
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_create(&threads[k], NULL, launch_init_read_buffer_node_kernels, &(thread_data[k]));
                if(retval) {
                    fprintf(stderr, "Error creating read thread %d: %d\n", k, retval);
                    return 1;
                }
            }

            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining read thread %d: %d\n", k, retval);
                    return 1;
                }
            }

            //send data
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_create(&threads[k], NULL, launch_write_node_kernels, &(thread_data[k]));
                if(retval) {
                    fprintf(stderr, "Error creating send thread %d: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining write thread %d: %d\n", k, retval);
                    return 1;
                }
            }

            //marginalize
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_create(&threads[k], NULL, launch_marginalize_node_kernels, &(thread_data[k]));
                if(retval) {
                    fprintf(stderr, "Error creating marginalize thread %d: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining marginalize thread %d: %d\n", k, retval);
                    return 1;
                }
            }

            test_error();
            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        test_error();
        CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
        //printf("Current delta: %f\n", host_delta);

        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            break;
        }
        previous_delta = host_delta;
    }

    for(i = 0; i < NUM_THREAD_PARTITIONS; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    // copy data back
    CUDA_CHECK_RETURN(cudaMemcpy(graph->node_states, node_states, sizeof(struct belief) * num_vertices, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(graph->edges_messages, current_messages, sizeof(struct belief) * num_edges, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(dest_node_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(dest_node_to_edges_edges));
    CUDA_CHECK_RETURN(cudaFree(src_node_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(src_node_to_edges_edges));

    CUDA_CHECK_RETURN(cudaFree(edges_joint_probabilities));

    CUDA_CHECK_RETURN(cudaFree(current_messages));

    CUDA_CHECK_RETURN(cudaFree(node_states));

    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));

    CUDA_CHECK_RETURN(cudaFree(read_buffer));

    CUDA_CHECK_RETURN(cudaFree(work_queue_nodes));
    CUDA_CHECK_RETURN(cudaFree(work_queue_scratch));
    CUDA_CHECK_RETURN(cudaFree(num_work_items));

    /*printf("After=====");
    print_nodes(graph);
    print_edges(graph);*/

    return num_iter;
}


/**
 * Runs PageRank on the GPU
 * @param graph The graph to run
 * @param convergence The convergence threshold; when the delta falls below this threshold, execution will halt
 * @param max_iterations The number of executions to stop at
 * @return The actual number of iterations ran
 */
unsigned int page_rank_until_cuda(Graph_t graph, float convergence, unsigned int max_iterations){
    unsigned int i, j, num_iter, num_vertices, num_edges;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct joint_probability * edges_joint_probabilities;

    struct belief * current_messages;

    struct belief * node_states;

    host_delta = 0.0;

    struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    unsigned int * dest_node_to_edges_nodes;
    unsigned int * dest_node_to_edges_edges;
    unsigned int * src_node_to_edges_nodes;
    unsigned int * src_node_to_edges_edges;

    num_vertices = graph->current_num_vertices;
    num_edges = graph->current_num_edges;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_nodes, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_edges, sizeof(unsigned int) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_nodes, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_edges, sizeof(unsigned int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));


    // copy data
    CUDA_CHECK_RETURN(cudaMemcpy(edges_joint_probabilities, graph->edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges, cudaMemcpyHostToDevice ));

    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) *  graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(unsigned int) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(unsigned int) * num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_nodes, graph->src_nodes_to_edges_node_list, sizeof(unsigned int) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_edges, graph->src_nodes_to_edges_edge_list, sizeof(unsigned int) * num_edges, cudaMemcpyHostToDevice));

    const int edgeCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    const int nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            page_rank_main_loop<<<nodeCount, BLOCK_SIZE >>>(num_vertices, num_edges, node_states, edges_joint_probabilities, current_messages, src_node_to_edges_nodes, src_node_to_edges_edges, src_node_to_edges_nodes, src_node_to_edges_edges);
            test_error();
            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        test_error();
        CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
        //printf("Current delta: %f\n", host_delta);

        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            break;
        }
        previous_delta = host_delta;
    }

    // copy data back
    CUDA_CHECK_RETURN(cudaMemcpy(graph->node_states, node_states, sizeof(struct belief) * num_vertices, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(graph->edges_messages, current_messages, sizeof(struct belief) * num_edges, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(dest_node_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(dest_node_to_edges_edges));
    CUDA_CHECK_RETURN(cudaFree(src_node_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(src_node_to_edges_edges));

    CUDA_CHECK_RETURN(cudaFree(edges_joint_probabilities));

    CUDA_CHECK_RETURN(cudaFree(current_messages));

    CUDA_CHECK_RETURN(cudaFree(node_states));

    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));

    /*printf("After=====");
    print_nodes(graph);
    print_edges(graph);*/

    return num_iter;
}


/**
 * Runs Viterbi on the GPU
 * @param graph The graph to run
 * @param convergence The convergence threshold; when the delta falls below this threshold, execution will halt
 * @param max_iterations The number of executions to stop at
 * @return The actual number of iterations ran
 */
unsigned int viterbi_until_cuda(Graph_t graph, float convergence, unsigned int max_iterations){
    unsigned int i, j, num_iter, num_vertices, num_edges;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct joint_probability * edges_joint_probabilities;

    struct belief * current_messages;

    struct belief * node_states;

    host_delta = 0.0;

    struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    unsigned int * dest_node_to_edges_nodes;
    unsigned int * dest_node_to_edges_edges;
    unsigned int * src_node_to_edges_nodes;
    unsigned int * src_node_to_edges_edges;

    num_vertices = graph->current_num_vertices;
    num_edges = graph->current_num_edges;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_nodes, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_edges, sizeof(unsigned int) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_nodes, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_edges, sizeof(unsigned int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));


    // copy data
    CUDA_CHECK_RETURN(cudaMemcpy(edges_joint_probabilities, graph->edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges, cudaMemcpyHostToDevice ));

    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) *  graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(unsigned int) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(unsigned int) * num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_nodes, graph->src_nodes_to_edges_node_list, sizeof(unsigned int) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_edges, graph->src_nodes_to_edges_edge_list, sizeof(unsigned int) * num_edges, cudaMemcpyHostToDevice));

    const int edgeCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    const int nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            viterbi_main_loop<<<nodeCount, BLOCK_SIZE >>>(num_vertices, num_edges, node_states, edges_joint_probabilities, current_messages, src_node_to_edges_nodes, src_node_to_edges_edges, src_node_to_edges_nodes, src_node_to_edges_edges);
            test_error();
            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        test_error();
        CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
        //printf("Current delta: %f\n", host_delta);

        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            marginalize_viterbi_beliefs<<<nodeCount, BLOCK_SIZE >>>(node_states, num_vertices);
            break;
        }
        previous_delta = host_delta;
    }

    // copy data back
    CUDA_CHECK_RETURN(cudaMemcpy(graph->node_states, node_states, sizeof(struct belief) * num_vertices, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(graph->edges_messages, current_messages, sizeof(struct belief) * num_edges, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(dest_node_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(dest_node_to_edges_edges));
    CUDA_CHECK_RETURN(cudaFree(src_node_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(src_node_to_edges_edges));

    CUDA_CHECK_RETURN(cudaFree(edges_joint_probabilities));

    CUDA_CHECK_RETURN(cudaFree(current_messages));

    CUDA_CHECK_RETURN(cudaFree(node_states));

    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));

    /*printf("After=====");
    print_nodes(graph);
    print_edges(graph);*/

    return num_iter;
}

/**
 * Runs the edge-optimized loopy BP code
 * @param graph The graph to use
 * @param convergence The convergence threshold; when the delta falls below this threshold, execution will stop
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations ran
 */
unsigned int loopy_propagate_until_cuda_edge(Graph_t graph, float convergence, unsigned int max_iterations){
    unsigned int i, j, num_iter, num_vertices, num_edges;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct joint_probability * edges_joint_probabilities;
    struct belief * current_messages;
    struct belief * node_states;

    unsigned int * edges_src_index;
    unsigned int * edges_dest_index;
    unsigned int * dest_nodes_to_edges_nodes;
    unsigned int * dest_nodes_to_edges_edges;

    unsigned int * work_queue_edges;
    unsigned int * work_queue_scratch;
    unsigned int * num_work_items;

    init_work_queue_edges(graph);

    host_delta = 0.0;

    struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    num_vertices = graph->current_num_vertices;
    num_edges = graph->current_num_edges;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_src_index, sizeof(unsigned int) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_dest_index, sizeof(unsigned int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_nodes, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_edges, sizeof(unsigned int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_edges, sizeof(unsigned int) * num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_scratch, sizeof(unsigned int) * num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&num_work_items, sizeof(unsigned int)));


    // copy data
    CUDA_CHECK_RETURN(cudaMemcpy(edges_joint_probabilities, graph->edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges, cudaMemcpyHostToDevice ));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(edges_src_index, graph->edges_src_index, sizeof(unsigned int) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(edges_dest_index, graph->edges_dest_index, sizeof(unsigned int) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(unsigned int) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(unsigned int) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(work_queue_edges, graph->work_queue_edges, sizeof(unsigned int) * num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(num_work_items, &num_edges, sizeof(unsigned int), cudaMemcpyHostToDevice));

    const int edgeCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    const int nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            send_message_for_edge_iteration_cuda_work_queue_kernel<<<edgeCount, BLOCK_SIZE >>>(num_edges, edges_src_index, node_states, edges_joint_probabilities, current_messages, work_queue_edges, num_work_items);
            test_error();
            combine_loopy_edge_cuda_work_queue_kernel<<<edgeCount, BLOCK_SIZE>>>(num_edges, edges_dest_index, current_messages, node_states, work_queue_edges, num_work_items, work_queue_scratch);
            test_error();
            //marginalize_loop_node_edge_kernel<<<nodeCount, BLOCK_SIZE>>>(node_states, num_vars, num_vertices);
            marginalize_nodes<<<nodeCount, BLOCK_SIZE>>>(node_states, current_messages,
            dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
            test_error();

            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        test_error();
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

    CUDA_CHECK_RETURN(cudaFree(edges_joint_probabilities));
    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(node_states));

    CUDA_CHECK_RETURN(cudaFree(edges_src_index));
    CUDA_CHECK_RETURN(cudaFree(edges_dest_index));

    CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_edges));

    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));

    CUDA_CHECK_RETURN(cudaFree(work_queue_edges));
    CUDA_CHECK_RETURN(cudaFree(work_queue_scratch));
    CUDA_CHECK_RETURN(cudaFree(num_work_items));

    /*printf("After=====");
    print_nodes(graph);
    print_edges(graph);*/

    return num_iter;
}

static void* launch_send_message_kernel(void * data) {
    struct edge_stream_data * stream_data;

    stream_data = (struct edge_stream_data *)data;

    send_message_for_edge_iteration_cuda_work_queue_kernel_streaming<<<stream_data->streamEdgeCount, BLOCK_SIZE_EDGE_STREAMING, 0, stream_data->stream >>>(stream_data->begin_index, stream_data->end_index, stream_data->edges_src_index, stream_data->node_states, stream_data->joint_probabilities, stream_data->current_edge_messages, stream_data->work_queue_edges, stream_data->num_work_items);

    cudaStreamSynchronize(stream_data->stream);
    test_error();

    return NULL;
}

static void* launch_combine_message_kernel(void * data) {
    struct edge_stream_data * stream_data;

    stream_data = (struct edge_stream_data *)data;

    combine_loopy_edge_cuda_work_queue_kernel_streaming<<<stream_data->streamEdgeCount, BLOCK_SIZE_EDGE_STREAMING, 0, stream_data->stream>>>(stream_data->begin_index, stream_data->end_index, stream_data->edges_dest_index, stream_data->current_edge_messages, stream_data->node_states, stream_data->work_queue_edges, stream_data->num_work_items, stream_data->work_queue_scratch);

    cudaStreamSynchronize(stream_data->stream);
    test_error();

    return NULL;
}

static void* launch_marginalize_streaming_kernel(void * data) {
    struct node_stream_data *stream_data;

    stream_data = (struct node_stream_data *)data;


    marginalize_nodes_streaming<<<stream_data->streamNodeCount, BLOCK_SIZE_NODE_EDGE_STREAMING, 0, stream_data->stream>>>(
            stream_data->begin_index, stream_data->end_index,
            stream_data->node_messages, stream_data->current_edge_messages,
            stream_data->dest_nodes_to_edges_nodes, stream_data->dest_nodes_to_edges_edges, stream_data->num_vertices, stream_data->num_edges);

    cudaStreamSynchronize(stream_data->stream);
    test_error();

    return NULL;
}

__global__
void update_work_queue_cuda_kernel(unsigned int *work_queue_edges, unsigned int *num_work_items, unsigned int* work_queue_scratch,
                                   struct belief *current_messages, unsigned int num_edges) {
    update_work_queue_edges_cuda(work_queue_edges, num_work_items, work_queue_scratch, current_messages, num_edges, PRECISION_ITERATION);
}

/**
 * Runs the edge-optimized loopy BP code
 * @param graph The graph to use
 * @param convergence The convergence threshold; when the delta falls below this threshold, execution will stop
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations ran
 */
unsigned int loopy_propagate_until_cuda_edge_streaming(Graph_t graph, float convergence, unsigned int max_iterations){
    unsigned int i, j, k, num_iter, num_vertices, num_edges;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct joint_probability * edges_joint_probabilities;
    struct belief * current_messages;
    struct belief * node_states;

    unsigned int * edges_src_index;
    unsigned int * edges_dest_index;
    unsigned int * dest_nodes_to_edges_nodes;
    unsigned int * dest_nodes_to_edges_edges;

    unsigned int * work_queue_edges;
    unsigned int * work_queue_scratch;
    unsigned int * num_work_items;

    int retval;

    init_work_queue_edges(graph);

    host_delta = 0.0;

    struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    num_vertices = graph->current_num_vertices;
    num_edges = graph->current_num_edges;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    pthread_t threads[NUM_THREAD_PARTITIONS];
    cudaStream_t streams[NUM_THREAD_PARTITIONS];
    struct edge_stream_data thread_data[NUM_THREAD_PARTITIONS];
    struct node_stream_data node_thread_data[NUM_THREAD_PARTITIONS];

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_src_index, sizeof(unsigned int) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_dest_index, sizeof(unsigned int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_nodes, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_edges, sizeof(unsigned int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_edges, sizeof(unsigned int) * num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_scratch, sizeof(unsigned int) * num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&num_work_items, sizeof(unsigned int)));


    // copy data
    CUDA_CHECK_RETURN(cudaMemcpy(edges_joint_probabilities, graph->edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges, cudaMemcpyHostToDevice ));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(edges_src_index, graph->edges_src_index, sizeof(unsigned int) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(edges_dest_index, graph->edges_dest_index, sizeof(unsigned int) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(unsigned int) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(unsigned int) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(work_queue_edges, graph->work_queue_edges, sizeof(unsigned int) * num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(num_work_items, &num_edges, sizeof(unsigned int), cudaMemcpyHostToDevice));

    const int edgeCount = (num_edges + BLOCK_SIZE_EDGE_STREAMING - 1)/ BLOCK_SIZE_EDGE_STREAMING;
    const int nodeCount = (num_vertices + BLOCK_SIZE_EDGE_STREAMING - 1) / BLOCK_SIZE_EDGE_STREAMING;

    //const int partitionSize = (num_vertices + NUM_THREAD_PARTITIONS - 1) / NUM_THREAD_PARTITIONS;
    //const int partitionCount = (partitionSize + BLOCK_SIZE_NODE_STREAMING - 1) / BLOCK_SIZE_NODE_STREAMING;
    const int edgePartitionSize = (num_edges + NUM_THREAD_PARTITIONS - 1) / NUM_THREAD_PARTITIONS;
    const int edgePartitionCount = (edgePartitionSize + BLOCK_SIZE_EDGE_STREAMING - 1) / BLOCK_SIZE_EDGE_STREAMING;
    const int nodePartitionSize = (num_vertices + NUM_THREAD_PARTITIONS - 1) / NUM_THREAD_PARTITIONS;
    const int nodePartitionCount = (nodePartitionSize + BLOCK_SIZE_NODE_EDGE_STREAMING - 1) / BLOCK_SIZE_NODE_EDGE_STREAMING;

    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE_EDGE_STREAMING, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE_EDGE_STREAMING <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE_EDGE_STREAMING * sizeof(float);

    unsigned int curr_index = 0;
    unsigned int curr_node_index = 0;
    // init streams + data
    for(i = 0; i < NUM_THREAD_PARTITIONS; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

        thread_data[i].streamEdgeCount = edgePartitionCount;
        thread_data[i].begin_index = curr_index;
        thread_data[i].end_index = curr_index + edgePartitionSize;
        if(thread_data[i].end_index > num_edges) {
            thread_data[i].end_index = num_edges;
        }
        curr_index += edgePartitionSize;

        thread_data[i].joint_probabilities = edges_joint_probabilities;
        thread_data[i].num_vertices = num_vertices;
        thread_data[i].num_edges = num_edges;
        thread_data[i].dest_nodes_to_edges_edges = dest_nodes_to_edges_edges;
        thread_data[i].dest_nodes_to_edges_nodes = dest_nodes_to_edges_nodes;
        thread_data[i].edges_src_index = edges_src_index;
        thread_data[i].edges_dest_index = edges_dest_index;
        thread_data[i].num_work_items = num_work_items;
        thread_data[i].work_queue_edges = work_queue_edges;
        thread_data[i].work_queue_scratch = work_queue_scratch;
        thread_data[i].current_edge_messages = current_messages;
        thread_data[i].node_states = node_states;

        thread_data[i].stream = streams[i];

        node_thread_data[i].begin_index = curr_node_index;
        node_thread_data[i].end_index = curr_node_index + nodePartitionSize;
        if(node_thread_data[i].end_index > num_vertices) {
            node_thread_data[i].end_index = num_vertices;
        }
        curr_node_index += nodePartitionSize;
        node_thread_data[i].streamNodeCount = nodePartitionCount;
        node_thread_data[i].stream = streams[i];

        node_thread_data[i].node_messages = node_states;
        node_thread_data[i].current_edge_messages = current_messages;
        node_thread_data[i].dest_nodes_to_edges_nodes = dest_nodes_to_edges_edges;
        node_thread_data[i].dest_nodes_to_edges_edges = dest_nodes_to_edges_edges;
        node_thread_data[i].num_vertices = num_vertices;
        node_thread_data[i].num_edges = num_edges;

        node_thread_data[i].num_work_items = NULL;
        node_thread_data[i].work_queue_scratch = NULL;
        node_thread_data[i].work_queue_nodes = NULL;
        node_thread_data[i].joint_probabilities = NULL;
        node_thread_data[i].src_nodes_to_edges_edges = NULL;
        node_thread_data[i].src_nodes_to_edges_nodes = NULL;
        node_thread_data[i].buffers = NULL;
    }

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_create(&threads[k], NULL, launch_send_message_kernel, &thread_data[k]);
                if(retval) {
                    fprintf(stderr, "Error creating send message thread %d: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining send message thread %d: %d\n", k, retval);
                    return 1;
                }
            }

            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_create(&threads[k], NULL, launch_combine_message_kernel, &thread_data[k]);
                if(retval) {
                    fprintf(stderr, "Error creating combine message thread %d: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining combine message thread %d: %d\n", k, retval);
                    return 1;
                }
            }


            update_work_queue_cuda_kernel<<<edgeCount, BLOCK_SIZE_EDGE_STREAMING>>>(work_queue_edges, num_work_items, work_queue_scratch, current_messages, num_edges);
            test_error();
            //marginalize_loop_node_edge_kernel<<<nodeCount, BLOCK_SIZE>>>(node_states, num_vars, num_vertices);
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_create(&threads[k], NULL, launch_marginalize_streaming_kernel, &node_thread_data[k]);
                if(retval) {
                    fprintf(stderr, "Error creating marginalize node thread %d: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining marginalize node thread %d: %d\n", k, retval);
                    return 1;
                }
            }


            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        test_error();
        CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
        //   printf("Current delta: %f\n", host_delta);

        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            break;
        }
        previous_delta = host_delta;
    }

    for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
        cudaStreamDestroy(streams[k]);
    }

    // copy data back
    CUDA_CHECK_RETURN(cudaMemcpy(graph->node_states, node_states, sizeof(struct belief) * num_vertices, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(graph->edges_messages, current_messages, sizeof(struct belief) * num_edges, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(edges_joint_probabilities));
    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(node_states));

    CUDA_CHECK_RETURN(cudaFree(edges_src_index));
    CUDA_CHECK_RETURN(cudaFree(edges_dest_index));

    CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_edges));

    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));

    CUDA_CHECK_RETURN(cudaFree(work_queue_edges));
    CUDA_CHECK_RETURN(cudaFree(work_queue_scratch));
    CUDA_CHECK_RETURN(cudaFree(num_work_items));

    /*printf("After=====");
    print_nodes(graph);
    print_edges(graph);*/

    return num_iter;
}


/**
 * Runs the edge-optimized PageRank code
 * @param graph The graph to use
 * @param convergence The convergence threshold; when the delta falls below this threshold, execution will stop
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations ran
 */
unsigned int page_rank_until_cuda_edge(Graph_t graph, float convergence, unsigned int max_iterations){
    unsigned int i, j, num_iter, num_vertices, num_edges;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct joint_probability * edges_joint_probabilities;
    struct belief * current_messages;
    struct belief * node_states;

    unsigned int * edges_src_index;
    unsigned int * edges_dest_index;
    unsigned int * dest_nodes_to_edges_nodes;
    unsigned int * dest_nodes_to_edges_edges;

    host_delta = 0.0;

    struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    num_vertices = graph->current_num_vertices;
    num_edges = graph->current_num_edges;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_src_index, sizeof(unsigned int) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_dest_index, sizeof(unsigned int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_nodes, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_edges, sizeof(unsigned int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));


    // copy data
    CUDA_CHECK_RETURN(cudaMemcpy(edges_joint_probabilities, graph->edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges, cudaMemcpyHostToDevice ));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(edges_src_index, graph->edges_src_index, sizeof(unsigned int) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(edges_dest_index, graph->edges_dest_index, sizeof(unsigned int) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(unsigned int) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(unsigned int) * graph->current_num_edges, cudaMemcpyHostToDevice));

    const int edgeCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    const int nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            send_message_for_edge_iteration_cuda_kernel<<<edgeCount, BLOCK_SIZE >>>(num_edges, edges_src_index, node_states, edges_joint_probabilities, current_messages);
            test_error();
            combine_loopy_edge_cuda_kernel<<<edgeCount, BLOCK_SIZE>>>(num_edges, edges_dest_index, current_messages, node_states);
            test_error();
            //marginalize_loop_node_edge_kernel<<<nodeCount, BLOCK_SIZE>>>(node_states, num_vars, num_vertices);
            marginalize_page_rank_nodes<<<nodeCount, BLOCK_SIZE>>>(node_states, current_messages,
                    dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
            test_error();

            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        test_error();
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

    CUDA_CHECK_RETURN(cudaFree(edges_joint_probabilities));
    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(node_states));

    CUDA_CHECK_RETURN(cudaFree(edges_src_index));
    CUDA_CHECK_RETURN(cudaFree(edges_dest_index));

    CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_edges));

    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));

    /*printf("After=====");
    print_nodes(graph);
    print_edges(graph);*/

    return num_iter;
}


/**
 * Runs the edge-optimized Viterbi code
 * @param graph The graph to use
 * @param convergence The convergence threshold; when the delta falls below this threshold, execution will stop
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations ran
 */
unsigned int viterbi_until_cuda_edge(Graph_t graph, float convergence, unsigned int max_iterations){
    unsigned int i, j, num_iter, num_vertices, num_edges;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct joint_probability * edges_joint_probabilities;
    struct belief * current_messages;
    struct belief * node_states;

    unsigned int * edges_src_index;
    unsigned int * edges_dest_index;
    unsigned int * dest_nodes_to_edges_nodes;
    unsigned int * dest_nodes_to_edges_edges;

    host_delta = 0.0;

    struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    num_vertices = graph->current_num_vertices;
    num_edges = graph->current_num_edges;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_src_index, sizeof(unsigned int) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_dest_index, sizeof(unsigned int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_nodes, sizeof(unsigned int) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_edges, sizeof(unsigned int) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));


    // copy data
    CUDA_CHECK_RETURN(cudaMemcpy(edges_joint_probabilities, graph->edges_joint_probabilities, sizeof(struct joint_probability) * graph->current_num_edges, cudaMemcpyHostToDevice ));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(edges_src_index, graph->edges_src_index, sizeof(unsigned int) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(edges_dest_index, graph->edges_dest_index, sizeof(unsigned int) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(unsigned int) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(unsigned int) * graph->current_num_edges, cudaMemcpyHostToDevice));

    const int edgeCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    const int nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            send_message_for_edge_iteration_cuda_kernel<<<edgeCount, BLOCK_SIZE >>>(num_edges, edges_src_index, node_states, edges_joint_probabilities, current_messages);
            test_error();
            combine_loopy_edge_cuda_kernel<<<edgeCount, BLOCK_SIZE>>>(num_edges, edges_dest_index, current_messages, node_states);
            test_error();
            //marginalize_loop_node_edge_kernel<<<nodeCount, BLOCK_SIZE>>>(node_states, num_vars, num_vertices);
            argmax_nodes<<<nodeCount, BLOCK_SIZE>>>(node_states, current_messages,
                    dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
            test_error();

            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        test_error();
        CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
        //   printf("Current delta: %f\n", host_delta);

        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            marginalize_viterbi_beliefs<<<nodeCount, BLOCK_SIZE >>>(node_states, num_vertices);
            break;
        }
        previous_delta = host_delta;
    }

    // copy data back
    CUDA_CHECK_RETURN(cudaMemcpy(graph->node_states, node_states, sizeof(struct belief) * num_vertices, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(graph->edges_messages, current_messages, sizeof(struct belief) * num_edges, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(edges_joint_probabilities));
    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(node_states));

    CUDA_CHECK_RETURN(cudaFree(edges_src_index));
    CUDA_CHECK_RETURN(cudaFree(edges_dest_index));

    CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_nodes));
    CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_edges));

    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));

    /*printf("After=====");
    print_nodes(graph);
    print_edges(graph);*/

    return num_iter;
}

/**
 * Runs loopy BP and outputs the result
 * @param expression The BNF expression holding the graph
 * @param file_name The file name of the graph data
 * @param out The file handle for the CSV file
 */
void run_test_loopy_belief_propagation_cuda(struct expression * expression, const char * file_name, FILE * out){
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
    //calculate_diameter(graph);

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

/**
 * Runs loopy BP on the XML file
 * @param file_name The name of the XML file
 * @param out The file handle for the CSV file to output to
 */
void run_test_loopy_belief_propagation_xml_file_cuda(const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

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

/**
 * Runs loopy BP on the XML file
 * @param file_name The name of the XML file
 * @param out The file handle for the CSV file to output to
 */
void run_test_loopy_belief_propagation_xml_file_cuda_streaming(const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_cuda_streaming(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy-streaming,%d,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

/**
 * Runs the edge-optimized version of loopy BP
 * @param file_name The path of the file to read
 * @param out The file handle for the CSV output
 */
void run_test_loopy_belief_propagation_xml_file_edge_cuda(const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_cuda_edge(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy-edge,%d,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

/**
 * Runs the edge-optimized version of loopy BP
 * @param file_name The path of the file to read
 * @param out The file handle for the CSV output
 */
void run_test_loopy_belief_propagation_xml_file_edge_cuda_streaming(const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_cuda_edge_streaming(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy-edge-streaming,%d,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}



/**
 * Runs loopy BP on the XML file
 * @param edge_file_name The file to read for the SNAP edges
 * @param node_file_name The file to read for the SNAP observed nodes
 * @param out The CSV file to output to
 */
void run_test_loopy_belief_propagation_snap_file_cuda(const char * edge_file_name, const char * node_file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    graph = parse_graph_from_snap_files(edge_file_name, node_file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_cuda(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy,%d,%d,%d,%d,%lf\n", edge_file_name, node_file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

/**
 * Runs the edge-optimized version of loopy BP
 * @param edge_file_name The file to read for the SNAP edges
 * @param node_file_name The file to read for the SNAP observed nodes
 * @param out The CSV file to output to
 */
void run_test_loopy_belief_propagation_snap_file_edge_cuda(const char * edge_file_name, const char * node_file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    graph = parse_graph_from_snap_files(edge_file_name, node_file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_cuda_edge(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy-edge,%d,%d,%d,%d,%lf\n", edge_file_name, node_file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}


void run_test_loopy_belief_propagation_mtx_files_cuda(const char * edge_mtx, const char *node_mtx, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    graph = build_graph_from_mtx(edge_mtx, node_mtx);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_cuda(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy,%d,%d,%d,%d,%lf\n", edge_mtx, node_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_mtx_files_cuda_streaming(const char * edge_mtx, const char *node_mtx, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    graph = build_graph_from_mtx(edge_mtx, node_mtx);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_cuda_streaming(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy-streaming,%d,%d,%d,%d,%lf\n", edge_mtx, node_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_mtx_files_edge_cuda(const char * edge_mtx, const char * node_mtx, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    graph = build_graph_from_mtx(edge_mtx, node_mtx);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_cuda_edge(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy-edge,%d,%d,%d,%d,%lf\n", edge_mtx, node_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_mtx_files_edge_cuda_streaming(const char * edge_mtx, const char * node_mtx, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    graph = build_graph_from_mtx(edge_mtx, node_mtx);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_cuda_edge_streaming(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy-edge-streaming,%d,%d,%d,%d,%lf\n", edge_mtx, node_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

/**
 * Checks that the CUDA kernel completed
 * @param file The source code file
 * @param line The line within the source code file that executes the kernel
 * @param statement The name of the kernel
 * @param err The error message
 */
void CheckCudaErrorAux (const char *file, unsigned int line, const char *statement, cudaError_t err)
{
    if (err == cudaSuccess) {
        return;
    }
    printf("%s returned %s (%d) at %s:%d\n", statement, cudaGetErrorString(err), err, file, line);
    exit (1);
}

