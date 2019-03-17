#include "belief-propagation.hpp"

static MPI_Datatype joint_probability_struct, belief_struct;
__constant__ struct joint_probability edge_joint_probability[1];

__device__ __forceinline__ unsigned int LaneMaskLt()
{
    unsigned int ret;
    asm ("mov.u32 %0, %%lanemask_lt;" : "=r"(ret) );
    return ret;
}



__device__
unsigned long long int atomic_add_inc(unsigned long long int * __restrict__ ctr) {
    // from https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
    unsigned long long int active = __activemask();
    unsigned long long int leader = __ffs(active) - 1;
    unsigned long long int change = __popc(active);
    unsigned long long int rank = __popc(active & LaneMaskLt());
    unsigned long long int warp_res;
    if(rank == 0) {
        warp_res = atomicAdd(ctr, change);
    }
    warp_res = __shfl_sync(active, warp_res, leader);
    return (warp_res + rank);
}


__device__
void update_work_queue_nodes_cuda(size_t * __restrict__ work_queue_nodes, unsigned long long int * __restrict__ num_work_items,
        size_t * __restrict__ work_queue_scratch, const float * __restrict__ node_states_current,
        const float * __restrict__ node_states_previous, size_t num_vertices, float precision) {
    size_t i, index;
    unsigned long long int orig_num_work_items = *num_work_items;

    atomicCAS(num_work_items, orig_num_work_items, 0);
    __syncthreads();

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < *num_work_items && i < num_vertices; i += blockDim.x * gridDim.x){
        index = work_queue_nodes[i];
        if(index < num_vertices && *num_work_items < num_vertices && fabs(node_states_current[index] - node_states_previous[index]) >= precision) {
            work_queue_scratch[*num_work_items] = work_queue_nodes[i];
            atomic_add_inc(num_work_items);
        }
    }

    __syncthreads();
    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vertices; i += blockDim.x * gridDim.x){
        work_queue_nodes[i] = work_queue_scratch[i];
    }
}

__device__
void update_work_queue_edges_cuda(size_t * __restrict__ work_queue_edge, unsigned long long int * __restrict__ num_work_items,
        size_t * __restrict__ work_queue_scratch, const float * __restrict__ edge_states_previous,
        const float * __restrict__ edge_states_current, size_t num_edges, float precision) {
    size_t i, index;
    unsigned long long int orig_num_work_items = *num_work_items;

    atomicCAS(num_work_items, orig_num_work_items, 0);
    __syncthreads();

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < num_edges && *num_work_items < num_edges; i += blockDim.x * gridDim.x){
        index = work_queue_edge[i];
        if(index < num_edges && *num_work_items < num_edges && fabs(edge_states_current[index] - edge_states_previous[index]) >= precision) {
            work_queue_scratch[*num_work_items] = work_queue_edge[i];
            atomic_add_inc(num_work_items);
        }
    }

    __syncthreads();
    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < num_edges; i += blockDim.x * gridDim.x){
        work_queue_edge[i] = work_queue_scratch[i];
    }
}

/**
 * Initialize the message buffer to what is stored in node_states
 * @param buffer The message buffer
 * @param node_states The states to init to
 * @param num_variables The size of the arrays
 * @param node_index The index of the current belief
  */
__device__
void init_message_buffer_cuda(struct belief * __restrict__ buffer, const  struct belief * __restrict__ node_states, const size_t num_variables, size_t node_index){
    size_t j;

    for(j = 0; j < num_variables; ++j){
        buffer->data[j] = node_states[node_index].data[j];
    }

}

__global__
void init_and_read_message_buffer_cuda_streaming(
        size_t begin_index, size_t end_index,
        struct belief * __restrict__ buffers, const struct belief * __restrict__ node_states, const size_t num_variables,
                                                 const struct belief * __restrict__ previous_messages,
                                                 const size_t * __restrict__ dest_nodes_to_edges_nodes,
                                                 const size_t * __restrict__ dest_nodes_to_edges_edges,
                                                 size_t current_num_edges,
                                                 size_t num_vertices,
        const size_t * __restrict__ work_queue, const unsigned long long int * __restrict__ num_work_queue_items) {
    size_t i, node_index;

    for(i = blockIdx.x * blockDim.x + threadIdx.x + begin_index; i < end_index && i < *num_work_queue_items; i += blockDim.x * gridDim.x){
        node_index = work_queue[i];

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
void combine_message_cuda(struct belief * __restrict__ dest, const struct belief * __restrict__ edge_messages, const size_t length, const size_t offset){
    size_t i;
    const float *buffer = dest->data;
    const float *messages = edge_messages[offset].data;

    for(i = 0; i < length; ++i){
       // if(messages[i] == messages[i]){
       assert(messages[i] == messages[i]);
            dest->data[i] = buffer[i] * messages[i];
        //}
    }
}

__device__
void combine_message_cuda_node_streaming(struct belief * __restrict__ dest, const struct belief * __restrict__ edge_messages, const size_t length, const size_t offset){
    size_t i;
    const float *buffer = dest->data;
    const float *messages = edge_messages[offset].data;

    for(i = 0; i < length; ++i){
        assert(messages[i] == messages[i]);
        //if(messages[i] == messages[i]){
            dest->data[i] = buffer[i] * messages[i];
        //}
    }
}

__device__
void combine_message_cuda_edge_streaming(struct belief * __restrict__ dest, const struct belief * __restrict__ edge_messages, const size_t length, const size_t offset){
    size_t i;
    const float *buffer = dest->data;
    const float *messages = edge_messages[offset].data;

    for(i = 0; i < length; ++i){
        assert(messages[i] == messages[i]);
        //if(messages[i] == messages[i]){
            dest->data[i] = buffer[i] * messages[i];
        //}
    }
}

__device__
void combine_page_rank_message_cuda(struct belief * __restrict__ dest, const struct belief * __restrict__ edge_messages, const size_t length, const size_t offset){
    size_t i;
    const float * buffer = dest->data;
    const float * messages = edge_messages[offset].data;

    for(i = 0; i < length; ++i){
        assert(messages[i] == messages[i]);
        //if(messages[i] == messages[i]){
            dest->data[i] = buffer[i] + messages[i];
        //}
    }
}

__device__
void combine_viterbi_message_cuda(struct belief * __restrict__ dest, const struct belief * __restrict__ edge_messages, const size_t length, const size_t offset){
    size_t i;

    const float *buffer;
    const float *messages;

    buffer = dest->data;
    messages = edge_messages[offset].data;

    for(i = 0; i < length; ++i){
        assert(messages[i] == messages[i]);
        //if(messages[i] == messages[i]){
            dest->data[i] = fmaxf(buffer[i], messages[i]);
       // }
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
void read_incoming_messages_cuda(struct belief * __restrict__ message_buffer,
                                 const struct belief * __restrict__ previous_messages,
                                 const size_t * __restrict__ dest_nodes_to_edges_nodes,
                                 const size_t * __restrict__ dest_nodes_to_edges_edges,
                                 const size_t current_num_edges,
                            const size_t num_vertices, const size_t num_variables, const size_t idx){
    size_t start_index, end_index, j, edge_index;

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
void send_message_for_edge_cuda(const struct belief * __restrict__  buffer, size_t edge_index,
                                const size_t num_src, const size_t num_dest,
                                struct belief * __restrict__ edge_messages,
                                float * __restrict__ edge_messages_previous,
                                float * __restrict__ edge_messages_current){
    size_t i, j;

    float sums, partial_sums, joint_prob, belief_prob;
    const float  * __restrict__  belief_data;


    belief_data = buffer->data;

    sums = 0.0f;
    for(i = 0; i < num_src; ++i){
        partial_sums = 0.0f;
        for(j = 0; j < num_dest; ++j){
            joint_prob = edge_joint_probability[0].data[i][j];
            belief_prob = belief_data[j];
            partial_sums +=  joint_prob * belief_prob;
        }
        edge_messages[edge_index].data[i] = partial_sums;
        sums += partial_sums;
    }
    if(sums <= 0.0f){
        sums = 1.0f;
    }
    edge_messages_previous[edge_index] = edge_messages_current[edge_index];
    edge_messages_current[edge_index] = sums;
    for(i = 0; i < num_src; ++i){
        edge_messages[edge_index].data[i] /= sums;
    }
}

__device__
void send_message_for_edge_cuda_streaming(const struct belief * __restrict__ buffer, const size_t edge_index,
                            const size_t num_src, const size_t num_dest,
                                struct belief * __restrict__ edge_messages,
                                float * edge_messages_previous,
                                float * edge_messages_current){
    size_t i, j;
    float sums, partial_sums;
    const float * belief_data;

    belief_data = buffer->data;

    sums = 0.0f;
    for(i = 0; i < num_src; ++i){
        partial_sums = 0.0f;
        for(j = 0; j < num_dest; ++j){
            partial_sums += edge_joint_probability[0].data[i][j] * belief_data[j];
        }
        edge_messages[edge_index].data[i] = partial_sums;
        sums += partial_sums;
    }
    if(sums <= 0.0f){
        sums = 1.0f;
    }
    edge_messages_previous[edge_index] = edge_messages_current[edge_index];
    edge_messages_current[edge_index] = sums;
    for(i = 0; i < num_src; ++i){
        edge_messages[edge_index].data[i] /= sums;
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
void send_message_for_node_cuda(const struct belief * __restrict__ message_buffer, size_t current_num_edges,
                                const size_t num_src,
                                const size_t num_dest,
                                struct belief * __restrict__ current_edge_messages,
                                float * __restrict__ edge_messages_previous,
                                float * __restrict__ edge_messages_current,
                                const size_t * __restrict__ src_nodes_to_edges_nodes,
                                const size_t * __restrict__ src_nodes_to_edges_edges,
                                const size_t num_vertices, const size_t idx){
    size_t start_index, end_index, j, edge_index;

    start_index = src_nodes_to_edges_nodes[idx];
    if(idx + 1 >= num_vertices){
        end_index = current_num_edges;
    }
    else{
        end_index = src_nodes_to_edges_nodes[idx + 1];
    }

    for(j = start_index; j < end_index; ++j){
        edge_index = src_nodes_to_edges_edges[j];
        send_message_for_edge_cuda(message_buffer, edge_index, num_src, num_dest,
                current_edge_messages, edge_messages_previous, edge_messages_current);
    }
}

__device__
void send_message_for_node_cuda_streaming(const struct belief * __restrict__ message_buffer, size_t current_num_edges,
                                const size_t num_src, const size_t num_dest,
                                struct belief * __restrict__ current_edge_messages,
                                float * __restrict__ edge_messages_previous,
                                float * __restrict__ edge_messages_current,
                                const size_t * __restrict__ src_nodes_to_edges_nodes,
                                const size_t * __restrict__ src_nodes_to_edges_edges,
                                const size_t num_vertices, const size_t idx){
    size_t start_index, end_index, j, edge_index;

    start_index = src_nodes_to_edges_nodes[idx];
    if(idx + 1 >= num_vertices){
        end_index = current_num_edges;
    }
    else{
        end_index = src_nodes_to_edges_nodes[idx + 1];
    }

    for(j = start_index; j < end_index; ++j){
        edge_index = src_nodes_to_edges_edges[j];
        send_message_for_edge_cuda_streaming(message_buffer, edge_index, num_src, num_dest, current_edge_messages, edge_messages_previous,
                edge_messages_current);
    }
}

__global__
void
//__launch_bounds__(BLOCK_SIZE_NODE_STREAMING, MIN_BLOCKS_PER_MP)
send_message_for_node_cuda_streaming_kernel(size_t begin_index, size_t end_index,
                                          const size_t * __restrict__ work_queue, const unsigned long long int * __restrict__ num_work_queue_items,
                                          const struct belief * __restrict__ message_buffers, size_t current_num_edges,
                                          const size_t num_src, const size_t num_dest,
                                          struct belief * __restrict__ current_edge_messages,
                                          float * edge_messages_previous,
                                          float * edge_messages_current,
                                          const size_t * __restrict__ src_nodes_to_edges_nodes, const size_t * __restrict__ src_nodes_to_edges_edges,
                                          const size_t num_vertices) {
    size_t i, node_index;

    for(i = blockIdx.x * blockDim.x + threadIdx.x + begin_index; i < end_index && i < *num_work_queue_items; i += blockDim.x * gridDim.x) {
        node_index = work_queue[i];

        send_message_for_node_cuda_streaming(&(message_buffers[node_index]), current_num_edges,
                                   num_src, num_dest,
                                   current_edge_messages, edge_messages_previous, edge_messages_current,
                                   src_nodes_to_edges_nodes, src_nodes_to_edges_edges,
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
void marginalize_node(struct belief * __restrict__ node_states, const size_t num_variables, const size_t idx,
                      const struct belief * __restrict__ current_edges_messages,
                      const size_t * __restrict__ dest_nodes_to_edges_nodes,
                      const size_t * __restrict__ dest_nodes_to_edges_edges,
                      const size_t num_vertices, const size_t num_edges){
    size_t i, start_index, end_index, edge_index;
    float sum;
    float *new_belief_data, *node_states_data;

    struct belief new_belief;
    new_belief_data = new_belief.data;
    node_states_data = node_states[idx].data;

    for(i = 0; i < num_variables; ++i){
        new_belief_data[i] = 1.0f;
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
    sum = 0.0f;
    if(start_index < end_index){
        for(i = 0; i < num_variables; ++i){
             new_belief_data[i] *= node_states_data[i];
             sum += new_belief_data[i];
        }
    }
    if(sum <= 0.0f){
        sum = 1.0f;
    }
    for(i = 0; i < num_variables; ++i){
        node_states_data[i] /= sum;
    }
}

__device__
void marginalize_node_node_streaming(struct belief * __restrict__ node_states, const size_t num_variables, const size_t idx,
                      const struct belief * __restrict__ current_edges_messages,
                      const size_t * __restrict__ dest_nodes_to_edges_nodes,
                      const size_t * __restrict__ dest_nodes_to_edges_edges,
                      const size_t num_vertices, const size_t num_edges){
    size_t i, start_index, end_index, edge_index;
    float sum;

    struct belief new_belief;

    for(i = 0; i < num_variables; ++i){
        new_belief.data[i] = 1.0f;
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
    sum = 0.0f;
    for(i = 0; i < num_variables; ++i){
        sum += new_belief.data[i];
    }
    if(sum <= 0.0f){
        sum = 1.0f;
    }
    for(i = 0; i < num_variables; ++i){
        node_states[idx].data[i] /= sum;
    }
}

__device__
void marginalize_node_edge_streaming(struct belief * __restrict__ node_states, const size_t num_variables, size_t idx,
                      const struct belief * __restrict__ current_edges_messages,
                      const size_t * __restrict__ dest_nodes_to_edges_nodes,
                      const size_t * __restrict__ dest_nodes_to_edges_edges,
                      size_t num_vertices, size_t num_edges){
    size_t i, start_index, end_index, edge_index;
    float sum;

    struct belief new_belief;

    for(i = 0; i < num_variables; ++i){
        new_belief.data[i] = 1.0f;
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
    sum = 0.0f;
    for(i = 0; i < num_variables; ++i){
        sum += new_belief.data[i];
    }
    if(sum <= 0.0f){
        sum = 1.0f;
    }
    for(i = 0; i < num_variables; ++i){
        node_states[idx].data[i] /= sum;
    }
}

__global__
void marginalize_node_cuda_streaming( size_t begin_index, size_t end_index,
                                const size_t * __restrict__ work_queue,
                                const unsigned long long int * __restrict__ num_work_queue_items,
                                struct belief * __restrict__ node_states,
                                const size_t node_states_size,
                                const struct belief * __restrict__ current_edges_messages,
                                const size_t * __restrict__  dest_nodes_to_edges_nodes, const size_t * __restrict__ dest_nodes_to_edges_edges,
                                const size_t num_vertices, const size_t num_edges) {
    size_t i, node_index;

    for(i = blockIdx.x * blockDim.x + threadIdx.x + begin_index; i < end_index && i < *num_work_queue_items; i += blockDim.x * gridDim.x) {
        node_index = work_queue[i];

        marginalize_node_node_streaming(node_states, node_states_size, node_index, current_edges_messages, dest_nodes_to_edges_nodes,
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
void marginalize_page_rank_node(struct belief * __restrict__ node_states, const size_t num_variables, const size_t idx,
                                const struct belief * __restrict__ current_edges_messages,
                                const size_t * __restrict__ dest_nodes_to_edges_nodes,
                                const size_t * __restrict__ dest_nodes_to_edges_edges,
                                const size_t num_vertices, const size_t num_edges) {
    size_t i, start_index, end_index, edge_index;
    float factor;

    struct belief new_belief;

    for (i = 0; i < num_variables; ++i) {
        new_belief.data[i] = 0.0f;
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
void argmax_node(struct belief * __restrict__ node_states, const size_t num_variables,  size_t idx,
                      const struct belief * __restrict__ current_edges_messages,
                      const size_t * __restrict__ dest_nodes_to_edges_nodes, const size_t * __restrict__ dest_nodes_to_edges_edges,
                      size_t num_vertices, size_t num_edges){
    size_t i, start_index, end_index, edge_index;

    struct belief new_belief;

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
void marginalize_nodes(struct belief * __restrict__ node_states, const size_t  num_variables,
                       const struct belief * __restrict__ current_edges_messages,
                       const size_t * __restrict__ dest_nodes_to_edges_nodes,
                       const size_t * __restrict__ dest_nodes_to_edges_edges,
                       const size_t num_vertices, const size_t num_edges) {
    size_t idx;
    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices; idx += blockDim.x * gridDim.x){
        marginalize_node(node_states, num_variables, idx, current_edges_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
    }
}

__global__
void marginalize_nodes_streaming(const size_t begin_index, const size_t end_index, struct belief * __restrict__ node_states, const size_t node_states_size,
                                 const struct belief * __restrict__ current_edges_messages,
                                 const size_t * __restrict__ dest_nodes_to_edges_nodes, const size_t * __restrict__ dest_nodes_to_edges_edges,
                                 const size_t num_vertices, const size_t num_edges) {
    size_t idx;
    for (idx = blockIdx.x * blockDim.x + threadIdx.x + begin_index; idx < end_index && idx < num_vertices; idx += blockDim.x * gridDim.x) {
        marginalize_node_edge_streaming(node_states, node_states_size, idx, current_edges_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges,
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
void marginalize_page_rank_nodes(struct belief * __restrict__ node_states, const size_t node_states_size,
                       const struct belief * __restrict__ current_edges_messages,
                       const size_t * __restrict__ dest_nodes_to_edges_nodes, const size_t * __restrict__ dest_nodes_to_edges_edges,
                       const size_t num_vertices, const size_t num_edges) {
    size_t idx;
    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices; idx += blockDim.x * gridDim.x){
        marginalize_page_rank_node(node_states, node_states_size, idx, current_edges_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
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
void argmax_nodes(struct belief * __restrict__ node_states, const size_t node_states_size,
                       const struct belief * __restrict__ current_edges_messages,
                       const size_t * __restrict__ dest_nodes_to_edges_nodes, const size_t * __restrict__ dest_nodes_to_edges_edges,
                       const size_t num_vertices, const size_t num_edges) {
    size_t idx;
    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices; idx += blockDim.x * gridDim.x){
        argmax_node(node_states, node_states_size, idx, current_edges_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
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
void loopy_propagate_main_loop(size_t num_vertices, size_t num_edges,
                               struct belief * __restrict__ node_messages,
                               const size_t num_variables,
                               float * __restrict__ node_messages_previous,
                               float * __restrict__ node_messages_current,
                               const size_t num_src, const size_t num_dest,
                               struct belief *current_edge_messages,
                               float * edge_messages_previous,
                               float * edge_messages_current,
                               size_t * __restrict__ work_queue_nodes, unsigned long long int * __restrict__ num_work_items,
                               size_t * __restrict__ work_queue_scratch,
                               const size_t * __restrict__ src_nodes_to_edges_nodes, const size_t * __restrict__ src_nodes_to_edges_edges,
                               const size_t * __restrict__ dest_nodes_to_edges_nodes, const size_t * __restrict__ dest_nodes_to_edges_edges) {
    size_t i, idx;
    struct belief new_belief;

    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < *num_work_items; i += blockDim.x * gridDim.x) {
        idx = work_queue_nodes[i];

        init_message_buffer_cuda(&new_belief, node_messages, num_variables, idx);
        __syncthreads();

        read_incoming_messages_cuda(&new_belief, current_edge_messages, dest_nodes_to_edges_nodes,
                                    dest_nodes_to_edges_edges, num_edges, num_vertices, num_variables, idx);
        __syncthreads();

        send_message_for_node_cuda(&new_belief, num_edges, num_src, num_dest, current_edge_messages, edge_messages_previous, edge_messages_current,
                                   src_nodes_to_edges_nodes, src_nodes_to_edges_edges, num_vertices, idx);
        __syncthreads();
        marginalize_node(node_messages, num_variables, idx, current_edge_messages, dest_nodes_to_edges_nodes,
                         dest_nodes_to_edges_edges, num_vertices, num_edges);

        __syncthreads();
    }
    update_work_queue_nodes_cuda(work_queue_nodes, num_work_items, work_queue_scratch, node_messages_previous, node_messages_current, num_vertices, PRECISION_ITERATION);

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
void page_rank_main_loop(size_t num_vertices, size_t num_edges,
                               struct belief * __restrict__ node_messages,
                                 const size_t num_variables,
                                 float * __restrict__ node_messages_previous,
                                 float * __restrict__ node_messages_current,
                               size_t num_src, size_t num_dest,
                         struct belief *current_edge_messages,
                         float * edge_messages_previous,
                         float * edge_messages_current,
                               const size_t * __restrict__ src_nodes_to_edges_nodes, const size_t * __restrict__ src_nodes_to_edges_edges,
                               const size_t * __restrict__ dest_nodes_to_edges_nodes, const size_t * __restrict__ dest_nodes_to_edges_edges){
    size_t idx;
    struct belief new_belief;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices; idx += blockDim.x * gridDim.x){

        init_message_buffer_cuda(&new_belief, node_messages, num_variables, idx);
        __syncthreads();

        read_incoming_messages_cuda(&new_belief, current_edge_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_edges, num_vertices, num_variables, idx);
        __syncthreads();

        send_message_for_node_cuda(&new_belief, num_edges, num_src, num_dest,
                current_edge_messages, edge_messages_previous, edge_messages_current, src_nodes_to_edges_nodes, src_nodes_to_edges_edges, num_vertices, idx);
        __syncthreads();

        marginalize_page_rank_node(node_messages, num_variables, idx, current_edge_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
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
void viterbi_main_loop(size_t num_vertices, size_t num_edges,
                         struct belief * __restrict__ node_messages,
                       const size_t num_variables,
                       float * __restrict__ node_messages_previous,
                       float * __restrict__ node_messages_current,
                         size_t num_src, size_t num_dest,
                       struct belief *current_edge_messages,
                       float * edge_messages_previous,
                       float * edge_messages_current,
                         const size_t * __restrict__ src_nodes_to_edges_nodes, const size_t * __restrict__ src_nodes_to_edges_edges,
                         const size_t * __restrict__ dest_nodes_to_edges_nodes, const size_t * __restrict__ dest_nodes_to_edges_edges){
    size_t idx;
    struct belief new_belief;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices; idx += blockDim.x * gridDim.x){

        init_message_buffer_cuda(&new_belief, node_messages, num_variables, idx);
        __syncthreads();

        read_incoming_messages_cuda(&new_belief, current_edge_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_edges, num_vertices, num_variables, idx);
        __syncthreads();

        send_message_for_node_cuda(&new_belief, num_edges, num_src, num_dest, current_edge_messages, edge_messages_previous, edge_messages_current,
                src_nodes_to_edges_nodes, src_nodes_to_edges_edges, num_vertices, idx);
        __syncthreads();

        argmax_node(node_messages, num_variables, idx, current_edge_messages, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
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
void send_message_for_edge_iteration_cuda(const struct belief * __restrict__ buffer, const size_t src_index, const size_t edge_index,
                                                 const size_t num_src, const size_t num_dest,
                                                         struct belief * __restrict__ edge_messages,
                                                                 float * edge_messages_previous, float * edge_messages_current){
    size_t i, j;

    float sums, partial_sums, joint_prob, belief_prob;
    const float *belief_data;

    belief_data = buffer->data;

    sums = 0.0f;
    for(i = 0; i < num_src; ++i){
        partial_sums = 0.0f;
        for(j = 0; j < num_dest; ++j){
            joint_prob = edge_joint_probability[0].data[i][j];
            belief_prob = belief_data[j];
            partial_sums += joint_prob * belief_prob;
        }
        edge_messages[edge_index].data[i] = partial_sums;
        sums += partial_sums;
    }
    if(sums <= 0.0f){
        sums = 1.0f;
    }
    edge_messages_previous[edge_index] = edge_messages_current[edge_index];
    edge_messages_current[edge_index] = sums;
    for (i = 0; i < num_src; ++i) {
        edge_messages[edge_index].data[i] /= sums;
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
void send_message_for_edge_iteration_cuda_kernel(const size_t num_edges, const size_t * __restrict__ edges_src_index,
                                                 const struct belief * __restrict__ node_states,
                                                 const size_t num_src, const size_t num_dest,
                                                 struct belief * __restrict__ current_edge_messages,
                                                 float * edge_messages_previous, float * edge_messages_current){
    size_t idx, src_node_index;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x){
        src_node_index = edges_src_index[idx];

        send_message_for_edge_iteration_cuda(node_states, src_node_index, idx, num_src, num_dest, current_edge_messages,
                edge_messages_previous, edge_messages_current);
    }
}

__global__
void send_message_for_edge_iteration_cuda_work_queue_kernel(const size_t num_edges, const size_t * __restrict__ edges_src_index,
                                                            const struct belief * __restrict__ node_states,
                                                            const size_t num_src, const size_t num_dest,
                                                            struct belief * __restrict__ current_edge_messages,
                                                            float * edge_messages_previous, float * edge_messages_current,
                                                            size_t * __restrict__ work_queue_edges, unsigned long long int * __restrict__ num_work_queue_items) {
    size_t i, idx, src_node_index;
    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < *num_work_queue_items; i += blockDim.x * gridDim.x){
        idx = work_queue_edges[i];

        src_node_index = edges_src_index[idx];

        send_message_for_edge_iteration_cuda(node_states, src_node_index, idx, num_src, num_dest, current_edge_messages, edge_messages_previous, edge_messages_current);
    }
}

__global__
void send_message_for_edge_iteration_cuda_work_queue_kernel_streaming(
                                                            const size_t begin_index, const size_t end_index,
                                                            const size_t * __restrict__ edges_src_index,
                                                            const struct belief * __restrict__ node_states,
                                                            const size_t num_src, const size_t num_dest,
                                                            struct belief * __restrict__ current_edge_messages,
                                                            float * edge_messages_previous, float * edge_messages_current,
                                                            const size_t * __restrict__ work_queue_edges, const unsigned long long int *  __restrict__ num_work_queue_items) {
    size_t i, idx, src_node_index;
    for(i = blockIdx.x * blockDim.x + threadIdx.x + begin_index; i < end_index && i < *num_work_queue_items; i += blockDim.x * gridDim.x){
        idx = work_queue_edges[i];

        src_node_index = edges_src_index[idx];

        send_message_for_edge_iteration_cuda(node_states, src_node_index, idx, num_src, num_dest,
                current_edge_messages, edge_messages_previous, edge_messages_current);
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
void combine_loopy_edge_cuda(const size_t edge_index, const struct belief * __restrict__ current_messages, const size_t num_variables, const size_t dest_node_index,
                             struct belief * __restrict__ belief){
    size_t i;
    unsigned long long int * address_as_uint;
    unsigned long long int old, assumed;
    __shared__ float current_message_value[BLOCK_SIZE], current_belief_value[BLOCK_SIZE];

    address_as_uint = (unsigned long long int *)current_messages;
    if(num_variables > MAX_STATES || threadIdx.x >= BLOCK_SIZE) {
        return;
    }

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
void combine_loopy_edge_cuda_kernel(size_t num_edges, const size_t * __restrict__ edges_dest_index,
                                    const struct belief * __restrict__ current_edge_messages,
                                            const size_t num_variables,
                                            struct belief * __restrict__ node_states){
    size_t idx, dest_node_index;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x){
        dest_node_index = edges_dest_index[idx];

        combine_loopy_edge_cuda(idx, current_edge_messages, num_variables, dest_node_index, node_states);
    }
}

__global__
void combine_loopy_edge_cuda_work_queue_kernel(size_t num_edges, const size_t * __restrict__ edges_dest_index,
                                    const belief * current_edge_messages,
                                    const float * __restrict__ current_edge_message_previous,
                                    const float * __restrict__ current_edge_message_current,
                                            const size_t num_variables,
                                            struct belief * __restrict__ node_states,
                                               size_t * __restrict__ work_queue_edges, unsigned long long int * __restrict__ num_work_items,
                                               size_t * __restrict__ work_queue_scratch){
    size_t i, idx, dest_node_index;

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < *num_work_items; i += blockDim.x * gridDim.x){
        idx = work_queue_edges[i];

        dest_node_index = edges_dest_index[idx];

        combine_loopy_edge_cuda(idx, current_edge_messages, num_variables, dest_node_index, node_states);
    }

    __syncthreads();
    update_work_queue_edges_cuda(work_queue_edges, num_work_items, work_queue_scratch,
                                 current_edge_message_previous, current_edge_message_current, num_edges, PRECISION_ITERATION);
}

__global__
void combine_loopy_edge_cuda_work_queue_kernel_streaming(size_t begin_index, size_t end_index,
                                                         const size_t * __restrict__ edges_dest_index,
                                               const struct belief * __restrict__ current_edge_messages,
                                               const size_t num_variables,
                                               struct belief * __restrict__ node_states,
                                               const size_t * __restrict__ work_queue_edges, const unsigned long long int * __restrict__ num_work_items,
                                               const size_t * __restrict__ work_queue_scratch){
    size_t i, idx, dest_node_index;

    for(i = blockIdx.x * blockDim.x + threadIdx.x + begin_index; i < end_index && i < *num_work_items; i += blockDim.x * gridDim.x){
        idx = work_queue_edges[i];

        dest_node_index = edges_dest_index[idx];

        combine_loopy_edge_cuda(idx, current_edge_messages, num_variables, dest_node_index, node_states);
    }
}

/**
 * Marginalizes and normalizes a belief in the graph
 * @param belief The current belief
 * @param num_vertices The number of nodes in the graph
 */
__global__
void marginalize_loop_node_edge_kernel(struct belief * __restrict__ belief, const size_t num_variables, const size_t num_vertices){
    size_t i, idx;
    float sum;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices; idx += blockDim.x * gridDim.x){
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
float calculate_local_delta(size_t i, const float * __restrict__ current_messages_previous, const float * __restrict__ current_messages_current){
    float delta, diff;

    diff = current_messages_previous[i] - current_messages_current[i];
    if(diff != diff){
        diff = 0.0f;
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
void calculate_delta(const float * __restrict__ current_messages_previous, const float * __restrict__ current_messages_current,
                     float * __restrict__ delta, float * __restrict__ delta_array,
                     const size_t num_edges){
    extern __shared__ float shared_delta[];
    size_t tid, idx, i, s;

    tid = threadIdx.x;
    i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x){
        delta_array[idx] = calculate_local_delta(idx, current_messages_previous, current_messages_current);
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
            my_delta += __shfl_down_sync(FULL_MASK, my_delta, s);
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
void calculate_delta_6(const float * __restrict__ current_messages_previous,
                       const float * __restrict__ current_messages_current,
                       float * __restrict__ delta, float * __restrict__ delta_array,
                       const size_t num_edges, char n_is_pow_2, int warp_size) {
    extern __shared__ float shared_delta[];

    size_t offset;
    // perform first level of reduce
    // reading from global memory, writing to shared memory
    size_t idx;
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    size_t grid_size = blockDim.x * 2 * gridDim.x;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x){
        delta_array[idx] = calculate_local_delta(idx, current_messages_previous, current_messages_current);
    }
    __syncthreads();

    float my_delta = 0.0f;

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
            my_delta += __shfl_down_sync(FULL_MASK, my_delta, offset);
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
void calculate_delta_simple(const float * __restrict__ current_messages_previous,
                            const float * __restrict__ current_messages_current,
                            float * __restrict__ delta, float * __restrict__ delta_array,
                            const size_t num_edges) {
    extern __shared__ float shared_delta[];
    size_t tid, idx, i, s;

    tid = threadIdx.x;
    idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x){
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

__global__
void marginalize_viterbi_beliefs(struct belief * nodes, const size_t num_variables, const size_t num_nodes){
    size_t idx, i;
    float sum;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_nodes; idx += blockDim.x * gridDim.x){
        sum = 0.0f;
        for(i = 0; i < num_variables; ++i){
            sum += nodes[idx].data[i];
        }
        for(i = 0; i < num_variables; ++i){
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
int loopy_propagate_until_cuda(Graph_t graph, const float convergence, const int max_iterations){
    int num_iter;
    size_t i, j;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct belief * current_messages;
    float * current_messages_previous;
    float * current_messages_current;

    struct belief * node_states;
    float * node_states_previous;
    float * node_states_current;

    struct belief * read_buffer;

    host_delta = 0.0f;
    previous_delta = INFINITY;

    init_work_queue_nodes(graph);

    //struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    size_t * dest_node_to_edges_nodes;
    size_t * dest_node_to_edges_edges;
    size_t * src_node_to_edges_nodes;
    size_t * src_node_to_edges_edges;
    size_t * work_queue_nodes;
    size_t * work_queue_scratch;
    unsigned long long int * num_work_items;

    const size_t num_vertices = graph->current_num_vertices;
    const size_t num_edges = graph->current_num_edges;
    const size_t edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    const size_t edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;
    const size_t node_states_size = graph->node_states_size;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_nodes, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_edges, sizeof(size_t) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_nodes, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_edges, sizeof(size_t) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_previous, sizeof(float) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_current, sizeof(float) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states_previous, sizeof(float) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states_current, sizeof(float) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_nodes, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_scratch, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&num_work_items, sizeof(unsigned long long int)));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&read_buffer, sizeof(struct belief) * graph->current_num_vertices));

    unsigned long long int h_num_work_items = (unsigned long long int)graph->num_work_items_nodes;

    // copy data
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(edge_joint_probability, &(graph->edge_joint_probability), sizeof(struct joint_probability)));

    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_previous, graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_current, graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) *  graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states_previous, graph->node_states_previous, sizeof(float) *  graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states_current, graph->node_states_current, sizeof(float) *  graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(size_t) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(size_t) * num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_nodes, graph->src_nodes_to_edges_node_list, sizeof(size_t) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_edges, graph->src_nodes_to_edges_edge_list, sizeof(size_t) * num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(work_queue_nodes, graph->work_queue_nodes, sizeof(size_t) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(num_work_items, &h_num_work_items, sizeof(unsigned long long int), cudaMemcpyHostToDevice));

    const int edgeCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    const int nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            loopy_propagate_main_loop << < nodeCount, BLOCK_SIZE >> > (num_vertices, num_edges,
                    node_states, node_states_size,
                    node_states_previous, node_states_current,
                    edge_joint_probability_dim_x,
                    edge_joint_probability_dim_y,
                    current_messages,
                    current_messages_previous,
                    current_messages_current,
                    work_queue_nodes, num_work_items,
                    work_queue_scratch,
                    src_node_to_edges_nodes, src_node_to_edges_edges,
                    dest_node_to_edges_nodes, dest_node_to_edges_edges);
            test_error();
            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages_previous, current_messages_current, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
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

    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(current_messages_current));
    CUDA_CHECK_RETURN(cudaFree(current_messages_previous));

    CUDA_CHECK_RETURN(cudaFree(node_states));
    CUDA_CHECK_RETURN(cudaFree(node_states_previous));
    CUDA_CHECK_RETURN(cudaFree(node_states_current));

    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));

    CUDA_CHECK_RETURN(cudaFree(work_queue_nodes));
    CUDA_CHECK_RETURN(cudaFree(work_queue_scratch));
    CUDA_CHECK_RETURN(cudaFree(num_work_items));

    //CUDA_CHECK_RETURN(cudaHostUnregister(graph->node_states));
    //CUDA_CHECK_RETURN(cudaHostUnregister(graph->edges_messages));

    /*printf("After=====");
    print_nodes(graph);
    print_edges(graph);*/

    return num_iter;
}

static void *launch_init_read_buffer_node_kernels(void *data) {
    struct node_stream_data *stream_data;

    stream_data = (struct node_stream_data *)data;

    size_t blockCount = stream_data->streamNodeCount;

    init_and_read_message_buffer_cuda_streaming<<<blockCount, BLOCK_SIZE_NODE_STREAMING, 0, stream_data->stream>>>(stream_data->begin_index,
                                                stream_data->end_index, stream_data->buffers, stream_data->node_messages, stream_data->node_messages_size,
    stream_data->current_edge_messages, stream_data->dest_nodes_to_edges_nodes, stream_data->dest_nodes_to_edges_edges,
            stream_data->num_edges, stream_data->num_vertices, stream_data->work_queue_nodes, stream_data->num_work_items);

    cudaStreamSynchronize(stream_data->stream);
    test_error();

    return NULL;
}

static void *launch_write_node_kernels(void *data) {
    struct node_stream_data *stream_data;

    stream_data = (struct node_stream_data *)data;

    size_t blockCount = stream_data->streamNodeCount;

    send_message_for_node_cuda_streaming_kernel<<<blockCount, BLOCK_SIZE_NODE_STREAMING, 0, stream_data->stream>>>(stream_data->begin_index, stream_data->end_index,
            stream_data->work_queue_nodes, stream_data->num_work_items, stream_data->buffers, stream_data->num_edges,
    stream_data->edge_joint_probability_dim_x, stream_data->edge_joint_probability_dim_y, stream_data->current_edge_messages,
    stream_data->current_edge_messages_previous, stream_data->current_edge_messages_current,
    stream_data->src_nodes_to_edges_nodes,
            stream_data->src_nodes_to_edges_edges, stream_data->num_vertices);

    cudaStreamSynchronize(stream_data->stream);
    test_error();

    return NULL;
}

static void *launch_marginalize_node_kernels(void *data) {
    struct node_stream_data *stream_data;

    stream_data = (struct node_stream_data *)data;

    size_t blockCount = stream_data->streamNodeCount;

    marginalize_node_cuda_streaming<<<blockCount, BLOCK_SIZE_NODE_STREAMING, 0, stream_data->stream>>>(stream_data->begin_index, stream_data->end_index,
    stream_data->work_queue_nodes, stream_data->num_work_items, stream_data->node_messages, stream_data->node_messages_size,
            stream_data->current_edge_messages, stream_data->dest_nodes_to_edges_nodes,
            stream_data->dest_nodes_to_edges_edges, stream_data->num_vertices, stream_data->num_edges);

    cudaStreamSynchronize(stream_data->stream);
    test_error();

    return NULL;
}

static
__global__ void update_work_queue_nodes_cuda_kernel(size_t * work_queue_nodes, unsigned long long int * num_work_items, size_t * work_queue_scratch, const float * node_messages_previous, const float * node_messages_current, size_t num_vertices) {
    update_work_queue_nodes_cuda(work_queue_nodes, num_work_items, work_queue_scratch, node_messages_previous, node_messages_current, num_vertices, PRECISION_ITERATION);
}

int loopy_propagate_until_cuda_multiple_devices(Graph_t graph, const float convergence, const int max_iterations){
    int num_iter;
    size_t i, j, k, l, m, n, start_index, end_index;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct belief ** current_messages;
    float ** current_messages_previous;
    float ** current_messages_current;

    struct belief ** h_current_messages;
    float ** h_current_messages_previous;
    float ** h_current_messages_current;

    struct belief ** node_states;
    float ** node_states_previous;
    float ** node_states_current;


    struct belief ** read_buffer;
    float ** read_buffer_previous;
    float ** read_buffer_current;

    struct belief ** h_read_buffer;
    float ** h_read_buffer_previous;
    float ** h_read_buffer_current;

    struct belief * my_buffer;

    int retval;

    host_delta = 0.0f;
    previous_delta = INFINITY;

    init_work_queue_nodes(graph);

    int num_devices = -1;
    cudaGetDeviceCount(&num_devices);
    assert(num_devices >= 1);

    //struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * num_devices);
    assert(threads);
    cudaStream_t *streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * num_devices);
    assert(streams);
    struct node_stream_data *thread_data = (struct node_stream_data *)malloc(sizeof(struct node_stream_data) * num_devices);
    assert(thread_data);

    size_t ** dest_node_to_edges_nodes;
    size_t ** dest_node_to_edges_edges;
    size_t ** src_node_to_edges_nodes;
    size_t ** src_node_to_edges_edges;
    size_t ** work_queue_nodes;
    size_t ** work_queue_scratch;
    unsigned long long int ** num_work_items;

    const size_t num_vertices = graph->current_num_vertices;
    const size_t num_edges = graph->current_num_edges;
    const size_t edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    const size_t edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;

    const size_t node_states_size = graph->node_states_size;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // init buffers

    h_read_buffer = (struct belief **)malloc(sizeof(struct belief *) * num_devices);
    assert(h_read_buffer);
    h_read_buffer_current = (float **)malloc(sizeof(float *) * num_devices);
    assert(h_read_buffer_current);
    h_read_buffer_previous = (float **)malloc(sizeof(float *) * num_devices);
    assert(h_read_buffer_previous);

    h_current_messages = (struct belief **)malloc(sizeof(struct belief *) * num_devices);
    assert(h_current_messages);
    h_current_messages_current = (float **)malloc(sizeof(float *) * num_devices);
    assert(h_current_messages_current);
    h_current_messages_previous = (float **)malloc(sizeof(float *) * num_devices);
    assert(h_current_messages_previous);

    current_messages = (struct belief **)malloc(sizeof(struct belief *) * num_devices);
    assert(current_messages);
    current_messages_current = (float **)malloc(sizeof(float *) * num_devices);
    assert(current_messages_current);
    current_messages_previous = (float **)malloc(sizeof(float *) * num_devices);
    assert(current_messages_previous);

    read_buffer = (struct belief **)malloc(sizeof(struct belief *) * num_devices);
    assert(read_buffer);
    read_buffer_current = (float **)malloc(sizeof(float *) * num_devices);
    assert(read_buffer_current);
    read_buffer_previous = (float **)malloc(sizeof(float *) * num_devices);
    assert(read_buffer_previous);

    node_states = (struct belief **)malloc(sizeof(struct belief *) * num_devices);
    assert(node_states);
    node_states_previous = (float **)malloc(sizeof(float *) * num_devices);
    assert(node_states_previous);
    node_states_current = (float **)malloc(sizeof(float *) * num_devices);
    assert(node_states_current);

    dest_node_to_edges_nodes = (size_t **)malloc(sizeof(size_t) * num_devices);
    assert(dest_node_to_edges_nodes);
    dest_node_to_edges_edges = (size_t **)malloc(sizeof(size_t) * num_devices);
    assert(dest_node_to_edges_edges);
    src_node_to_edges_nodes = (size_t **)malloc(sizeof(size_t) * num_devices);
    assert(src_node_to_edges_nodes);
    src_node_to_edges_edges = (size_t **)malloc(sizeof(size_t) * num_devices);
    assert(src_node_to_edges_edges);
    work_queue_nodes = (size_t **)malloc(sizeof(size_t) * num_devices);
    assert(work_queue_nodes);
    work_queue_scratch = (size_t **)malloc(sizeof(size_t) * num_devices);
    assert(work_queue_scratch);
    num_work_items = (unsigned long long int **)malloc(sizeof(unsigned long long int) * num_devices);
    assert(num_work_items);

    my_buffer = (struct belief *)malloc(sizeof(struct belief) * graph->current_num_vertices);
    assert(my_buffer);

    unsigned long long int h_num_work_items = graph->num_work_items_nodes;

    // pin host memory
    CUDA_CHECK_RETURN(cudaHostRegister(graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(graph->work_queue_nodes, sizeof(size_t) * graph->current_num_vertices, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(&h_num_work_items, sizeof(unsigned long long int), cudaHostRegisterDefault));


    for(k = 0; k < num_devices; ++k) {
        cudaSetDevice(k);

        // allocate data
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(dest_node_to_edges_nodes[k]), sizeof(size_t) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(dest_node_to_edges_edges[k]), sizeof(size_t) * graph->current_num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(src_node_to_edges_nodes[k]), sizeof(size_t) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(src_node_to_edges_edges[k]), sizeof(size_t) * graph->current_num_edges));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &(current_messages[k]), sizeof(struct belief) * graph->current_num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(current_messages_current[k]), sizeof(float) * graph->current_num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(current_messages_previous[k]), sizeof(float) * graph->current_num_edges));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &(node_states[k]), sizeof(struct belief) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(node_states_previous[k]), sizeof(float) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(node_states_current[k]), sizeof(float) * graph->current_num_vertices));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &(work_queue_nodes[k]), sizeof(size_t) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(work_queue_scratch[k]), sizeof(size_t) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(num_work_items[k]), sizeof(unsigned long long int)));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &(read_buffer[k]), sizeof(struct belief) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(read_buffer_previous[k]), sizeof(float) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(read_buffer_current[k]), sizeof(float) * graph->current_num_vertices));

        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(h_read_buffer[k]), sizeof(struct belief) * graph->current_num_vertices, cudaHostAllocDefault));
        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(h_read_buffer_previous[k]), sizeof(float) * graph->current_num_vertices, cudaHostAllocDefault));
        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(h_read_buffer_current[k]), sizeof(float) * graph->current_num_vertices, cudaHostAllocDefault));

        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(h_current_messages[k]), sizeof(struct belief) * graph->current_num_edges, cudaHostAllocDefault));
        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(h_current_messages_previous[k]), sizeof(float) * graph->current_num_edges, cudaHostAllocDefault));
        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(h_current_messages_current[k]), sizeof(float) * graph->current_num_edges, cudaHostAllocDefault));
    }

    CUDA_CHECK_RETURN(cudaMalloc((void **) &delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &delta_array, sizeof(float) * num_edges));

    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(edge_joint_probability, &(graph->edge_joint_probability), sizeof(struct joint_probability)));

    for(k = 0; k < num_devices; ++k) {

        CUDA_CHECK_RETURN(
                cudaMemcpy(current_messages[k], graph->edges_messages, sizeof(struct belief) * graph->current_num_edges,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(current_messages_current[k], graph->edges_messages_current, sizeof(float) * graph->current_num_edges,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(current_messages_previous[k], graph->edges_messages_previous, sizeof(float) * graph->current_num_edges,
                           cudaMemcpyHostToDevice));

        CUDA_CHECK_RETURN(
                cudaMemcpy(node_states[k], graph->node_states, sizeof(struct belief) * graph->current_num_vertices,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(node_states_current[k], graph->node_states_current, sizeof(float) * graph->current_num_vertices,
                           cudaMemcpyHostToDevice)
        );
        CUDA_CHECK_RETURN(
                cudaMemcpy(node_states_previous[k], graph->node_states_previous, sizeof(float) * graph->current_num_vertices,
                           cudaMemcpyHostToDevice));

        CUDA_CHECK_RETURN(
                cudaMemcpy(dest_node_to_edges_nodes[k], graph->dest_nodes_to_edges_node_list, sizeof(size_t) * num_vertices,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(dest_node_to_edges_edges[k], graph->dest_nodes_to_edges_edge_list, sizeof(size_t) * num_edges,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(src_node_to_edges_nodes[k], graph->src_nodes_to_edges_node_list, sizeof(size_t) * num_vertices,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(src_node_to_edges_edges[k], graph->src_nodes_to_edges_edge_list, sizeof(size_t) * num_edges,
                           cudaMemcpyHostToDevice));

        CUDA_CHECK_RETURN(cudaMemcpy(work_queue_nodes[k], graph->work_queue_nodes, sizeof(size_t) * num_vertices,
                                     cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(num_work_items[k], &h_num_work_items, sizeof(unsigned long long int), cudaMemcpyHostToDevice));
    }

    const size_t partitionSize = (num_vertices + num_devices - 1) / num_devices;
    const size_t partitionCount = (partitionSize + BLOCK_SIZE_NODE_STREAMING - 1) / BLOCK_SIZE_NODE_STREAMING;

    const size_t edgeCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    const size_t nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;

    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE_NODE_STREAMING, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE_NODE_STREAMING <= 32) ? 2 * BLOCK_SIZE_NODE_STREAMING * sizeof(float) : BLOCK_SIZE_NODE_STREAMING * sizeof(float);

    size_t curr_index = 0;
    //prepare streams and data
    for(i = 0; i < num_devices; ++i) {
        CUDA_CHECK_RETURN(cudaSetDevice(i));
        CUDA_CHECK_RETURN(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

        thread_data[i].begin_index = curr_index;
        thread_data[i].end_index = curr_index + partitionSize;
        if(thread_data[i].end_index > num_vertices) {
            thread_data[i].end_index = num_vertices;
        }
        curr_index += partitionSize;
        thread_data[i].streamNodeCount = partitionCount;

        thread_data[i].buffers = read_buffer[i];
        thread_data[i].num_vertices = num_vertices;
        thread_data[i].num_edges = num_edges;
        thread_data[i].node_messages = node_states[i];
        thread_data[i].node_messages_size = node_states_size;
        thread_data[i].current_edge_messages = current_messages[i];
        thread_data[i].current_edge_messages_current = current_messages_current[i];
        thread_data[i].current_edge_messages_previous = current_messages_previous[i];
        thread_data[i].work_queue_nodes = work_queue_nodes[i];
        thread_data[i].num_work_items = num_work_items[i];
        thread_data[i].edge_joint_probability_dim_x = edge_joint_probability_dim_x;
        thread_data[i].edge_joint_probability_dim_y = edge_joint_probability_dim_y;
        thread_data[i].work_queue_scratch = work_queue_scratch[i];
        thread_data[i].src_nodes_to_edges_nodes = src_node_to_edges_nodes[i];
        thread_data[i].src_nodes_to_edges_edges = src_node_to_edges_edges[i];
        thread_data[i].dest_nodes_to_edges_nodes = dest_node_to_edges_nodes[i];
        thread_data[i].dest_nodes_to_edges_edges = dest_node_to_edges_edges[i];
        thread_data[i].stream = streams[i];
    }

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {

            //init + read data
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_create(&threads[k], NULL, launch_init_read_buffer_node_kernels, &(thread_data[k]));
                if(retval) {
                    fprintf(stderr, "Error creating read thread %ld: %d\n", k, retval);
                    return 1;
                }
            }

            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining read thread %ld: %d\n", k, retval);
                    return 1;
                }

            }

            //read data back
            // synchronize state

            // first get data back from devices
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                for(m = 0; m < num_devices; ++m) {
                    if(m == k) {
                        continue;
                    }
                    // update beliefs
                    for (l = thread_data[k].begin_index;
                         l < thread_data[k].end_index && l < graph->num_work_items_nodes; ++l) {
                        curr_index = graph->work_queue_nodes[l];
                        if (curr_index < graph->current_num_vertices) {
                            CUDA_CHECK_RETURN(cudaMemcpy(&(h_read_buffer[m][curr_index]), &(h_read_buffer[k][curr_index]), sizeof(struct belief), cudaMemcpyDeviceToDevice));
                        }
                    }
                }
            }

            //send data
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_create(&threads[k], NULL, launch_write_node_kernels, &(thread_data[k]));
                if(retval) {
                    fprintf(stderr, "Error creating send thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining write thread %ld: %d\n", k, retval);
                    return 1;
                }
            }

            // send to gpu
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                for(m = 0; m < num_devices; ++m) {
                    if(m == k) {
                        continue;
                    }
                    // update beliefs
                    for (l = thread_data[k].begin_index;
                         l < thread_data[k].end_index && l < graph->num_work_items_nodes; ++l) {
                        if (curr_index < graph->current_num_vertices) {
                            curr_index = graph->work_queue_nodes[l];
                            start_index = graph->src_nodes_to_edges_node_list[curr_index];
                            if(start_index+1 >= graph->current_num_vertices) {
                                end_index = graph->current_num_edges;
                            }
                            else {
                                end_index = graph->src_nodes_to_edges_edge_list[curr_index + 1];
                            }

                            for(n = start_index; n < end_index; ++n) {
                                CUDA_CHECK_RETURN(
                                        cudaMemcpy(&(h_read_buffer[m][n]), &(h_read_buffer[k][n]),
                                                   sizeof(struct belief), cudaMemcpyDeviceToDevice));
                                CUDA_CHECK_RETURN(cudaMemcpy(&(current_messages[m][n]), &(current_messages[k][n]),
                                                             sizeof(struct belief),
                                                             cudaMemcpyDeviceToDevice));
                                CUDA_CHECK_RETURN(
                                        cudaMemcpy(&(current_messages_previous[m][n]), &(current_messages_previous[k][n]),
                                                   sizeof(float), cudaMemcpyDeviceToDevice));
                                CUDA_CHECK_RETURN(cudaMemcpy(&(current_messages_current[m][n]), &(current_messages_current[k][n]),
                                                             sizeof(float),
                                                             cudaMemcpyDeviceToDevice));
                            }
                        }
                    }
                }
            }

            //marginalize
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_create(&threads[k], NULL, launch_marginalize_node_kernels, &(thread_data[k]));
                if(retval) {
                    fprintf(stderr, "Error creating marginalize thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining marginalize thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            // copy back
            // send to gpu
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                for(m = 0; m < num_devices; ++m) {
                    if(m == k) {
                        continue;
                    }
                    l = thread_data[k].begin_index;
                    if(thread_data[k].end_index > graph->current_num_vertices) {
                        n = graph->current_num_vertices - l;
                    }
                    else {
                        n = thread_data[k].end_index - l;
                    }

                    CUDA_CHECK_RETURN(cudaMemcpy(&(node_states[m][l]), &(node_states[k][l]),
                                                 sizeof(struct belief) * n,
                                                 cudaMemcpyDeviceToDevice));
                    CUDA_CHECK_RETURN(cudaMemcpy(&(node_states_previous[m][l]), &(node_states_previous[k][l]),
                                                 sizeof(float) * n,
                                                 cudaMemcpyDeviceToDevice));
                    CUDA_CHECK_RETURN(cudaMemcpy(&(node_states_current[m][l]), &(node_states_current[k][l]),
                                                 sizeof(float) * n,
                                                 cudaMemcpyDeviceToDevice));

                }
            }

            for(k = 0; k < num_devices; ++k) {
                update_work_queue_nodes_cuda_kernel<<<nodeCount, BLOCK_SIZE>>>(work_queue_nodes[k], num_work_items[k], work_queue_scratch[k], node_states_previous[k], node_states_current[k], graph->current_num_vertices);
                test_error();
                num_iter++;
            }
            //copy back
            CUDA_CHECK_RETURN(cudaSetDevice(0));
            CUDA_CHECK_RETURN(cudaMemcpy(graph->work_queue_nodes, work_queue_nodes[0], sizeof(size_t) * graph->current_num_vertices, cudaMemcpyDeviceToHost));
            CUDA_CHECK_RETURN(cudaMemcpy(&(graph->num_work_items_nodes), num_work_items[0], sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        }

        CUDA_CHECK_RETURN(cudaSetDevice(0));
        calculate_delta_6 << < dimReduceGrid, dimReduceBlock, reduceSmemSize >> >
                                                              (current_messages_previous[0], current_messages_current[0], delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
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

    for(i = 0; i < num_devices; ++i) {
        CUDA_CHECK_RETURN(cudaSetDevice(i));
        cudaStreamDestroy(streams[i]);
    }

    CUDA_CHECK_RETURN(cudaSetDevice(0));
    // copy data back
    CUDA_CHECK_RETURN(cudaMemcpy(graph->node_states, node_states[0], sizeof(struct belief) * num_vertices, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(graph->edges_messages, current_messages[0], sizeof(struct belief) * num_edges, cudaMemcpyDeviceToHost));

    for(i = 0; i < num_devices; ++i) {
        CUDA_CHECK_RETURN(cudaSetDevice(i));

        CUDA_CHECK_RETURN(cudaFreeHost(h_read_buffer[i]));
        CUDA_CHECK_RETURN(cudaFreeHost(h_read_buffer_current[i]));
        CUDA_CHECK_RETURN(cudaFreeHost(h_read_buffer_previous[i]));

        CUDA_CHECK_RETURN(cudaFreeHost(h_current_messages[i]));
        CUDA_CHECK_RETURN(cudaFreeHost(h_current_messages_current[i]));
        CUDA_CHECK_RETURN(cudaFreeHost(h_current_messages_previous[i]));

        CUDA_CHECK_RETURN(cudaFree(dest_node_to_edges_nodes[i]));
        CUDA_CHECK_RETURN(cudaFree(dest_node_to_edges_edges[i]));
        CUDA_CHECK_RETURN(cudaFree(src_node_to_edges_nodes[i]));
        CUDA_CHECK_RETURN(cudaFree(src_node_to_edges_edges[i]));

        CUDA_CHECK_RETURN(cudaFree(current_messages[i]));
        CUDA_CHECK_RETURN(cudaFree(current_messages_current[i]));
        CUDA_CHECK_RETURN(cudaFree(current_messages_previous[i]));

        CUDA_CHECK_RETURN(cudaFree(node_states[i]));
        CUDA_CHECK_RETURN(cudaFree(node_states_previous[i]));
        CUDA_CHECK_RETURN(cudaFree(node_states_current[i]));

        CUDA_CHECK_RETURN(cudaFree(read_buffer[i]));

        CUDA_CHECK_RETURN(cudaFree(work_queue_nodes[i]));
        CUDA_CHECK_RETURN(cudaFree(work_queue_scratch[i]));
        CUDA_CHECK_RETURN(cudaFree(num_work_items[i]));

    }

    cudaSetDevice(0);

    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));

    free(node_states);
    free(node_states_current);

    free(h_read_buffer);
    free(h_read_buffer_current);
    free(h_read_buffer_previous);

    free(h_current_messages);
    free(h_current_messages_current);
    free(h_current_messages_previous);

    free(dest_node_to_edges_edges);
    free(dest_node_to_edges_nodes);
    free(src_node_to_edges_nodes);
    free(src_node_to_edges_edges);

    free(read_buffer);
    free(read_buffer_current);
    free(read_buffer_previous);


    free(current_messages);
    free(current_messages_previous);
    free(current_messages_current);

    free(work_queue_nodes);
    free(work_queue_scratch);
    free(num_work_items);

    free(my_buffer);

    free(threads);
    free(streams);
    free(thread_data);

    CUDA_CHECK_RETURN(cudaHostUnregister(graph->edges_messages));
    CUDA_CHECK_RETURN(cudaHostUnregister(graph->edges_messages_previous));
    CUDA_CHECK_RETURN(cudaHostUnregister(graph->edges_messages_current));
    CUDA_CHECK_RETURN(cudaHostUnregister(graph->node_states));
    CUDA_CHECK_RETURN(cudaHostUnregister(graph->work_queue_nodes));
    CUDA_CHECK_RETURN(cudaHostUnregister(&h_num_work_items));


    /*printf("After=====");
    print_nodes(graph);
    print_edges(graph);*/

    return num_iter;
}

/**
 * Runs loopy BP on the GPU
 * @param graph The graph to run
 * @param convergence The convergence threshold; when the delta falls below this threshold, execution will halt
 * @param max_iterations The number of executions to stop at
 * @return The actual number of iterations ran
 */
int loopy_propagate_until_cuda_streaming(Graph_t graph, const float convergence, const int max_iterations){
    int num_iter;
    size_t i, j, k;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct belief * current_messages;
    float * current_messages_previous;
    float * current_messages_current;

    struct belief * node_states;
    float * node_states_previous;
    float * node_states_current;

    struct belief * read_buffer;
    int retval;

    host_delta = 0.0f;
    previous_delta = INFINITY;

    init_work_queue_nodes(graph);

    //struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    pthread_t threads[NUM_THREAD_PARTITIONS];
    cudaStream_t streams[NUM_THREAD_PARTITIONS];
    struct node_stream_data thread_data[NUM_THREAD_PARTITIONS];

    size_t * dest_node_to_edges_nodes;
    size_t * dest_node_to_edges_edges;
    size_t * src_node_to_edges_nodes;
    size_t * src_node_to_edges_edges;
    size_t * work_queue_nodes;
    size_t * work_queue_scratch;
    unsigned long long int * num_work_items;

    const size_t num_vertices = graph->current_num_vertices;
    const size_t num_edges = graph->current_num_edges;
    const size_t edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    const size_t edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;
    const size_t node_states_size = graph->node_states_size;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_nodes, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_edges, sizeof(size_t) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_nodes, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_edges, sizeof(size_t) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_previous, sizeof(float) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_current, sizeof(float) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states_previous, sizeof(float) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states_current, sizeof(float) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_nodes, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_scratch, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&num_work_items, sizeof(unsigned long long int)));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&read_buffer, sizeof(struct belief) * graph->current_num_vertices));

    unsigned long long int h_num_work_items = (unsigned long long int)graph->num_work_items_nodes;

    // copy data
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(edge_joint_probability, &(graph->edge_joint_probability), sizeof(struct joint_probability)));

    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_previous, graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_current, graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) *  graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states_previous, graph->node_states_previous, sizeof(float) *  graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states_current, graph->node_states_current, sizeof(float) *  graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(size_t) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(size_t) * num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_nodes, graph->src_nodes_to_edges_node_list, sizeof(size_t) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_edges, graph->src_nodes_to_edges_edge_list, sizeof(size_t) * num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(work_queue_nodes, graph->work_queue_nodes, sizeof(size_t) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(num_work_items, &h_num_work_items, sizeof(unsigned long long int), cudaMemcpyHostToDevice));

    const int edgeCount = (num_edges + BLOCK_SIZE_NODE_STREAMING - 1)/ BLOCK_SIZE_NODE_STREAMING;
    //const int nodeCount = (num_vertices + BLOCK_SIZE_NODE_STREAMING - 1) / BLOCK_SIZE_NODE_STREAMING;''
    const int partitionSize = (num_vertices + NUM_THREAD_PARTITIONS - 1) / NUM_THREAD_PARTITIONS;
    const int partitionCount = (partitionSize + BLOCK_SIZE_NODE_STREAMING - 1) / BLOCK_SIZE_NODE_STREAMING;

    const int nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;

    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE_NODE_STREAMING, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE_NODE_STREAMING <= 32) ? 2 * BLOCK_SIZE_NODE_STREAMING * sizeof(float) : BLOCK_SIZE_NODE_STREAMING * sizeof(float);

    size_t curr_index = 0;
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
        thread_data[i].node_messages_size = node_states_size;
        thread_data[i].current_edge_messages = current_messages;
        thread_data[i].current_edge_messages_previous = current_messages_previous;
        thread_data[i].current_edge_messages_current = current_messages_current;
        thread_data[i].work_queue_nodes = work_queue_nodes;
        thread_data[i].num_work_items = num_work_items;
        thread_data[i].edge_joint_probability_dim_x = edge_joint_probability_dim_x;
        thread_data[i].edge_joint_probability_dim_y = edge_joint_probability_dim_y;
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
                    fprintf(stderr, "Error creating read thread %ld: %d\n", k, retval);
                    return 1;
                }
            }

            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining read thread %ld: %d\n", k, retval);
                    return 1;
                }
            }

            //send data
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_create(&threads[k], NULL, launch_write_node_kernels, &(thread_data[k]));
                if(retval) {
                    fprintf(stderr, "Error creating send thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining write thread %ld: %d\n", k, retval);
                    return 1;
                }
            }


            update_work_queue_nodes_cuda_kernel<<<nodeCount, BLOCK_SIZE>>>(work_queue_nodes, num_work_items, work_queue_scratch, node_states_previous, node_states_current, graph->current_num_vertices);
            test_error();
            num_iter++;
        }

        //marginalize
        for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
            retval = pthread_create(&threads[k], NULL, launch_marginalize_node_kernels, &(thread_data[k]));
            if(retval) {
                fprintf(stderr, "Error creating marginalize thread %ld: %d\n", k, retval);
                return 1;
            }
        }
        for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
            retval = pthread_join(threads[k], NULL);
            if(retval) {
                fprintf(stderr, "Error joining marginalize thread %ld: %d\n", k, retval);
                return 1;
            }
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages_previous, current_messages_current, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
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

    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(current_messages_previous));
    CUDA_CHECK_RETURN(cudaFree(current_messages_current));

    CUDA_CHECK_RETURN(cudaFree(node_states));
    CUDA_CHECK_RETURN(cudaFree(node_states_previous));
    CUDA_CHECK_RETURN(cudaFree(node_states_current));

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
 * Runs loopy BP on the GPU
 * @param graph The graph to run
 * @param convergence The convergence threshold; when the delta falls below this threshold, execution will halt
 * @param max_iterations The number of executions to stop at
 * @return The actual number of iterations ran
 */
int loopy_propagate_until_cuda_openmpi(Graph_t graph, const float convergence, const int max_iterations,
        int my_rank, int num_ranks, int num_devices){
    int num_iter;
    size_t i, j, k, l;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct belief ** current_messages;
    float ** current_messages_previous;
    float ** current_messages_current;

    struct belief * recv_current_messages;
    float * recv_current_messages_previous;
    float * recv_current_messages_current;

    struct belief ** h_current_messages;
    float ** h_current_messages_previous;
    float ** h_current_messages_current;

    struct belief ** node_states;
    float ** node_states_previous;
    float ** node_states_current;


    struct belief ** read_buffer;
    float ** read_buffer_previous;
    float ** read_buffer_current;

    struct belief * recv_read_buffer;
    float * recv_read_buffer_previous;
    float * recv_read_buffer_current;

    struct belief ** h_read_buffer;
    float ** h_read_buffer_previous;
    float ** h_read_buffer_current;

    struct belief * my_buffer;

    int retval;
    float node_difference;

    host_delta = 0.0f;
    previous_delta = INFINITY;

    init_work_queue_nodes(graph);

    MPI_Barrier(MPI_COMM_WORLD);

    //struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * num_devices);
    assert(threads);
    cudaStream_t *streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * num_devices);
    assert(streams);
    struct node_stream_data *thread_data = (struct node_stream_data *)malloc(sizeof(struct node_stream_data) * num_devices);
    assert(thread_data);

    size_t ** dest_node_to_edges_nodes;
    size_t ** dest_node_to_edges_edges;
    size_t ** src_node_to_edges_nodes;
    size_t ** src_node_to_edges_edges;
    size_t ** work_queue_nodes;
    size_t ** work_queue_scratch;
    unsigned long long int ** num_work_items;

    const size_t num_vertices = graph->current_num_vertices;
    const size_t num_edges = graph->current_num_edges;
    const size_t edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    const size_t edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;
    const size_t node_states_size = graph->node_states_size;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // init buffers
    recv_current_messages = (struct belief *)malloc(sizeof(struct belief) * num_ranks * graph->current_num_edges);
    assert(recv_current_messages);
    recv_current_messages_previous = (float *)malloc(sizeof(float) * num_ranks * graph->current_num_edges);
    assert(recv_current_messages_previous);
    recv_current_messages_current = (float *)malloc(sizeof(float) * num_ranks * graph->current_num_edges);
    assert(recv_current_messages_current);

    recv_read_buffer = (struct belief *)malloc(sizeof(struct belief) * num_ranks * graph->current_num_vertices);
    assert(recv_read_buffer);
    recv_read_buffer_current = (float *)malloc(sizeof(float) * num_ranks * graph->current_num_vertices);
    assert(recv_read_buffer_current);
    recv_read_buffer_previous = (float *)malloc(sizeof(float) * num_ranks * graph->current_num_vertices);
    assert(recv_read_buffer_previous);

    h_read_buffer = (struct belief **)malloc(sizeof(struct belief *) * num_devices);
    assert(h_read_buffer);
    h_read_buffer_current = (float **)malloc(sizeof(float *) * num_devices);
    assert(h_read_buffer_current);
    h_read_buffer_previous = (float **)malloc(sizeof(float *) * num_devices);
    assert(h_read_buffer_previous);

    h_current_messages = (struct belief **)malloc(sizeof(struct belief *) * num_devices);
    assert(h_current_messages);
    h_current_messages_current = (float **)malloc(sizeof(float *) * num_devices);
    assert(h_current_messages_current);
    h_current_messages_previous = (float **)malloc(sizeof(float *) * num_devices);
    assert(h_current_messages_previous);

    current_messages = (struct belief **)malloc(sizeof(struct belief *) * num_devices);
    assert(current_messages);
    current_messages_current = (float **)malloc(sizeof(float *) * num_devices);
    assert(current_messages_current);
    current_messages_previous = (float **)malloc(sizeof(float *) * num_devices);
    assert(current_messages_previous);

    read_buffer = (struct belief **)malloc(sizeof(struct belief *) * num_devices);
    assert(read_buffer);
    read_buffer_current = (float **)malloc(sizeof(float *) * num_devices);
    assert(read_buffer_current);
    read_buffer_previous = (float **)malloc(sizeof(float *) * num_devices);
    assert(read_buffer_previous);

    node_states = (struct belief **)malloc(sizeof(struct belief *) * num_devices);
    assert(node_states);
    node_states_previous = (float **)malloc(sizeof(float *) * num_devices);
    assert(node_states_previous);
    node_states_current = (float **)malloc(sizeof(float *) * num_devices);
    assert(node_states_current);

    dest_node_to_edges_nodes = (size_t **)malloc(sizeof(size_t) * num_devices);
    assert(dest_node_to_edges_nodes);
    dest_node_to_edges_edges = (size_t **)malloc(sizeof(size_t) * num_devices);
    assert(dest_node_to_edges_edges);
    src_node_to_edges_nodes = (size_t **)malloc(sizeof(size_t) * num_devices);
    assert(src_node_to_edges_nodes);
    src_node_to_edges_edges = (size_t **)malloc(sizeof(size_t) * num_devices);
    assert(src_node_to_edges_edges);
    work_queue_nodes = (size_t **)malloc(sizeof(size_t) * num_devices);
    assert(work_queue_nodes);
    work_queue_scratch = (size_t **)malloc(sizeof(size_t) * num_devices);
    assert(work_queue_scratch);
    num_work_items = (unsigned long long int **)malloc(sizeof(unsigned long long int) * num_devices);
    assert(num_work_items);

    my_buffer = (struct belief *)malloc(sizeof(struct belief) * graph->current_num_vertices);
    assert(my_buffer);

    unsigned long long int h_num_work_items = graph->num_work_items_nodes;

    // pin host memory
    CUDA_CHECK_RETURN(cudaHostRegister(graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(graph->work_queue_nodes, sizeof(size_t) * graph->current_num_vertices, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(&h_num_work_items, sizeof(unsigned long long int), cudaHostRegisterDefault));


    for(k = 0; k < num_devices; ++k) {
        cudaSetDevice(k);

        // allocate data
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(dest_node_to_edges_nodes[k]), sizeof(size_t) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(dest_node_to_edges_edges[k]), sizeof(size_t) * graph->current_num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(src_node_to_edges_nodes[k]), sizeof(size_t) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(src_node_to_edges_edges[k]), sizeof(size_t) * graph->current_num_edges));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &(current_messages[k]), sizeof(struct belief) * graph->current_num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(current_messages_current[k]), sizeof(float) * graph->current_num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(current_messages_previous[k]), sizeof(float) * graph->current_num_edges));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &(node_states[k]), sizeof(struct belief) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(node_states_previous[k]), sizeof(float) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(node_states_current[k]), sizeof(float) * graph->current_num_vertices));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &(work_queue_nodes[k]), sizeof(size_t) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(work_queue_scratch[k]), sizeof(size_t) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(num_work_items[k]), sizeof(unsigned long long int)));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &(read_buffer[k]), sizeof(struct belief) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(read_buffer_previous[k]), sizeof(float) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &(read_buffer_current[k]), sizeof(float) * graph->current_num_vertices));

        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(h_read_buffer[k]), sizeof(struct belief) * graph->current_num_vertices, cudaHostAllocDefault));
        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(h_read_buffer_previous[k]), sizeof(float) * graph->current_num_vertices, cudaHostAllocDefault));
        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(h_read_buffer_current[k]), sizeof(float) * graph->current_num_vertices, cudaHostAllocDefault));

        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(h_current_messages[k]), sizeof(struct belief) * graph->current_num_edges, cudaHostAllocDefault));
        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(h_current_messages_previous[k]), sizeof(float) * graph->current_num_edges, cudaHostAllocDefault));
        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &(h_current_messages_current[k]), sizeof(float) * graph->current_num_edges, cudaHostAllocDefault));
    }

    if(my_rank == 0) {
        cudaSetDevice(0);

        CUDA_CHECK_RETURN(cudaMalloc((void **) &delta, sizeof(float)));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &delta_array, sizeof(float) * num_edges));
    }

    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(edge_joint_probability, &(graph->edge_joint_probability), sizeof(struct joint_probability)));

    for(k = 0; k < num_devices; ++k) {

        CUDA_CHECK_RETURN(
                cudaMemcpy(current_messages[k], graph->edges_messages, sizeof(struct belief) * graph->current_num_edges,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(current_messages_current[k], graph->edges_messages_current, sizeof(float) * graph->current_num_edges,
                        cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(current_messages_previous[k], graph->edges_messages_previous, sizeof(float) * graph->current_num_edges,
                        cudaMemcpyHostToDevice));

        CUDA_CHECK_RETURN(
                cudaMemcpy(node_states[k], graph->node_states, sizeof(struct belief) * graph->current_num_vertices,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(node_states_current[k], graph->node_states_current, sizeof(float) * graph->current_num_vertices,
                        cudaMemcpyHostToDevice)
                );
        CUDA_CHECK_RETURN(
                cudaMemcpy(node_states_previous[k], graph->node_states_previous, sizeof(float) * graph->current_num_vertices,
                        cudaMemcpyHostToDevice));

        CUDA_CHECK_RETURN(
                cudaMemcpy(dest_node_to_edges_nodes[k], graph->dest_nodes_to_edges_node_list, sizeof(size_t) * num_vertices,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(dest_node_to_edges_edges[k], graph->dest_nodes_to_edges_edge_list, sizeof(size_t) * num_edges,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(src_node_to_edges_nodes[k], graph->src_nodes_to_edges_node_list, sizeof(size_t) * num_vertices,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(src_node_to_edges_edges[k], graph->src_nodes_to_edges_edge_list, sizeof(size_t) * num_edges,
                           cudaMemcpyHostToDevice));

        CUDA_CHECK_RETURN(cudaMemcpy(work_queue_nodes[k], graph->work_queue_nodes, sizeof(size_t) * num_vertices,
                                     cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(num_work_items[k], &h_num_work_items, sizeof(unsigned long long int), cudaMemcpyHostToDevice));
    }

    const size_t partitionRankSize = (num_vertices + num_ranks - 1) / num_ranks;
    const size_t partitionSize = (partitionRankSize + num_devices - 1) / num_devices;
    const size_t partitionCount = (partitionSize + BLOCK_SIZE_NODE_STREAMING - 1) / BLOCK_SIZE_NODE_STREAMING;

    const size_t edgeCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    const size_t nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;

    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE_NODE_STREAMING, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE_NODE_STREAMING <= 32) ? 2 * BLOCK_SIZE_NODE_STREAMING * sizeof(float) : BLOCK_SIZE_NODE_STREAMING * sizeof(float);

    size_t curr_index = my_rank * partitionRankSize;
    //prepare streams and data
    for(i = 0; i < num_devices; ++i) {
        CUDA_CHECK_RETURN(cudaSetDevice(i));
        CUDA_CHECK_RETURN(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

        thread_data[i].begin_index = curr_index;
        thread_data[i].end_index = curr_index + partitionSize;
        if(thread_data[i].end_index > num_vertices) {
            thread_data[i].end_index = num_vertices;
        }
        curr_index += partitionSize;
        thread_data[i].streamNodeCount = partitionCount;

        thread_data[i].buffers = read_buffer[i];
        thread_data[i].num_vertices = num_vertices;
        thread_data[i].num_edges = num_edges;
        thread_data[i].node_messages = node_states[i];
        thread_data[i].node_messages_size = node_states_size;
        thread_data[i].current_edge_messages = current_messages[i];
        thread_data[i].current_edge_messages_current = current_messages_current[i];
        thread_data[i].current_edge_messages_previous = current_messages_previous[i];
        thread_data[i].work_queue_nodes = work_queue_nodes[i];
        thread_data[i].num_work_items = num_work_items[i];
        thread_data[i].edge_joint_probability_dim_x = edge_joint_probability_dim_x;
        thread_data[i].edge_joint_probability_dim_y = edge_joint_probability_dim_y;
        thread_data[i].work_queue_scratch = work_queue_scratch[i];
        thread_data[i].src_nodes_to_edges_nodes = src_node_to_edges_nodes[i];
        thread_data[i].src_nodes_to_edges_edges = src_node_to_edges_edges[i];
        thread_data[i].dest_nodes_to_edges_nodes = dest_node_to_edges_nodes[i];
        thread_data[i].dest_nodes_to_edges_edges = dest_node_to_edges_edges[i];
        thread_data[i].stream = streams[i];
    }

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {

            //init + read data
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_create(&threads[k], NULL, launch_init_read_buffer_node_kernels, &(thread_data[k]));
                if(retval) {
                    fprintf(stderr, "Error creating read thread %ld: %d\n", k, retval);
                    return 1;
                }
            }

            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining read thread %ld: %d\n", k, retval);
                    return 1;
                }

            }

            //read data back
            // synchronize state

            // first get data back from devices
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                CUDA_CHECK_RETURN(cudaMemcpy(h_read_buffer[k], thread_data[k].buffers, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyDeviceToHost));
                // update beliefs
                for(l = thread_data[k].begin_index; l < thread_data[k].end_index && l < graph->num_work_items_nodes; ++l) {
                    curr_index = graph->work_queue_nodes[l];
                    if(curr_index < graph->current_num_vertices) {
                        memcpy(&(my_buffer[curr_index]), &(h_read_buffer[k][curr_index]), sizeof(struct belief));
                    }
                }
            }
            // send it to others
            MPICHECK(MPI_Allgather(my_buffer, graph->current_num_vertices, belief_struct, recv_read_buffer, graph->current_num_vertices, belief_struct, MPI_COMM_WORLD));
            // rebuild
            for(l = 0; l < graph->current_num_vertices; ++l) {
                node_difference = 0.0f;
                for(k = 0; k < num_ranks && node_difference < NODE_DIFFERENCE_THRESHOLD; k++) {
                    node_difference = difference(&(my_buffer[l]), graph->node_states_size, &(recv_read_buffer[k * graph->current_num_vertices + l]), graph->node_states_size);
                    if(node_difference >= NODE_DIFFERENCE_THRESHOLD) {
                        memcpy(&(my_buffer), &(read_buffer[k * graph->current_num_vertices + l]), sizeof(struct belief));
                    }
                }
            }
            // send it gpu
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                CUDA_CHECK_RETURN(cudaMemcpy(read_buffer[k], my_buffer, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));
            }

            //send data
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_create(&threads[k], NULL, launch_write_node_kernels, &(thread_data[k]));
                if(retval) {
                    fprintf(stderr, "Error creating send thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining write thread %ld: %d\n", k, retval);
                    return 1;
                }
            }

            // send data back
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                CUDA_CHECK_RETURN(cudaMemcpy(h_current_messages[k], current_messages[k], sizeof(struct belief) * graph->current_num_edges, cudaMemcpyDeviceToHost));
                CUDA_CHECK_RETURN(cudaMemcpy(h_current_messages_current[k], current_messages_current[k], sizeof(float) * graph->current_num_edges, cudaMemcpyDeviceToHost));
                CUDA_CHECK_RETURN(cudaMemcpy(h_current_messages_previous[k], current_messages_previous[k], sizeof(float) * graph->current_num_edges, cudaMemcpyDeviceToHost));
            }
            for(l = 0; l < graph->current_num_edges; ++l) {
                node_difference = 0.0f;
                for(k = 0; k < num_devices && node_difference < NODE_DIFFERENCE_THRESHOLD; ++k) {
                    node_difference = difference(&(h_current_messages[k][l]), graph->edges_messages_size, &(graph->edges_messages[l]), graph->edges_messages_size);
                    if(node_difference >= NODE_DIFFERENCE_THRESHOLD) {
                        memcpy(&(graph->edges_messages[l]), &(h_current_messages[k][l]), sizeof(struct belief));
                        memcpy(&(graph->edges_messages_current[l]), &(h_current_messages_current[k][l]), sizeof(float));
                        memcpy(&(graph->edges_messages_previous[l]), &(h_current_messages_current[k][l]), sizeof(float));
                    }
                }
            }

            // send to others
            MPI_Allgather(graph->edges_messages, graph->current_num_edges, belief_struct, recv_current_messages, graph->current_num_edges, belief_struct, MPI_COMM_WORLD);
            MPI_Allgather(graph->edges_messages_previous, graph->current_num_edges, MPI_FLOAT, recv_current_messages_previous, graph->current_num_edges, MPI_FLOAT, MPI_COMM_WORLD);
            MPI_Allgather(graph->edges_messages_current, graph->current_num_edges, MPI_FLOAT, recv_current_messages_current, graph->current_num_edges, MPI_FLOAT, MPI_COMM_WORLD);

            // combine
            for(l = 0; l < graph->current_num_edges; ++l) {
                node_difference = 0.0f;
                for(k = 0; k < num_ranks && node_difference < NODE_DIFFERENCE_THRESHOLD; ++k) {
                    node_difference = difference(&(graph->edges_messages[l]), graph->edges_messages_size, &(recv_current_messages[k * graph->current_num_edges + l]), graph->edges_messages_size);
                    if(node_difference >= NODE_DIFFERENCE_THRESHOLD) {
                        memcpy(&(graph->edges_messages[l]), &(recv_current_messages[k * graph->current_num_edges + l]), sizeof(struct belief));
                        memcpy(&(graph->edges_messages_current[l]), &(recv_current_messages_current[k * graph->current_num_edges + l]), sizeof(float));
                        memcpy(&(graph->edges_messages_previous[l]), &(recv_current_messages_previous[k * graph->current_num_edges + l]), sizeof(float));
                    }
                }
            }

            // send to gpu
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                CUDA_CHECK_RETURN(cudaMemcpy(current_messages[k], graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpy(current_messages_previous[k], graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpy(current_messages_current[k], graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
            }

            //marginalize
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_create(&threads[k], NULL, launch_marginalize_node_kernels, &(thread_data[k]));
                if(retval) {
                    fprintf(stderr, "Error creating marginalize thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining marginalize thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            // copy back
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                CUDA_CHECK_RETURN(cudaMemcpy(h_read_buffer[k], node_states[k], sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyDeviceToHost));
                CUDA_CHECK_RETURN(cudaMemcpy(h_read_buffer_current[k], node_states_current[k], sizeof(float) * graph->current_num_vertices, cudaMemcpyDeviceToHost));
                CUDA_CHECK_RETURN(cudaMemcpy(h_read_buffer_previous[k], node_states_previous[k], sizeof(float) * graph->current_num_vertices, cudaMemcpyDeviceToHost));

                for(l = my_rank * partitionRankSize; l < (my_rank + 1) * partitionRankSize && l < graph->current_num_vertices; ++l) {
                    memcpy(&(graph->node_states[l]), &(h_read_buffer[k][l]), sizeof(struct belief));
                    memcpy(&(graph->node_states_previous[l]), &(h_read_buffer_previous[k][l]), sizeof(float));
                    memcpy(&(graph->node_states_current[l]), &(h_read_buffer_current[k][l]), sizeof(float));
                }
            }
            // send it out
            MPI_Allgather(graph->node_states, graph->current_num_vertices, belief_struct, recv_read_buffer, graph->current_num_vertices, belief_struct, MPI_COMM_WORLD);
            MPI_Allgather(graph->node_states_previous, graph->current_num_vertices, MPI_FLOAT, recv_read_buffer_previous, graph->current_num_vertices, MPI_FLOAT, MPI_COMM_WORLD);
            MPI_Allgather(graph->node_states_current, graph->current_num_vertices, MPI_FLOAT, recv_read_buffer_current, graph->current_num_vertices, MPI_FLOAT, MPI_COMM_WORLD);
            // rebuild
            for(k = 0; k < num_ranks; ++k) {
                for(l = my_rank * partitionRankSize; l < (my_rank + 1) * partitionRankSize && l < graph->current_num_vertices; ++l) {
                    memcpy(&(graph->node_states[l]), &(recv_read_buffer[k * graph->current_num_vertices + l]), sizeof(struct belief));
                    memcpy(&(graph->node_states_previous[l]), &(recv_read_buffer_previous[k * graph->current_num_vertices + l]), sizeof(float));
                    memcpy(&(graph->node_states_current[l]), &(recv_read_buffer_current[k * graph->current_num_vertices + l]), sizeof(float));
                }
            }
            // send to gpu
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                CUDA_CHECK_RETURN(cudaMemcpy(node_states[k], graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpy(node_states_previous[k], graph->node_states_previous, sizeof(float) * graph->current_num_vertices, cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpy(node_states_current[k], graph->node_states_current, sizeof(float) * graph->current_num_vertices, cudaMemcpyHostToDevice));
            }

            for(k = 0; k < num_devices; ++k) {
                update_work_queue_nodes_cuda_kernel<<<nodeCount, BLOCK_SIZE>>>(work_queue_nodes[k], num_work_items[k], work_queue_scratch[k], node_states_previous[k], node_states_current[k], graph->current_num_vertices);
                test_error();
                num_iter++;
            }
            //copy back
            CUDA_CHECK_RETURN(cudaSetDevice(0));
            CUDA_CHECK_RETURN(cudaMemcpy(graph->work_queue_nodes, work_queue_nodes[0], sizeof(size_t) * graph->current_num_vertices, cudaMemcpyDeviceToHost));
            CUDA_CHECK_RETURN(cudaMemcpy(&(graph->num_work_items_nodes), num_work_items[0], sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        }

        if(my_rank == 0) {
            CUDA_CHECK_RETURN(cudaSetDevice(0));
            calculate_delta_6 << < dimReduceGrid, dimReduceBlock, reduceSmemSize >> >
                                                                  (current_messages_previous[0], current_messages_current[0], delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
            //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
            //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
            test_error();
            CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(&host_delta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        //printf("Current delta: %f\n", host_delta);

        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            break;
        }
        previous_delta = host_delta;
    }

    for(i = 0; i < num_devices; ++i) {
        CUDA_CHECK_RETURN(cudaSetDevice(i));
        cudaStreamDestroy(streams[i]);
    }

    CUDA_CHECK_RETURN(cudaSetDevice(0));
    // copy data back
    CUDA_CHECK_RETURN(cudaMemcpy(graph->node_states, node_states[0], sizeof(struct belief) * num_vertices, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(graph->edges_messages, current_messages[0], sizeof(struct belief) * num_edges, cudaMemcpyDeviceToHost));

    for(i = 0; i < num_devices; ++i) {
        CUDA_CHECK_RETURN(cudaSetDevice(i));

        CUDA_CHECK_RETURN(cudaFreeHost(h_read_buffer[i]));
        CUDA_CHECK_RETURN(cudaFreeHost(h_read_buffer_current[i]));
        CUDA_CHECK_RETURN(cudaFreeHost(h_read_buffer_previous[i]));

        CUDA_CHECK_RETURN(cudaFreeHost(h_current_messages[i]));
        CUDA_CHECK_RETURN(cudaFreeHost(h_current_messages_current[i]));
        CUDA_CHECK_RETURN(cudaFreeHost(h_current_messages_previous[i]));

        CUDA_CHECK_RETURN(cudaFree(dest_node_to_edges_nodes[i]));
        CUDA_CHECK_RETURN(cudaFree(dest_node_to_edges_edges[i]));
        CUDA_CHECK_RETURN(cudaFree(src_node_to_edges_nodes[i]));
        CUDA_CHECK_RETURN(cudaFree(src_node_to_edges_edges[i]));

        CUDA_CHECK_RETURN(cudaFree(current_messages[i]));
        CUDA_CHECK_RETURN(cudaFree(current_messages_current[i]));
        CUDA_CHECK_RETURN(cudaFree(current_messages_previous[i]));

        CUDA_CHECK_RETURN(cudaFree(node_states[i]));
        CUDA_CHECK_RETURN(cudaFree(node_states_previous[i]));
        CUDA_CHECK_RETURN(cudaFree(node_states_current[i]));

        CUDA_CHECK_RETURN(cudaFree(read_buffer[i]));

        CUDA_CHECK_RETURN(cudaFree(work_queue_nodes[i]));
        CUDA_CHECK_RETURN(cudaFree(work_queue_scratch[i]));
        CUDA_CHECK_RETURN(cudaFree(num_work_items[i]));

    }

    if(my_rank == 0) {
        cudaSetDevice(0);

        CUDA_CHECK_RETURN(cudaFree(delta));
        CUDA_CHECK_RETURN(cudaFree(delta_array));
    }

    free(node_states);
    free(node_states_current);

    free(h_read_buffer);
    free(h_read_buffer_current);
    free(h_read_buffer_previous);

    free(h_current_messages);
    free(h_current_messages_current);
    free(h_current_messages_previous);

    free(dest_node_to_edges_edges);
    free(dest_node_to_edges_nodes);
    free(src_node_to_edges_nodes);
    free(src_node_to_edges_edges);

    free(read_buffer);
    free(read_buffer_current);
    free(read_buffer_previous);


    free(current_messages);
    free(current_messages_previous);
    free(current_messages_current);

    free(work_queue_nodes);
    free(work_queue_scratch);
    free(num_work_items);

    free(my_buffer);

    free(threads);
    free(streams);
    free(thread_data);

    free(recv_current_messages);
    free(recv_current_messages_current);
    free(recv_current_messages_previous);

    free(recv_read_buffer);
    free(recv_read_buffer_current);
    free(recv_read_buffer_previous);

    CUDA_CHECK_RETURN(cudaHostUnregister(graph->edges_messages));
    CUDA_CHECK_RETURN(cudaHostUnregister(graph->edges_messages_previous));
    CUDA_CHECK_RETURN(cudaHostUnregister(graph->edges_messages_current));
    CUDA_CHECK_RETURN(cudaHostUnregister(graph->node_states));
    CUDA_CHECK_RETURN(cudaHostUnregister(graph->work_queue_nodes));
    CUDA_CHECK_RETURN(cudaHostUnregister(&h_num_work_items));


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
int page_rank_until_cuda(Graph_t graph, const float convergence, const int max_iterations){
    int num_iter;
    size_t i, j;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct belief * current_messages;
    float * current_messages_previous;
    float * current_messages_current;

    struct belief * node_states;
    float * node_states_previous;
    float * node_states_current;

    host_delta = 0.0f;
    previous_delta = INFINITY;

    //struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    size_t * dest_node_to_edges_nodes;
    size_t * dest_node_to_edges_edges;
    size_t * src_node_to_edges_nodes;
    size_t * src_node_to_edges_edges;

    const size_t num_vertices = graph->current_num_vertices;
    const size_t num_edges = graph->current_num_edges;
    const size_t edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    const size_t edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;
    const size_t node_states_size = graph->node_states_size;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_nodes, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_edges, sizeof(size_t) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_nodes, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_edges, sizeof(size_t) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_previous, sizeof(float) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_current, sizeof(float) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states_previous, sizeof(float) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states_current, sizeof(float) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));


    // copy data
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(edge_joint_probability, &(graph->edge_joint_probability), sizeof(struct joint_probability)));

    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_current, graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_previous, graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) *  graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states_previous, graph->node_states_previous, sizeof(float) *  graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states_current, graph->node_states_current, sizeof(float) *  graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(size_t) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(size_t) * num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_nodes, graph->src_nodes_to_edges_node_list, sizeof(size_t) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_edges, graph->src_nodes_to_edges_edge_list, sizeof(size_t) * num_edges, cudaMemcpyHostToDevice));

    const int edgeCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    const int nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            page_rank_main_loop<<<nodeCount, BLOCK_SIZE >>>(num_vertices, num_edges,
                    node_states, node_states_size, node_states_previous, node_states_current,
                    edge_joint_probability_dim_x, edge_joint_probability_dim_y,
                    current_messages, current_messages_previous, current_messages_current,
                    src_node_to_edges_nodes, src_node_to_edges_edges, src_node_to_edges_nodes, src_node_to_edges_edges);
            test_error();
            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages_previous, current_messages_current, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
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

    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(current_messages_current));
    CUDA_CHECK_RETURN(cudaFree(current_messages_previous));

    CUDA_CHECK_RETURN(cudaFree(node_states));
    CUDA_CHECK_RETURN(cudaFree(node_states_previous));
    CUDA_CHECK_RETURN(cudaFree(node_states_current));

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
int viterbi_until_cuda(Graph_t graph, const float convergence, const int max_iterations){
    int num_iter;
    size_t i, j;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct belief * current_messages;
    float * current_messages_previous;
    float * current_messages_current;

    struct belief * node_states;
    float * node_states_previous;
    float * node_states_current;

    host_delta = 0.0f;
    previous_delta = INFINITY;

    //struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    size_t * dest_node_to_edges_nodes;
    size_t * dest_node_to_edges_edges;
    size_t * src_node_to_edges_nodes;
    size_t * src_node_to_edges_edges;

    const size_t num_vertices = graph->current_num_vertices;
    const size_t num_edges = graph->current_num_edges;
    const size_t edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    const size_t edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;
    const size_t node_states_size = graph->node_states_size;


    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_nodes, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_node_to_edges_edges, sizeof(size_t) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_nodes, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&src_node_to_edges_edges, sizeof(size_t) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_previous, sizeof(float) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_current, sizeof(float) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states_previous, sizeof(float) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states_current, sizeof(float) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));


    // copy data
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(edge_joint_probability, &(graph->edge_joint_probability), sizeof(struct joint_probability)));

    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_previous, graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_current, graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) *  graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states_previous, graph->node_states_previous, sizeof(float) *  graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(node_states_current, graph->node_states_current, sizeof(float) *  graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(size_t) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_node_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(size_t) * num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_nodes, graph->src_nodes_to_edges_node_list, sizeof(size_t) * num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(src_node_to_edges_edges, graph->src_nodes_to_edges_edge_list, sizeof(size_t) * num_edges, cudaMemcpyHostToDevice));

    const int edgeCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    const int nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE){
        for(j = 0; j < BATCH_SIZE; ++j) {
            viterbi_main_loop<<<nodeCount, BLOCK_SIZE >>>(num_vertices, num_edges,
                    node_states, node_states_size, node_states_previous, node_states_current,
                    edge_joint_probability_dim_x, edge_joint_probability_dim_y,
                    current_messages, current_messages_previous, current_messages_current,
                    src_node_to_edges_nodes, src_node_to_edges_edges, src_node_to_edges_nodes, src_node_to_edges_edges);
            test_error();
            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages_previous, current_messages_current, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        test_error();
        CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
        //printf("Current delta: %f\n", host_delta);

        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            marginalize_viterbi_beliefs<<<nodeCount, BLOCK_SIZE >>>(node_states, node_states_size, num_vertices);
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

    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(current_messages_previous));
    CUDA_CHECK_RETURN(cudaFree(current_messages_current));

    CUDA_CHECK_RETURN(cudaFree(node_states));
    CUDA_CHECK_RETURN(cudaFree(node_states_previous));
    CUDA_CHECK_RETURN(cudaFree(node_states_current));

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
int loopy_propagate_until_cuda_edge(Graph_t graph, const float convergence, const int max_iterations){
    int num_iter;
    size_t i, j;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct belief * current_messages;
    float * current_messages_previous;
    float * current_messages_current;

    struct belief * node_states;

    size_t * edges_src_index;
    size_t * edges_dest_index;
    size_t * dest_nodes_to_edges_nodes;
    size_t * dest_nodes_to_edges_edges;

    size_t * work_queue_edges;
    size_t * work_queue_scratch;
    unsigned long long int * num_work_items;

    init_work_queue_edges(graph);

    host_delta = 0.0f;
    previous_delta = INFINITY;

    //struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    const size_t num_vertices = graph->current_num_vertices;
    const size_t num_edges = graph->current_num_edges;
    const size_t edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    const size_t edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;
    const size_t current_messages_size = graph->edges_messages_size;
    const size_t node_states_size = graph->node_states_size;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_src_index, sizeof(size_t) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_dest_index, sizeof(size_t) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_previous, sizeof(float) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_current, sizeof(float) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_nodes, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_edges, sizeof(size_t) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_edges, sizeof(size_t) * num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_scratch, sizeof(size_t) * num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&num_work_items, sizeof(unsigned long long int)));

    unsigned long long int h_num_work_items = num_edges;

    // copy data
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(edge_joint_probability, &(graph->edge_joint_probability), sizeof(struct joint_probability)));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_previous, graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_current, graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(edges_src_index, graph->edges_src_index, sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(edges_dest_index, graph->edges_dest_index, sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(size_t) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(work_queue_edges, graph->work_queue_edges, sizeof(size_t) * num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(num_work_items, &h_num_work_items, sizeof(unsigned long long int), cudaMemcpyHostToDevice));


    const int edgeCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    const int nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE_EDGE){
        for(j = 0; j < BATCH_SIZE_EDGE; ++j) {
            send_message_for_edge_iteration_cuda_work_queue_kernel<<<edgeCount, BLOCK_SIZE >>>(num_edges, edges_src_index,
                    node_states, edge_joint_probability_dim_x, edge_joint_probability_dim_y,
                    current_messages, current_messages_previous, current_messages_current,
                    work_queue_edges, num_work_items);
            test_error();
            combine_loopy_edge_cuda_work_queue_kernel<<<edgeCount, BLOCK_SIZE>>>(num_edges, edges_dest_index,
                    current_messages, current_messages_previous, current_messages_current, current_messages_size,
                    node_states, work_queue_edges, num_work_items, work_queue_scratch);
            test_error();
            //marginalize_loop_node_edge_kernel<<<nodeCount, BLOCK_SIZE>>>(node_states, num_vars, num_vertices);
            num_iter++;
        }
        marginalize_nodes<<<nodeCount, BLOCK_SIZE>>>(node_states, node_states_size,
                current_messages,
                dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
        test_error();
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages_previous, current_messages_current, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
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

    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(current_messages_current));
    CUDA_CHECK_RETURN(cudaFree(current_messages_previous));

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

    send_message_for_edge_iteration_cuda_work_queue_kernel_streaming<<<stream_data->streamEdgeCount, BLOCK_SIZE_EDGE_STREAMING, 0, stream_data->stream >>>(
            stream_data->begin_index, stream_data->end_index, stream_data->edges_src_index,
            stream_data->node_states,
            stream_data->edge_joint_probability_dim_x, stream_data->edge_joint_probability_dim_y,
            stream_data->current_edge_messages, stream_data->current_edge_messages_previous, stream_data->current_edge_messages_current,
            stream_data->work_queue_edges, stream_data->num_work_items);

    cudaStreamSynchronize(stream_data->stream);
    test_error();

    return NULL;
}

static void* launch_combine_message_kernel(void * data) {
    struct edge_stream_data * stream_data;

    stream_data = (struct edge_stream_data *)data;

    combine_loopy_edge_cuda_work_queue_kernel_streaming<<<stream_data->streamEdgeCount, BLOCK_SIZE_EDGE_STREAMING, 0, stream_data->stream>>>(
            stream_data->begin_index, stream_data->end_index, stream_data->edges_dest_index,
            stream_data->current_edge_messages, stream_data->current_edge_messages_size,
            stream_data->node_states, stream_data->work_queue_edges, stream_data->num_work_items, stream_data->work_queue_scratch);

    cudaStreamSynchronize(stream_data->stream);
    test_error();

    return NULL;
}

static void* launch_marginalize_streaming_kernel(void * data) {
    struct node_stream_data *stream_data;

    stream_data = (struct node_stream_data *)data;


    marginalize_nodes_streaming<<<stream_data->streamNodeCount, BLOCK_SIZE_NODE_EDGE_STREAMING, 0, stream_data->stream>>>(
            stream_data->begin_index, stream_data->end_index,
            stream_data->node_messages, stream_data->node_messages_size,
            stream_data->current_edge_messages,
            stream_data->dest_nodes_to_edges_nodes, stream_data->dest_nodes_to_edges_edges, stream_data->num_vertices, stream_data->num_edges);

    cudaStreamSynchronize(stream_data->stream);
    test_error();

    return NULL;
}

__global__
void update_work_queue_cuda_kernel(size_t * work_queue_edges, unsigned long long int * num_work_items, size_t* work_queue_scratch,
                                   float *current_messages_previous, float *current_messages_current, const size_t num_edges) {
    update_work_queue_edges_cuda(work_queue_edges, num_work_items, work_queue_scratch,
            current_messages_previous, current_messages_current, num_edges, PRECISION_ITERATION);
}

__global__
void update_work_queue_edges_cuda_kernel(size_t * work_queue_edges, unsigned long long int * num_work_items, size_t * work_queue_scratch,
        float *current_edge_messages_previous, float *current_edge_messages_current, const size_t num_edges) {
    update_work_queue_edges_cuda(work_queue_edges, num_work_items, work_queue_scratch,
            current_edge_messages_previous, current_edge_messages_current, num_edges, PRECISION_ITERATION);
}

/**
 * Runs the edge-optimized loopy BP code
 * @param graph The graph to use
 * @param convergence The convergence threshold; when the delta falls below this threshold, execution will stop
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations ran
 */
int loopy_propagate_until_cuda_edge_streaming(Graph_t graph, const float convergence, const int max_iterations){
    int num_iter;
    size_t i, j, k;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct belief * current_messages;
    float * current_messages_previous;
    float * current_messages_current;

    struct belief * node_states;

    size_t * edges_src_index;
    size_t * edges_dest_index;
    size_t * dest_nodes_to_edges_nodes;
    size_t * dest_nodes_to_edges_edges;

    size_t * work_queue_edges;
    size_t * work_queue_scratch;
    unsigned long long int * num_work_items;

    int retval;

    init_work_queue_edges(graph);

    host_delta = 0.0f;
    previous_delta = INFINITY;

    //struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    const size_t num_vertices = graph->current_num_vertices;
    const size_t num_edges = graph->current_num_edges;
    const size_t edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    const size_t edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;
    const size_t current_messages_size = graph->edges_messages_size;
    const size_t node_states_size = graph->node_states_size;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    pthread_t threads[NUM_THREAD_PARTITIONS];
    cudaStream_t streams[NUM_THREAD_PARTITIONS];
    struct edge_stream_data thread_data[NUM_THREAD_PARTITIONS];
    struct node_stream_data node_thread_data[NUM_THREAD_PARTITIONS];

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_src_index, sizeof(size_t) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_dest_index, sizeof(size_t) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_previous, sizeof(float) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_current, sizeof(float) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_nodes, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_edges, sizeof(size_t) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_edges, sizeof(size_t) * num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&work_queue_scratch, sizeof(size_t) * num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&num_work_items, sizeof(unsigned long long int)));

    unsigned long long int h_num_work_items = num_edges;

    // copy data
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(edge_joint_probability, &(graph->edge_joint_probability), sizeof(struct joint_probability)));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_previous, graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_current, graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(edges_src_index, graph->edges_src_index, sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(edges_dest_index, graph->edges_dest_index, sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(size_t) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(work_queue_edges, graph->work_queue_edges, sizeof(size_t) * num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(num_work_items, &h_num_work_items, sizeof(unsigned long long int), cudaMemcpyHostToDevice));

    const size_t edgeCount = (num_edges + BLOCK_SIZE_EDGE_STREAMING - 1)/ BLOCK_SIZE_EDGE_STREAMING;
//    const int nodeCount = (num_vertices + BLOCK_SIZE_EDGE_STREAMING - 1) / BLOCK_SIZE_EDGE_STREAMING;

    //const int partitionSize = (num_vertices + NUM_THREAD_PARTITIONS - 1) / NUM_THREAD_PARTITIONS;
    //const int partitionCount = (partitionSize + BLOCK_SIZE_NODE_STREAMING - 1) / BLOCK_SIZE_NODE_STREAMING;
    const size_t edgePartitionSize = (num_edges + NUM_THREAD_PARTITIONS - 1) / NUM_THREAD_PARTITIONS;
    const size_t edgePartitionCount = (edgePartitionSize + BLOCK_SIZE_EDGE_STREAMING - 1) / BLOCK_SIZE_EDGE_STREAMING;
    const size_t nodePartitionSize = (num_vertices + NUM_THREAD_PARTITIONS - 1) / NUM_THREAD_PARTITIONS;
    const size_t nodePartitionCount = (nodePartitionSize + BLOCK_SIZE_NODE_EDGE_STREAMING - 1) / BLOCK_SIZE_NODE_EDGE_STREAMING;

    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE_EDGE_STREAMING, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE_EDGE_STREAMING <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE_EDGE_STREAMING * sizeof(float);

    int curr_index = 0;
    size_t curr_node_index = 0;
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

        thread_data[i].edge_joint_probability_dim_x = edge_joint_probability_dim_x;
        thread_data[i].edge_joint_probability_dim_y = edge_joint_probability_dim_y;

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
        thread_data[i].current_edge_messages_size = current_messages_size;
        thread_data[i].current_edge_messages_previous = current_messages_previous;
        thread_data[i].current_edge_messages_current = current_messages_current;

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
        node_thread_data[i].node_messages_size = node_states_size;

        node_thread_data[i].current_edge_messages = current_messages;
        node_thread_data[i].current_edge_messages_current = current_messages_current;
        node_thread_data[i].current_edge_messages_previous = current_messages_previous;

        node_thread_data[i].dest_nodes_to_edges_nodes = dest_nodes_to_edges_edges;
        node_thread_data[i].dest_nodes_to_edges_edges = dest_nodes_to_edges_edges;
        node_thread_data[i].num_vertices = num_vertices;
        node_thread_data[i].num_edges = num_edges;

        node_thread_data[i].num_work_items = NULL;
        node_thread_data[i].work_queue_scratch = NULL;
        node_thread_data[i].work_queue_nodes = NULL;
        node_thread_data[i].src_nodes_to_edges_edges = NULL;
        node_thread_data[i].src_nodes_to_edges_nodes = NULL;
        node_thread_data[i].buffers = NULL;
    }

    for(i = 0; i < max_iterations; i+= BATCH_SIZE_EDGE){
        for(j = 0; j < BATCH_SIZE_EDGE; ++j) {
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_create(&threads[k], NULL, launch_send_message_kernel, &thread_data[k]);
                if(retval) {
                    fprintf(stderr, "Error creating send message thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining send message thread %ld: %d\n", k, retval);
                    return 1;
                }
            }

            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_create(&threads[k], NULL, launch_combine_message_kernel, &thread_data[k]);
                if(retval) {
                    fprintf(stderr, "Error creating combine message thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining combine message thread %ld: %d\n", k, retval);
                    return 1;
                }
            }


            update_work_queue_cuda_kernel<<<edgeCount, BLOCK_SIZE_EDGE_STREAMING>>>(work_queue_edges, num_work_items, work_queue_scratch, current_messages_previous, current_messages_current, num_edges);
            test_error();
            //marginalize_loop_node_edge_kernel<<<nodeCount, BLOCK_SIZE>>>(node_states, num_vars, num_vertices);
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_create(&threads[k], NULL, launch_marginalize_streaming_kernel, &node_thread_data[k]);
                if(retval) {
                    fprintf(stderr, "Error creating marginalize node thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < NUM_THREAD_PARTITIONS; ++k) {
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining marginalize node thread %ld: %d\n", k, retval);
                    return 1;
                }
            }

            update_work_queue_edges_cuda_kernel<<<edgeCount, BLOCK_SIZE>>>(work_queue_edges, num_work_items, work_queue_scratch, current_messages_previous, current_messages_current, graph->current_num_edges);
            test_error();
            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages_previous, current_messages_current,
                delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
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

    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(current_messages_previous));
    CUDA_CHECK_RETURN(cudaFree(current_messages_current));

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
 * Runs the edge-optimized loopy BP code
 * @param graph The graph to use
 * @param convergence The convergence threshold; when the delta falls below this threshold, execution will stop
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations ran
 */
int loopy_propagate_until_cuda_edge_multiple_devices(Graph_t graph, float convergence, int max_iterations){
    int num_iter, copy_amount;
    size_t i, j, k, l, m;
    float *delta;
    float *delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct belief ** current_messages;
    float ** current_messages_previous;
    float ** current_messages_current;

    struct belief ** h_current_messages;
    float ** h_current_messages_previous;
    float ** h_current_messages_current;

    struct belief ** node_states;

    struct belief ** h_node_states;

    size_t ** edges_src_index;
    size_t ** edges_dest_index;
    size_t ** dest_nodes_to_edges_nodes;
    size_t ** dest_nodes_to_edges_edges;

    size_t ** work_queue_edges;
    size_t ** work_queue_scratch;
    unsigned long long int ** num_work_items;

    int retval;

    init_work_queue_edges(graph);

    int num_devices = -1;
    cudaGetDeviceCount(&num_devices);
    assert(num_devices >= 1);

    // allocate device arrays


    current_messages = (struct belief **)malloc(num_devices * sizeof(struct belief *));
    assert(current_messages);
    current_messages_previous = (float **)malloc(num_devices * sizeof(float *));
    assert(current_messages_previous);
    current_messages_current = (float **)malloc(num_devices * sizeof(float *));
    assert(current_messages_current);

    h_current_messages = (struct belief **)malloc(num_devices * sizeof(struct belief *));
    assert(h_current_messages);
    h_current_messages_previous = (float **)malloc(num_devices * sizeof(float *));
    assert(h_current_messages_previous);
    h_current_messages_current = (float **)malloc(num_devices * sizeof(float *));
    assert(h_current_messages_current);

    node_states = (struct belief **)malloc(num_devices * sizeof(struct belief *));
    assert(node_states);

    h_node_states = (struct belief **)malloc(num_devices * sizeof(struct belief *));
    assert(h_node_states);

    edges_src_index = (size_t **)malloc(num_devices * sizeof(size_t *));
    assert(edges_src_index);
    edges_dest_index = (size_t **)malloc(num_devices * sizeof(size_t *));
    assert(edges_dest_index);
    dest_nodes_to_edges_nodes = (size_t **)malloc(num_devices * sizeof(size_t *));
    assert(dest_nodes_to_edges_nodes);
    dest_nodes_to_edges_edges = (size_t **)malloc(num_devices * sizeof(size_t *));
    assert(dest_nodes_to_edges_edges);

    work_queue_edges = (size_t **)malloc(num_devices * sizeof(size_t *));
    assert(work_queue_edges);
    work_queue_scratch = (size_t **)malloc(num_devices * sizeof(size_t *));
    assert(work_queue_scratch);
    num_work_items = (unsigned long long int **)malloc(num_devices * sizeof(unsigned long long int *));
    assert(num_work_items);


    unsigned long long int h_num_work_items = (unsigned long long int)graph->num_work_items_edges;
    // pin host memory
    CUDA_CHECK_RETURN(cudaHostRegister(graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(graph->work_queue_edges, sizeof(size_t) * graph->current_num_edges, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(&h_num_work_items, sizeof(unsigned long long int), cudaHostRegisterDefault));

    host_delta = 0.0f;
    previous_delta = INFINITY;

    //struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    const size_t num_vertices = graph->current_num_vertices;
    const size_t num_edges = graph->current_num_edges;
    const size_t edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    const size_t edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;
    const size_t current_messages_size = graph->edges_messages_size;
    const size_t node_states_size = graph->node_states_size;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * num_devices);
    cudaStream_t *streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * num_devices);
    struct edge_stream_data *thread_data = (struct edge_stream_data *)malloc(sizeof(struct edge_stream_data) * num_devices);
    struct node_stream_data *node_thread_data = (struct node_stream_data *)malloc(sizeof(struct node_stream_data) * num_devices);

    cudaSetDevice(0);

    CUDA_CHECK_RETURN(cudaMalloc((void **) &delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &delta_array, sizeof(float) * num_edges));


    for(k = 0; k < num_devices; ++k) {
        cudaSetDevice(k);
        // allocate data
        CUDA_CHECK_RETURN(cudaMalloc((void **) &edges_src_index[k], sizeof(size_t) * graph->current_num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &edges_dest_index[k], sizeof(size_t) * graph->current_num_edges));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &node_states[k], sizeof(struct belief) * graph->current_num_vertices));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &current_messages[k], sizeof(struct belief) * graph->current_num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &current_messages_previous[k], sizeof(float) * graph->current_num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &current_messages_current[k], sizeof(float) * graph->current_num_edges));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &dest_nodes_to_edges_nodes[k], sizeof(size_t) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &dest_nodes_to_edges_edges[k], sizeof(size_t) * graph->current_num_edges));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &work_queue_edges[k], sizeof(size_t) * num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &work_queue_scratch[k], sizeof(size_t) * num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &num_work_items[k], sizeof(unsigned long long int)));

        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &h_node_states[k], sizeof(struct belief) * num_vertices, cudaHostAllocDefault));

        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &h_current_messages[k], sizeof(struct belief) * num_edges, cudaHostAllocDefault));
        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &h_current_messages_previous[k], sizeof(float) * num_edges, cudaHostAllocDefault));
        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &h_current_messages_current[k], sizeof(float) * num_edges, cudaHostAllocDefault));


        // copy data
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(edge_joint_probability, &(graph->edge_joint_probability), sizeof(struct joint_probability)));

        CUDA_CHECK_RETURN(
                cudaMemcpy(node_states[k], graph->node_states, sizeof(struct belief) * graph->current_num_vertices,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(current_messages[k], graph->edges_messages, sizeof(struct belief) * graph->current_num_edges,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(current_messages_previous[k], graph->edges_messages_previous, sizeof(float) * graph->current_num_edges,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(current_messages_current[k], graph->edges_messages_current, sizeof(float) * graph->current_num_edges,
                           cudaMemcpyHostToDevice));

        CUDA_CHECK_RETURN(cudaMemcpy(edges_src_index[k], graph->edges_src_index, sizeof(size_t) * graph->current_num_edges,
                                     cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(edges_dest_index[k], graph->edges_dest_index, sizeof(size_t) * graph->current_num_edges,
                                     cudaMemcpyHostToDevice));

        CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_nodes[k], graph->dest_nodes_to_edges_node_list,
                                     sizeof(size_t) * graph->current_num_vertices, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_edges[k], graph->dest_nodes_to_edges_edge_list,
                                     sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));

        CUDA_CHECK_RETURN(
                cudaMemcpy(work_queue_edges[k], graph->work_queue_edges, sizeof(size_t) * num_edges, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(num_work_items[k], &h_num_work_items, sizeof(unsigned long long int), cudaMemcpyHostToDevice));

    }

    const size_t edgeCount = (num_edges + BLOCK_SIZE_EDGE_STREAMING - 1)/ BLOCK_SIZE_EDGE_STREAMING;
//    const int nodeCount = (num_vertices + BLOCK_SIZE_EDGE_STREAMING - 1) / BLOCK_SIZE_EDGE_STREAMING;

    //const int partitionSize = (num_vertices + NUM_THREAD_PARTITIONS - 1) / NUM_THREAD_PARTITIONS;
    //const int partitionCount = (partitionSize + BLOCK_SIZE_NODE_STREAMING - 1) / BLOCK_SIZE_NODE_STREAMING;
    const size_t edgePartitionSize = (num_edges + num_devices - 1) / num_devices;
    const size_t edgePartitionCount = (edgePartitionSize + BLOCK_SIZE_EDGE_STREAMING - 1) / BLOCK_SIZE_EDGE_STREAMING;
    const size_t nodePartitionSize = (num_vertices + num_devices - 1) / num_devices;
    const size_t nodePartitionCount = (nodePartitionSize + BLOCK_SIZE_NODE_EDGE_STREAMING - 1) / BLOCK_SIZE_NODE_EDGE_STREAMING;

    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE_EDGE_STREAMING, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    size_t reduceSmemSize = (BLOCK_SIZE_EDGE_STREAMING <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE_EDGE_STREAMING * sizeof(float);

    size_t curr_index = 0;
    size_t curr_node_index = 0;
    // init streams + data
    for(i = 0; i < num_devices; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

        thread_data[i].streamEdgeCount = edgePartitionCount;
        thread_data[i].begin_index = curr_index;
        thread_data[i].end_index = curr_index + edgePartitionSize;
        if(thread_data[i].end_index > num_edges) {
            thread_data[i].end_index = num_edges;
        }
        curr_index += edgePartitionSize;

        thread_data[i].edge_joint_probability_dim_x = edge_joint_probability_dim_x;
        thread_data[i].edge_joint_probability_dim_y = edge_joint_probability_dim_y;

        thread_data[i].num_vertices = num_vertices;
        thread_data[i].num_edges = num_edges;

        thread_data[i].dest_nodes_to_edges_edges = dest_nodes_to_edges_edges[i];
        thread_data[i].dest_nodes_to_edges_nodes = dest_nodes_to_edges_nodes[i];
        thread_data[i].edges_src_index = edges_src_index[i];
        thread_data[i].edges_dest_index = edges_dest_index[i];
        thread_data[i].num_work_items = num_work_items[i];
        thread_data[i].work_queue_edges = work_queue_edges[i];
        thread_data[i].work_queue_scratch = work_queue_scratch[i];

        thread_data[i].current_edge_messages = current_messages[i];
        thread_data[i].current_edge_messages_size = current_messages_size;
        thread_data[i].current_edge_messages_previous = current_messages_previous[i];
        thread_data[i].current_edge_messages_current = current_messages_current[i];

        thread_data[i].node_states = node_states[i];

        thread_data[i].stream = streams[i];

        node_thread_data[i].begin_index = curr_node_index;
        node_thread_data[i].end_index = curr_node_index + nodePartitionSize;
        if(node_thread_data[i].end_index > num_vertices) {
            node_thread_data[i].end_index = num_vertices;
        }
        curr_node_index += nodePartitionSize;
        node_thread_data[i].streamNodeCount = nodePartitionCount;
        node_thread_data[i].stream = streams[i];

        node_thread_data[i].node_messages = node_states[i];
        node_thread_data[i].node_messages_size = node_states_size;

        node_thread_data[i].current_edge_messages = current_messages[i];
        node_thread_data[i].current_edge_messages_current = current_messages_current[i];
        node_thread_data[i].current_edge_messages_previous = current_messages_previous[i];


        node_thread_data[i].dest_nodes_to_edges_nodes = dest_nodes_to_edges_edges[i];
        node_thread_data[i].dest_nodes_to_edges_edges = dest_nodes_to_edges_edges[i];
        node_thread_data[i].num_vertices = num_vertices;
        node_thread_data[i].num_edges = num_edges;

        node_thread_data[i].num_work_items = NULL;
        node_thread_data[i].work_queue_scratch = NULL;
        node_thread_data[i].work_queue_nodes = NULL;
        node_thread_data[i].src_nodes_to_edges_edges = NULL;
        node_thread_data[i].src_nodes_to_edges_nodes = NULL;
        node_thread_data[i].buffers = NULL;
    }

    for(i = 0; i < max_iterations; i+= BATCH_SIZE_EDGE){
        for(j = 0; j < BATCH_SIZE_EDGE; ++j) {
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_create(&threads[k], NULL, launch_send_message_kernel, &thread_data[k]);
                if(retval) {
                    fprintf(stderr, "Error creating send message thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining send message thread %ld: %d\n", k, retval);
                    return 1;
                }
            }

            // synchronize state

            // first get data from devices


            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                for(m = 0; m < num_devices; ++m) {
                    if(m == k) {
                        continue;
                    }
                    l = thread_data[k].begin_index;
                    if(thread_data[k].end_index > num_edges) {
                        copy_amount = num_edges - l;
                    }
                    else {
                        copy_amount = thread_data[k].end_index - l;
                    }

                    CUDA_CHECK_RETURN(cudaMemcpy(&(current_messages[m][l]), &(current_messages[k][l]),
                                                 sizeof(belief) * copy_amount,
                                                 cudaMemcpyDeviceToDevice));
                    CUDA_CHECK_RETURN(cudaMemcpy(&(current_messages_previous[m][l]), &(current_messages[k][l]),
                                                 sizeof(float) * copy_amount, cudaMemcpyDeviceToDevice));
                    CUDA_CHECK_RETURN(cudaMemcpy(&(current_messages_current[m][l]), &(current_messages[k][l]),
                                                 sizeof(float) * copy_amount, cudaMemcpyDeviceToDevice));

                }

                retval = pthread_create(&threads[k], NULL, launch_combine_message_kernel, &thread_data[k]);
                if(retval) {
                    fprintf(stderr, "Error creating combine message thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining combine message thread %ld: %d\n", k, retval);
                    return 1;
                }
            }

            // synchronize state

            //marginalize_loop_node_edge_kernel<<<nodeCount, BLOCK_SIZE>>>(node_states, num_vars, num_vertices);
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_create(&threads[k], NULL, launch_marginalize_streaming_kernel, &node_thread_data[k]);
                if(retval) {
                    fprintf(stderr, "Error creating marginalize node thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining marginalize node thread %ld: %d\n", k, retval);
                    return 1;
                }
            }


            // first get data back from devices
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                for(m = 0; m < num_devices; ++m) {
                    if(m == k) {
                        continue;
                    }
                    // update beliefs
                    for (l = thread_data[k].begin_index; l < thread_data[k].end_index; ++l) {
                        curr_index = graph->edges_dest_index[l];
                        if (curr_index < graph->current_num_vertices) {
                            CUDA_CHECK_RETURN(cudaMemcpy(&(h_node_states[m][l]), &(h_node_states[k][l]), sizeof(struct belief), cudaMemcpyDeviceToDevice));
                        }
                    }
                }
            }

            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                update_work_queue_edges_cuda_kernel << < edgeCount, BLOCK_SIZE >> >
                                                                    (work_queue_edges[k], num_work_items[k], work_queue_scratch[k], current_messages_previous[k], current_messages_current[k], graph->current_num_edges);
                test_error();
            }
            num_iter++;
        }
        cudaSetDevice(0);
        calculate_delta_6 << < dimReduceGrid, dimReduceBlock, reduceSmemSize >> >
                                                              (current_messages_previous[0], current_messages_current[0], delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        test_error();
        CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));

        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            break;
        }
        previous_delta = host_delta;
    }

    for(k = 0; k < num_devices; ++k) {
        cudaStreamDestroy(streams[k]);
    }
    for(k = 0; k < num_devices; ++k) {
        // copy data back
        CUDA_CHECK_RETURN(cudaMemcpy(graph->node_states, node_states[k], sizeof(struct belief) * num_vertices,
                                     cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(graph->edges_messages, current_messages[k], sizeof(struct belief) * num_edges,
                                     cudaMemcpyDeviceToHost));

        CUDA_CHECK_RETURN(cudaFree(current_messages[k]));
        CUDA_CHECK_RETURN(cudaFree(current_messages_previous[k]));
        CUDA_CHECK_RETURN(cudaFree(current_messages_current[k]));

        CUDA_CHECK_RETURN(cudaFree(node_states[k]));

        CUDA_CHECK_RETURN(cudaFreeHost(h_current_messages[k]));
        CUDA_CHECK_RETURN(cudaFreeHost(h_current_messages_previous[k]));
        CUDA_CHECK_RETURN(cudaFreeHost(h_current_messages_current[k]));

        CUDA_CHECK_RETURN(cudaFree(edges_src_index[k]));
        CUDA_CHECK_RETURN(cudaFree(edges_dest_index[k]));

        CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_nodes[k]));
        CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_edges[k]));

        CUDA_CHECK_RETURN(cudaFree(work_queue_edges[k]));
        CUDA_CHECK_RETURN(cudaFree(work_queue_scratch[k]));
        CUDA_CHECK_RETURN(cudaFree(num_work_items[k]));

        CUDA_CHECK_RETURN(cudaFreeHost(h_node_states[k]));
    }

    CUDA_CHECK_RETURN(cudaFree(delta));
    CUDA_CHECK_RETURN(cudaFree(delta_array));


    free(current_messages);
    free(current_messages_previous);
    free(current_messages_current);


    free(node_states);

    free(edges_src_index);
    free(edges_dest_index);
    free(dest_nodes_to_edges_nodes);
    free(dest_nodes_to_edges_edges);
    free(work_queue_edges);
    free(work_queue_scratch);
    free(num_work_items);

    free(h_current_messages);
    free(h_current_messages_current);
    free(h_current_messages_previous);

    free(h_node_states);

    CUDA_CHECK_RETURN(cudaHostUnregister(graph->edges_messages));

    CUDA_CHECK_RETURN(cudaHostUnregister(graph->node_states));

    CUDA_CHECK_RETURN(cudaHostUnregister(&h_num_work_items));
    CUDA_CHECK_RETURN(cudaHostUnregister(graph->work_queue_edges));

    free(threads);
    free(streams);
    free(thread_data);
    free(node_thread_data);

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
int loopy_propagate_until_cuda_edge_openmpi(Graph_t graph, const float convergence, const int max_iterations, const int my_rank,
        int num_ranks, int num_devices){
    int num_iter;
    size_t i, j, k, l;
    float *delta;
    float *delta_array;
    float previous_delta, host_delta;
    char is_pow_2;
    float node_difference;

    struct belief ** current_messages;
    float ** current_messages_previous;
    float ** current_messages_current;

    struct belief * recv_current_messages;
    float * recv_current_messages_previous;
    float * recv_current_messages_current;

    struct belief ** h_current_messages;
    float ** h_current_messages_previous;
    float ** h_current_messages_current;

    struct belief ** node_states;

    struct belief ** h_node_states;

    struct belief * recv_node_states;

    size_t ** edges_src_index;
    size_t ** edges_dest_index;
    size_t ** dest_nodes_to_edges_nodes;
    size_t ** dest_nodes_to_edges_edges;

    size_t ** work_queue_edges;
    size_t ** work_queue_scratch;
    unsigned long long int ** num_work_items;

    int retval;

    size_t end_index;

    init_work_queue_edges(graph);
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // allocate device arrays


    current_messages = (struct belief **)malloc(num_devices * sizeof(struct belief *));
    assert(current_messages);
    current_messages_previous = (float **)malloc(num_devices * sizeof(float *));
    assert(current_messages_previous);
    current_messages_current = (float **)malloc(num_devices * sizeof(float *));
    assert(current_messages_current);

    h_current_messages = (struct belief **)malloc(num_devices * sizeof(struct belief *));
    assert(h_current_messages);
    h_current_messages_previous = (float **)malloc(num_devices * sizeof(float *));
    assert(h_current_messages_previous);
    h_current_messages_current = (float **)malloc(num_devices * sizeof(float *));
    assert(h_current_messages_current);

    node_states = (struct belief **)malloc(num_devices * sizeof(struct belief *));
    assert(node_states);

    h_node_states = (struct belief **)malloc(num_devices * sizeof(struct belief *));
    assert(h_node_states);

    recv_current_messages = (struct belief *)malloc(num_ranks * graph->current_num_edges * sizeof(struct belief));
    assert(recv_current_messages);
    recv_current_messages_previous = (float *)malloc(num_ranks * graph->current_num_edges * sizeof(float));
    assert(recv_current_messages_previous);
    recv_current_messages_current = (float *)malloc(num_ranks * graph->current_num_edges * sizeof(float));
    assert(recv_current_messages_previous);

    recv_node_states = (struct belief *)malloc(num_ranks * graph->current_num_vertices * sizeof(struct belief));
    assert(recv_node_states);

    edges_src_index = (size_t **)malloc(num_devices * sizeof(size_t *));
    assert(edges_src_index);
    edges_dest_index = (size_t **)malloc(num_devices * sizeof(size_t *));
    assert(edges_dest_index);
    dest_nodes_to_edges_nodes = (size_t **)malloc(num_devices * sizeof(size_t *));
    assert(dest_nodes_to_edges_nodes);
    dest_nodes_to_edges_edges = (size_t **)malloc(num_devices * sizeof(size_t *));
    assert(dest_nodes_to_edges_edges);

    work_queue_edges = (size_t **)malloc(num_devices * sizeof(size_t *));
    assert(work_queue_edges);
    work_queue_scratch = (size_t **)malloc(num_devices * sizeof(size_t *));
    assert(work_queue_scratch);
    num_work_items = (unsigned long long int **)malloc(num_devices * sizeof(unsigned long long int *));
    assert(num_work_items);


    unsigned long long int h_num_work_items = (unsigned long long int)graph->num_work_items_edges;
    // pin host memory
    CUDA_CHECK_RETURN(cudaHostRegister(graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(graph->work_queue_edges, sizeof(size_t) * graph->current_num_edges, cudaHostRegisterDefault));
    CUDA_CHECK_RETURN(cudaHostRegister(&h_num_work_items, sizeof(unsigned long long int), cudaHostRegisterDefault));

    host_delta = 0.0f;
    previous_delta = INFINITY;

    //struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    const size_t num_vertices = graph->current_num_vertices;
    const size_t num_edges = graph->current_num_edges;
    const size_t edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    const size_t edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;
    const size_t current_messages_size = graph->edges_messages_size;
    const size_t node_states_size = graph->node_states_size;


    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * num_devices);
    cudaStream_t *streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * num_devices);
    struct edge_stream_data *thread_data = (struct edge_stream_data *)malloc(sizeof(struct edge_stream_data) * num_devices);
    struct node_stream_data *node_thread_data = (struct node_stream_data *)malloc(sizeof(struct node_stream_data) * num_devices);

    if(my_rank == 0) {
        cudaSetDevice(0);

        CUDA_CHECK_RETURN(cudaMalloc((void **) &delta, sizeof(float)));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &delta_array, sizeof(float) * num_edges));

    }

    for(k = 0; k < num_devices; ++k) {
        cudaSetDevice(k);
        // allocate data
        CUDA_CHECK_RETURN(cudaMalloc((void **) &edges_src_index[k], sizeof(size_t) * graph->current_num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &edges_dest_index[k], sizeof(size_t) * graph->current_num_edges));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &node_states[k], sizeof(struct belief) * graph->current_num_vertices));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &current_messages[k], sizeof(struct belief) * graph->current_num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &current_messages_previous[k], sizeof(float) * graph->current_num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &current_messages_current[k], sizeof(float) * graph->current_num_edges));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &dest_nodes_to_edges_nodes[k], sizeof(size_t) * graph->current_num_vertices));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &dest_nodes_to_edges_edges[k], sizeof(size_t) * graph->current_num_edges));

        CUDA_CHECK_RETURN(cudaMalloc((void **) &work_queue_edges[k], sizeof(size_t) * num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &work_queue_scratch[k], sizeof(size_t) * num_edges));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &num_work_items[k], sizeof(unsigned long long int)));

        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &h_node_states[k], sizeof(struct belief) * num_vertices, cudaHostAllocDefault));

        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &h_current_messages[k], sizeof(struct belief) * num_edges, cudaHostAllocDefault));
        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &h_current_messages_previous[k], sizeof(float) * num_edges, cudaHostAllocDefault));
        CUDA_CHECK_RETURN(cudaHostAlloc((void **) &h_current_messages_current[k], sizeof(float) * num_edges, cudaHostAllocDefault));


        // copy data
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(edge_joint_probability, &(graph->edge_joint_probability), sizeof(struct joint_probability)));

        CUDA_CHECK_RETURN(
                cudaMemcpy(node_states[k], graph->node_states, sizeof(struct belief) * graph->current_num_vertices,
                           cudaMemcpyHostToDevice));

        CUDA_CHECK_RETURN(
                cudaMemcpy(current_messages[k], graph->edges_messages, sizeof(struct belief) * graph->current_num_edges,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(current_messages_previous[k], graph->edges_messages_previous, sizeof(float) * graph->current_num_edges,
                           cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(
                cudaMemcpy(current_messages_current[k], graph->edges_messages_current, sizeof(float) * graph->current_num_edges,
                           cudaMemcpyHostToDevice));

        CUDA_CHECK_RETURN(cudaMemcpy(edges_src_index[k], graph->edges_src_index, sizeof(size_t) * graph->current_num_edges,
                                     cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(edges_dest_index[k], graph->edges_dest_index, sizeof(size_t) * graph->current_num_edges,
                                     cudaMemcpyHostToDevice));

        CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_nodes[k], graph->dest_nodes_to_edges_node_list,
                                     sizeof(size_t) * graph->current_num_vertices, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_edges[k], graph->dest_nodes_to_edges_edge_list,
                                     sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));

        CUDA_CHECK_RETURN(
                cudaMemcpy(work_queue_edges[k], graph->work_queue_edges, sizeof(size_t) * num_edges, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(num_work_items[k], &h_num_work_items, sizeof(unsigned long long int), cudaMemcpyHostToDevice));

    }

    const size_t edgeCount = (num_edges + BLOCK_SIZE_EDGE_STREAMING - 1)/ BLOCK_SIZE_EDGE_STREAMING;
//    const int nodeCount = (num_vertices + BLOCK_SIZE_EDGE_STREAMING - 1) / BLOCK_SIZE_EDGE_STREAMING;

    //const int partitionSize = (num_vertices + NUM_THREAD_PARTITIONS - 1) / NUM_THREAD_PARTITIONS;
    //const int partitionCount = (partitionSize + BLOCK_SIZE_NODE_STREAMING - 1) / BLOCK_SIZE_NODE_STREAMING;
    const size_t edgeRankPartitionSize = (num_edges + num_ranks - 1) / num_ranks;
    const size_t edgePartitionSize = (edgeRankPartitionSize + num_devices - 1) / num_devices;
    const size_t edgePartitionCount = (edgePartitionSize + BLOCK_SIZE_EDGE_STREAMING - 1) / BLOCK_SIZE_EDGE_STREAMING;
    const size_t nodeRankPartitionSize = (num_vertices + num_ranks - 1) / num_ranks;
    const size_t nodePartitionSize = (nodeRankPartitionSize + num_devices - 1) / num_devices;
    const size_t nodePartitionCount = (nodePartitionSize + BLOCK_SIZE_NODE_EDGE_STREAMING - 1) / BLOCK_SIZE_NODE_EDGE_STREAMING;

    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE_EDGE_STREAMING, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    size_t reduceSmemSize = (BLOCK_SIZE_EDGE_STREAMING <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE_EDGE_STREAMING * sizeof(float);

    size_t curr_index = edgeRankPartitionSize * my_rank;
    size_t curr_node_index = nodeRankPartitionSize * my_rank;
    // init streams + data
    for(i = 0; i < num_devices; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

        thread_data[i].streamEdgeCount = edgePartitionCount;
        thread_data[i].begin_index = curr_index;
        thread_data[i].end_index = curr_index + edgePartitionSize;
        if(thread_data[i].end_index > num_edges) {
            thread_data[i].end_index = num_edges;
        }
        curr_index += edgePartitionSize;

        thread_data[i].edge_joint_probability_dim_x = edge_joint_probability_dim_x;
        thread_data[i].edge_joint_probability_dim_y = edge_joint_probability_dim_y;

        thread_data[i].num_vertices = num_vertices;
        thread_data[i].num_edges = num_edges;

        thread_data[i].dest_nodes_to_edges_edges = dest_nodes_to_edges_edges[i];
        thread_data[i].dest_nodes_to_edges_nodes = dest_nodes_to_edges_nodes[i];
        thread_data[i].edges_src_index = edges_src_index[i];
        thread_data[i].edges_dest_index = edges_dest_index[i];
        thread_data[i].num_work_items = num_work_items[i];
        thread_data[i].work_queue_edges = work_queue_edges[i];
        thread_data[i].work_queue_scratch = work_queue_scratch[i];

        thread_data[i].current_edge_messages = current_messages[i];
        thread_data[i].current_edge_messages_size = current_messages_size;
        thread_data[i].current_edge_messages_previous = current_messages_previous[i];
        thread_data[i].current_edge_messages_current = current_messages_current[i];

        thread_data[i].node_states = node_states[i];

        thread_data[i].stream = streams[i];

        node_thread_data[i].begin_index = curr_node_index;
        node_thread_data[i].end_index = curr_node_index + nodePartitionSize;
        if(node_thread_data[i].end_index > num_vertices) {
            node_thread_data[i].end_index = num_vertices;
        }
        curr_node_index += nodePartitionSize;
        node_thread_data[i].streamNodeCount = nodePartitionCount;
        node_thread_data[i].stream = streams[i];

        node_thread_data[i].node_messages = node_states[i];
        node_thread_data[i].node_messages_size = node_states_size;

        node_thread_data[i].current_edge_messages = current_messages[i];
        node_thread_data[i].current_edge_messages_current = current_messages_current[i];
        node_thread_data[i].current_edge_messages_previous = current_messages_previous[i];


        node_thread_data[i].dest_nodes_to_edges_nodes = dest_nodes_to_edges_edges[i];
        node_thread_data[i].dest_nodes_to_edges_edges = dest_nodes_to_edges_edges[i];
        node_thread_data[i].num_vertices = num_vertices;
        node_thread_data[i].num_edges = num_edges;

        node_thread_data[i].num_work_items = NULL;
        node_thread_data[i].work_queue_scratch = NULL;
        node_thread_data[i].work_queue_nodes = NULL;
        node_thread_data[i].src_nodes_to_edges_edges = NULL;
        node_thread_data[i].src_nodes_to_edges_nodes = NULL;
        node_thread_data[i].buffers = NULL;
    }

    for(i = 0; i < max_iterations; i+= BATCH_SIZE_EDGE){
        for(j = 0; j < BATCH_SIZE_EDGE; ++j) {
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_create(&threads[k], NULL, launch_send_message_kernel, &thread_data[k]);
                if(retval) {
                    fprintf(stderr, "Error creating send message thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining send message thread %ld: %d\n", k, retval);
                    return 1;
                }
            }

            // synchronize state

            // first get data from devices
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                // first get subset back
                CUDA_CHECK_RETURN(cudaMemcpy(h_current_messages[k], current_messages[k], sizeof(struct belief) * graph->current_num_edges, cudaMemcpyDeviceToHost));
                CUDA_CHECK_RETURN(cudaMemcpy(h_current_messages_current[k], current_messages_current[k], sizeof(float) * graph->current_num_edges, cudaMemcpyDeviceToHost));
                CUDA_CHECK_RETURN(cudaMemcpy(h_current_messages_previous[k], current_messages_previous[k], sizeof(float) * graph->current_num_edges, cudaMemcpyDeviceToHost));
                // copy
                for(l = thread_data[k].begin_index; l < thread_data[k].end_index && l < graph->current_num_edges; ++l) {
                    memcpy(&(graph->edges_messages[l]), &(h_current_messages[k][l]), sizeof(struct belief));
                    memcpy(&(graph->edges_messages_previous[l]), &(h_current_messages_previous[k][l]), sizeof(float));
                    memcpy(&(graph->edges_messages_current[l]), &(h_current_messages_current[k][l]), sizeof(float));
                }
            }
            // send it to others
            MPICHECK(MPI_Allgather(graph->edges_messages, (int)graph->current_num_edges, belief_struct, recv_current_messages, (int)graph->current_num_edges, belief_struct, MPI_COMM_WORLD));
            MPICHECK(MPI_Allgather(graph->edges_messages_previous, (int)graph->current_num_edges, MPI_FLOAT, recv_current_messages_previous, (int)graph->current_num_edges, MPI_FLOAT, MPI_COMM_WORLD));
            MPICHECK(MPI_Allgather(graph->edges_messages_current, (int)graph->current_num_edges, MPI_FLOAT, recv_current_messages_current, (int)graph->current_num_edges, MPI_FLOAT, MPI_COMM_WORLD));
            // rebuild edges messages
            for(k = 0; k < num_ranks; ++k) {
                for(l = edgeRankPartitionSize * my_rank; l < edgeRankPartitionSize * (k + 1) && l < graph->current_num_edges; ++l) {
                    memcpy(&(graph->edges_messages[l]), &(recv_current_messages[graph->current_num_edges * k + l]), sizeof(struct belief));
                    memcpy(&(graph->edges_messages_previous[l]), &(recv_current_messages_previous[graph->current_num_edges * k + l]), sizeof(float));
                    memcpy(&(graph->edges_messages_current[l]), &(recv_current_messages_current[graph->current_num_edges * k + l]), sizeof(float));
                }
            }

            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                CUDA_CHECK_RETURN(cudaMemcpy(current_messages[k], graph->edges_messages, sizeof(belief) * graph->current_num_edges, cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpy(current_messages_previous[k], graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpy(current_messages_current[k], graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
                retval = pthread_create(&threads[k], NULL, launch_combine_message_kernel, &thread_data[k]);
                if(retval) {
                    fprintf(stderr, "Error creating combine message thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining combine message thread %ld: %d\n", k, retval);
                    return 1;
                }
            }

            // synchronize state

            // first get data back from devices
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                CUDA_CHECK_RETURN(cudaMemcpy(h_node_states[k], thread_data[k].node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyDeviceToHost));
                // update beliefs
                for(l = thread_data[k].begin_index; l < thread_data[k].end_index; ++l) {
                    curr_index = graph->edges_dest_index[l];
                    if(curr_index < graph->current_num_vertices) {
                        memcpy(&(graph->node_states[curr_index]), &(h_node_states[k][curr_index]), sizeof(struct belief));
                    }
                }
            }
            // send it to others
            MPICHECK(MPI_Allgather(graph->node_states, graph->current_num_vertices, belief_struct, recv_node_states, graph->current_num_vertices, belief_struct, MPI_COMM_WORLD));
            // rebuild
            for(k = 0; k < num_ranks; ++k) {
                end_index = edgeRankPartitionSize * (k + 1);
                if(end_index > graph->current_num_edges) {
                    end_index = graph->current_num_vertices;
                }
                for(l = edgeRankPartitionSize * k; l < end_index; ++l) {
                    curr_node_index = graph->edges_dest_index[l];
                    memcpy(&(graph->node_states[curr_node_index]), &(recv_node_states[k * graph->current_num_vertices + curr_node_index]), sizeof(struct belief));
                }
            }
            // send it gpu
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                CUDA_CHECK_RETURN(cudaMemcpy(node_states[k], graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));
            }

            if(my_rank == 0) {
                CUDA_CHECK_RETURN(cudaSetDevice(0));
                update_work_queue_cuda_kernel << < edgeCount, BLOCK_SIZE_EDGE_STREAMING >> >
                                                              (work_queue_edges[0], num_work_items[0], work_queue_scratch[0], current_messages_previous[0], current_messages_current[0], num_edges);
                test_error();
                // copy back
                CUDA_CHECK_RETURN(cudaMemcpy(graph->work_queue_edges, work_queue_edges[0], sizeof(size_t) * graph->current_num_edges, cudaMemcpyDeviceToHost));
                CUDA_CHECK_RETURN(cudaMemcpy(&graph->num_work_items_edges, num_work_items[0], sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
            }

            MPICHECK(MPI_Bcast(&h_num_work_items, h_num_work_items, MPI_INT64_T, 0, MPI_COMM_WORLD));
            MPICHECK(MPI_Bcast(&h_num_work_items, 1, MPI_INT, 0, MPI_COMM_WORLD));
            // update gpu
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                CUDA_CHECK_RETURN(cudaMemcpy(work_queue_edges[k], graph->work_queue_edges, sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpy(num_work_items[k], &h_num_work_items, sizeof(unsigned long long int), cudaMemcpyHostToDevice));
            }

            //marginalize_loop_node_edge_kernel<<<nodeCount, BLOCK_SIZE>>>(node_states, num_vars, num_vertices);
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_create(&threads[k], NULL, launch_marginalize_streaming_kernel, &node_thread_data[k]);
                if(retval) {
                    fprintf(stderr, "Error creating marginalize node thread %ld: %d\n", k, retval);
                    return 1;
                }
            }
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                retval = pthread_join(threads[k], NULL);
                if(retval) {
                    fprintf(stderr, "Error joining marginalize node thread %ld: %d\n", k, retval);
                    return 1;
                }
            }


            // first get data back from devices
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                CUDA_CHECK_RETURN(cudaMemcpy(h_node_states[k], thread_data[k].node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyDeviceToHost));
                // update beliefs
                for(l = thread_data[k].begin_index; l < thread_data[k].end_index; ++l) {
                    curr_index = graph->edges_dest_index[l];
                    if(curr_index < graph->current_num_vertices) {
                        memcpy(&(graph->node_states[curr_index]), &(h_node_states[k][curr_index]), sizeof(struct belief));
                    }
                }
            }
            // send it to others
            MPICHECK(MPI_Allgather(graph->node_states, graph->current_num_vertices, belief_struct, recv_node_states, graph->current_num_vertices, belief_struct, MPI_COMM_WORLD));
            // rebuild
            for(l = 0; l < graph->current_num_vertices; ++l) {
                node_difference = 0.0f;
                for(k = 0; k < num_ranks && node_difference < NODE_DIFFERENCE_THRESHOLD; ++k) {
                    node_difference = difference(&(graph->node_states[l]), graph->node_states_size, &(recv_node_states[k * graph->current_num_vertices + l]), graph->node_states_size);
                    if(node_difference >= NODE_DIFFERENCE_THRESHOLD) {
                        memcpy(&(graph->node_states[l]), &(recv_node_states[k * graph->current_num_vertices + l]), sizeof(struct belief));
                    }
                }
            }
            // send it gpu
            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                CUDA_CHECK_RETURN(cudaMemcpy(node_states[k], graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));
            }

            for(k = 0; k < num_devices; ++k) {
                CUDA_CHECK_RETURN(cudaSetDevice(k));
                update_work_queue_edges_cuda_kernel << < edgeCount, BLOCK_SIZE >> >
                                                                    (work_queue_edges[k], num_work_items[k], work_queue_scratch[k], current_messages_previous[k], current_messages_current[k], graph->current_num_edges);
                test_error();
            }
            num_iter++;
        }
        if(my_rank == 0) {
            cudaSetDevice(0);
            calculate_delta_6 << < dimReduceGrid, dimReduceBlock, reduceSmemSize >> >
                                                                  (current_messages_previous[0], current_messages_current[0], delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
            //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
            //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
            test_error();
            CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
            //   printf("Current delta: %f\n", host_delta);
        }
        MPICHECK(MPI_Bcast(&host_delta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD));
        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            break;
        }
        previous_delta = host_delta;
    }

    for(k = 0; k < num_devices; ++k) {
        cudaStreamDestroy(streams[k]);
    }
    for(k = 0; k < num_devices; ++k) {
        // copy data back
        CUDA_CHECK_RETURN(cudaMemcpy(graph->node_states, node_states[k], sizeof(struct belief) * num_vertices,
                                     cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(graph->edges_messages, current_messages[k], sizeof(struct belief) * num_edges,
                                     cudaMemcpyDeviceToHost));

        CUDA_CHECK_RETURN(cudaFree(current_messages[k]));
        CUDA_CHECK_RETURN(cudaFree(current_messages_previous[k]));
        CUDA_CHECK_RETURN(cudaFree(current_messages_current[k]));

        CUDA_CHECK_RETURN(cudaFree(node_states[k]));

        CUDA_CHECK_RETURN(cudaFreeHost(h_current_messages[k]));
        CUDA_CHECK_RETURN(cudaFreeHost(h_current_messages_previous[k]));
        CUDA_CHECK_RETURN(cudaFreeHost(h_current_messages_current[k]));

        CUDA_CHECK_RETURN(cudaFree(edges_src_index[k]));
        CUDA_CHECK_RETURN(cudaFree(edges_dest_index[k]));

        CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_nodes[k]));
        CUDA_CHECK_RETURN(cudaFree(dest_nodes_to_edges_edges[k]));

        CUDA_CHECK_RETURN(cudaFree(work_queue_edges[k]));
        CUDA_CHECK_RETURN(cudaFree(work_queue_scratch[k]));
        CUDA_CHECK_RETURN(cudaFree(num_work_items[k]));

        CUDA_CHECK_RETURN(cudaFreeHost(h_node_states[k]));
    }

    if(my_rank == 0) {
        CUDA_CHECK_RETURN(cudaFree(delta));
        CUDA_CHECK_RETURN(cudaFree(delta_array));
    }

    // synchronize across cluster
    MPICHECK(MPI_Allgather(graph->node_states, graph->current_num_vertices, belief_struct, recv_node_states, graph->current_num_vertices, belief_struct, MPI_COMM_WORLD));


    free(current_messages);
    free(current_messages_previous);
    free(current_messages_current);


    free(node_states);


    free(edges_src_index);
    free(edges_dest_index);
    free(dest_nodes_to_edges_nodes);
    free(dest_nodes_to_edges_edges);
    free(work_queue_edges);
    free(work_queue_scratch);
    free(num_work_items);

    free(recv_current_messages);
    free(recv_current_messages_previous);
    free(recv_current_messages_current);

    free(recv_node_states);

    free(h_current_messages);
    free(h_current_messages_current);
    free(h_current_messages_previous);

    free(h_node_states);

    CUDA_CHECK_RETURN(cudaHostUnregister(graph->edges_messages));

    CUDA_CHECK_RETURN(cudaHostUnregister(graph->node_states));

    CUDA_CHECK_RETURN(cudaHostUnregister(&h_num_work_items));
    CUDA_CHECK_RETURN(cudaHostUnregister(graph->work_queue_edges));

    free(threads);
    free(streams);
    free(thread_data);
    free(node_thread_data);

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
int page_rank_until_cuda_edge(Graph_t graph, const float convergence, const int max_iterations){
    int num_iter;
    size_t i, j;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct belief * current_messages;
    float * current_messages_previous;
    float * current_messages_current;

    struct belief * node_states;

    size_t * edges_src_index;
    size_t * edges_dest_index;
    size_t * dest_nodes_to_edges_nodes;
    size_t * dest_nodes_to_edges_edges;

    host_delta = 0.0f;
    previous_delta = INFINITY;

    //struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    const size_t num_vertices = graph->current_num_vertices;
    const size_t num_edges = graph->current_num_edges;
    const size_t edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    const size_t edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;
    const size_t current_messages_size = graph->edges_messages_size;
    const size_t node_states_size = graph->node_states_size;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_src_index, sizeof(size_t) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_dest_index, sizeof(size_t) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_previous, sizeof(float) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_current, sizeof(float) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_nodes, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_edges, sizeof(size_t) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));


    // copy data
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(edge_joint_probability, &(graph->edge_joint_probability), sizeof(struct joint_probability)));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_previous, graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_current, graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(edges_src_index, graph->edges_src_index, sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(edges_dest_index, graph->edges_dest_index, sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(size_t) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));

    const int edgeCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    const int nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE_EDGE){
        for(j = 0; j < BATCH_SIZE_EDGE; ++j) {
            send_message_for_edge_iteration_cuda_kernel<<<edgeCount, BLOCK_SIZE >>>(num_edges, edges_src_index,
                    node_states,
                    edge_joint_probability_dim_x, edge_joint_probability_dim_y,
                    current_messages, current_messages_previous, current_messages_current);
            test_error();
            combine_loopy_edge_cuda_kernel<<<edgeCount, BLOCK_SIZE>>>(num_edges, edges_dest_index, current_messages, current_messages_size, node_states);
            test_error();
            //marginalize_loop_node_edge_kernel<<<nodeCount, BLOCK_SIZE>>>(node_states, num_vars, num_vertices);
            marginalize_page_rank_nodes<<<nodeCount, BLOCK_SIZE>>>(node_states, node_states_size,
                    current_messages,
                    dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
            test_error();

            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages_previous, current_messages_current, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
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

    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(current_messages_previous));
    CUDA_CHECK_RETURN(cudaFree(current_messages_current));

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
int viterbi_until_cuda_edge(Graph_t graph, const float convergence, const int max_iterations){
    int num_iter;
    size_t i, j;
    float * delta;
    float * delta_array;
    float previous_delta, host_delta;
    char is_pow_2;

    struct belief * current_messages;
    float * current_messages_previous;
    float * current_messages_current;

    struct belief * node_states;

    size_t * edges_src_index;
    size_t * edges_dest_index;
    size_t * dest_nodes_to_edges_nodes;
    size_t * dest_nodes_to_edges_edges;

    host_delta = 0.0f;
    previous_delta = INFINITY;

    //struct cudaChannelFormatDesc channel_desc_unsigned_int = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    const size_t num_vertices = graph->current_num_vertices;
    const size_t num_edges = graph->current_num_edges;
    const size_t edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
    const size_t edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;
    const size_t current_messages_size = graph->edges_messages_size;
    const size_t node_states_size = graph->node_states_size;

    /*printf("Before=====");
    print_edges(graph);
    print_nodes(graph);*/


    is_pow_2 = num_vertices % 2 == 0;

    // allocate data
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_src_index, sizeof(size_t) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&edges_dest_index, sizeof(size_t) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&node_states, sizeof(struct belief) * graph->current_num_vertices));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages, sizeof(struct belief) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_previous, sizeof(float) * graph->current_num_edges));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&current_messages_current, sizeof(float) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_nodes, sizeof(size_t) * graph->current_num_vertices));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dest_nodes_to_edges_edges, sizeof(size_t) * graph->current_num_edges));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&delta_array, sizeof(float) * num_edges));


    // copy data
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(edge_joint_probability, &(graph->edge_joint_probability), sizeof(struct joint_probability)));

    CUDA_CHECK_RETURN(cudaMemcpy(node_states, graph->node_states, sizeof(struct belief) * graph->current_num_vertices, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(current_messages, graph->edges_messages, sizeof(struct belief) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_previous, graph->edges_messages_previous, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(current_messages_current, graph->edges_messages_current, sizeof(float) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(edges_src_index, graph->edges_src_index, sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(edges_dest_index, graph->edges_dest_index, sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_nodes, graph->dest_nodes_to_edges_node_list, sizeof(size_t) * graph->current_num_vertices, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dest_nodes_to_edges_edges, graph->dest_nodes_to_edges_edge_list, sizeof(size_t) * graph->current_num_edges, cudaMemcpyHostToDevice));

    const int edgeCount = (num_edges + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    const int nodeCount = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_iter = 0;

    dim3 dimReduceBlock(BLOCK_SIZE, 1, 1);
    dim3 dimReduceGrid(edgeCount, 1, 1);
    int reduceSmemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(float) : BLOCK_SIZE * sizeof(float);

    for(i = 0; i < max_iterations; i+= BATCH_SIZE_EDGE){
        for(j = 0; j < BATCH_SIZE_EDGE; ++j) {
            send_message_for_edge_iteration_cuda_kernel<<<edgeCount, BLOCK_SIZE >>>(num_edges, edges_src_index,
                    node_states,
                    edge_joint_probability_dim_x, edge_joint_probability_dim_y,
                    current_messages, current_messages_previous, current_messages_current);
            test_error();
            combine_loopy_edge_cuda_kernel<<<edgeCount, BLOCK_SIZE>>>(num_edges, edges_dest_index, current_messages, current_messages_size, node_states);
            test_error();
            //marginalize_loop_node_edge_kernel<<<nodeCount, BLOCK_SIZE>>>(node_states, num_vars, num_vertices);
            argmax_nodes<<<nodeCount, BLOCK_SIZE>>>(node_states, node_states_size, current_messages,
                    dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, num_vertices, num_edges);
            test_error();

            num_iter++;
        }
        calculate_delta_6<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages_previous, current_messages_current, delta, delta_array, num_edges, is_pow_2, WARP_SIZE);
        //calculate_delta<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        //calculate_delta_simple<<<dimReduceGrid, dimReduceBlock, reduceSmemSize>>>(current_messages, delta, delta_array, num_edges);
        test_error();
        CUDA_CHECK_RETURN(cudaMemcpy(&host_delta, delta, sizeof(float), cudaMemcpyDeviceToHost));
        //   printf("Current delta: %f\n", host_delta);

        if(host_delta < convergence || fabs(host_delta - previous_delta) < convergence){
            marginalize_viterbi_beliefs<<<nodeCount, BLOCK_SIZE >>>(node_states, node_states_size, num_vertices);
            break;
        }
        previous_delta = host_delta;
    }

    // copy data back
    CUDA_CHECK_RETURN(cudaMemcpy(graph->node_states, node_states, sizeof(struct belief) * num_vertices, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(graph->edges_messages, current_messages, sizeof(struct belief) * num_edges, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(current_messages));
    CUDA_CHECK_RETURN(cudaFree(current_messages_current));
    CUDA_CHECK_RETURN(cudaFree(current_messages_previous));

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
    int num_iterations;

    graph = build_graph(expression);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_cuda(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy,%ld,%ld,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
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
    int num_iterations;

    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_cuda(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy,%ld,%ld,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
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
    int num_iterations;

    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_cuda_streaming(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy-streaming,%ld,%ld,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
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
    int num_iterations;

    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_cuda_edge(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy-edge,%ld,%ld,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
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
    int num_iterations;

    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_cuda_edge_streaming(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy-edge-streaming,%ld,%ld,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
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
    int num_iterations;

    graph = parse_graph_from_snap_files(edge_file_name, node_file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_cuda(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy,%ld,%ld,%d,%d,%lf\n", edge_file_name, node_file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
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
    int num_iterations;

    graph = parse_graph_from_snap_files(edge_file_name, node_file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_cuda_edge(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy-edge,%ld,%ld,%d,%d,%lf\n", edge_file_name, node_file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_mtx_files_cuda(const char * edge_mtx, const char *node_mtx, const struct joint_probability * edge_prob, size_t num_src, size_t num_dest, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    int num_iterations;

    graph = build_graph_from_mtx(edge_mtx, node_mtx, edge_prob, num_src, num_dest);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges_no_hsearch(graph);
    set_up_dest_nodes_to_edges_no_hsearch(graph);
    //calculate_diameter(graph);

    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_cuda(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy,%ld,%ld,%d,%d,%lf,%d,%lf,%d,%lf\n", edge_mtx, node_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, graph->max_in_degree, graph->avg_in_degree, graph->max_out_degree, graph->avg_out_degree, num_iterations+1, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_mtx_files_cuda_streaming(const char * edge_mtx, const char *node_mtx, const struct joint_probability * edge_prob, size_t num_src, size_t num_dest, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    int num_iterations;

    graph = build_graph_from_mtx(edge_mtx, node_mtx, edge_prob, num_src, num_dest);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges_no_hsearch(graph);
    set_up_dest_nodes_to_edges_no_hsearch(graph);
    //calculate_diameter(graph);

    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_cuda_streaming(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy-streaming,%ld,%ld,%d,%d,%lf,%d,%lf,%d,%lf\n", edge_mtx, node_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, graph->max_in_degree, graph->avg_in_degree, graph->max_out_degree, graph->avg_out_degree, num_iterations+1, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_mtx_files_cuda_multiple_devices(const char * edge_mtx, const char *node_mtx, const struct joint_probability * edge_prob, size_t num_src, size_t num_dest, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    int num_iterations;

    graph = build_graph_from_mtx(edge_mtx, node_mtx, edge_prob, num_src, num_dest);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges_no_hsearch(graph);
    set_up_dest_nodes_to_edges_no_hsearch(graph);
    //calculate_diameter(graph);


    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_cuda_multiple_devices(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy-multiple-devices,%ld,%ld,%d,%d,%lf,%d,%lf,%d,%lf\n", edge_mtx, node_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, graph->max_in_degree, graph->avg_in_degree, graph->max_out_degree, graph->avg_out_degree, num_iterations+1, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_mtx_files_edge_cuda(const char * edge_mtx, const char * node_mtx, const struct joint_probability * edge_prob, size_t num_src, size_t num_dest, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    int num_iterations;

    graph = build_graph_from_mtx(edge_mtx, node_mtx, edge_prob, num_src, num_dest);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges_no_hsearch(graph);
    set_up_dest_nodes_to_edges_no_hsearch(graph);
    //calculate_diameter(graph);

    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_cuda_edge(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy-edge,%ld,%ld,%d,%d,%lf,%d,%lf,%d,%lf\n", edge_mtx, node_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, graph->max_in_degree, graph->avg_in_degree, graph->max_out_degree, graph->avg_out_degree, num_iterations+1, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_mtx_files_edge_cuda_streaming(const char * edge_mtx, const char * node_mtx, const struct joint_probability * edge_prob, size_t num_src, size_t num_dest, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    int num_iterations;

    graph = build_graph_from_mtx(edge_mtx, node_mtx, edge_prob, num_src, num_dest);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges_no_hsearch(graph);
    set_up_dest_nodes_to_edges_no_hsearch(graph);
    //calculate_diameter(graph);

    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_cuda_edge_streaming(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy-edge-streaming,%ld,%ld,%d,%d,%lf,%d,%lf,%d,%lf\n", edge_mtx, node_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, graph->max_in_degree, graph->avg_in_degree, graph->max_out_degree, graph->avg_out_degree, num_iterations+1, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_mtx_files_edge_cuda_multiple_devices(const char * edge_mtx, const char * node_mtx, const struct joint_probability * edge_prob, size_t num_src, size_t num_dest, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    int num_iterations;

    graph = build_graph_from_mtx(edge_mtx, node_mtx, edge_prob, num_src, num_dest);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges_no_hsearch(graph);
    set_up_dest_nodes_to_edges_no_hsearch(graph);
    //calculate_diameter(graph);

    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_cuda_edge_multiple_devices(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy-edge-multiple-devices,%ld,%ld,%d,%d,%lf,%d,%lf,%d,%lf\n", edge_mtx, node_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, graph->max_in_degree, graph->avg_in_degree, graph->max_out_degree, graph->avg_out_degree, num_iterations+1, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}


static void register_belief() {

    MPI_Type_contiguous(sizeof(struct belief), MPI_BYTE, &belief_struct);
    MPI_Type_commit(&belief_struct);
}

static void register_joint_probability() {
    MPI_Type_contiguous(sizeof(struct joint_probability), MPI_BYTE, &joint_probability_struct);
    MPI_Type_commit(&joint_probability_struct);
}


void run_test_loopy_belief_propagation_mtx_files_cuda_openmpi(const char * edge_mtx, const char *node_mtx, const struct joint_probability * edge_prob, size_t num_src, size_t num_dest, FILE * out,
                                                              int my_rank, int n_ranks, int num_devices){
    // each node runs this....
    Graph_t graph = NULL;
    clock_t start;
    clock_t end;
    double time_elapsed;
    int num_iterations;
    size_t num_vertices, num_edges;
    num_vertices = 0;
    num_edges = 0;

    // set up structs
    register_belief();
    register_joint_probability();

    if(my_rank == 0) {
        graph = build_graph_from_mtx(edge_mtx, node_mtx, edge_prob, num_src, num_dest);
        assert(graph != NULL);
        num_vertices = graph->current_num_vertices;
        num_edges = graph->current_num_edges;
        set_up_src_nodes_to_edges_no_hsearch(graph);
        set_up_dest_nodes_to_edges_no_hsearch(graph);
        init_previous_edge(graph);
        start = clock();
    }

    MPICHECK(MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    assert(num_vertices > 0);
    assert(num_edges > 0);
    if(my_rank > 0) {
        graph = create_graph(num_vertices, num_edges, edge_prob, num_src, num_dest);
        graph->current_num_edges = num_edges;
        graph->current_num_vertices = num_vertices;
    }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    // copy rest of graph
    MPICHECK(MPI_Bcast(&(graph->max_degree), 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->edges_src_index), num_edges, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->edges_dest_index), num_edges, MPI_INT, 0, MPI_COMM_WORLD));

    MPICHECK(MPI_Bcast(&(graph->edges_messages), num_edges, belief_struct, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->edges_messages_size), 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->edges_messages_previous), num_edges, MPI_FLOAT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->edges_messages_current), num_edges, MPI_FLOAT, 0, MPI_COMM_WORLD));

    MPICHECK(MPI_Bcast(&(graph->node_states), num_vertices, belief_struct, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->node_states_size), 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->node_states_previous), num_vertices, MPI_FLOAT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->node_states_current), num_vertices, MPI_FLOAT, 0, MPI_COMM_WORLD));

    MPICHECK(MPI_Bcast(&(graph->src_nodes_to_edges_node_list), num_vertices, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->src_nodes_to_edges_edge_list), num_edges, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->dest_nodes_to_edges_edge_list), num_edges, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->dest_nodes_to_edges_node_list), num_vertices, MPI_INT, 0, MPI_COMM_WORLD));

    // wait until everyone is set up
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    num_iterations = loopy_propagate_until_cuda_openmpi(graph, PRECISION, NUM_ITERATIONS, my_rank, n_ranks, num_devices);

    if(my_rank == 0) {
        end = clock();
        time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;

        fprintf(out, "%s-%s,loopy-openmpi,%ld,%ld,%d,%d,%lf,%d,%lf,%d,%lf\n", edge_mtx, node_mtx,
                graph->current_num_vertices, graph->current_num_edges, graph->diameter, graph->max_in_degree,
                graph->avg_in_degree, graph->max_out_degree, graph->avg_out_degree, num_iterations+1, time_elapsed);
        fflush(out);
    }

    //graph_destroy(graph);
    MPI_Type_free(&joint_probability_struct);
    MPI_Type_free(&belief_struct);
}

void run_test_loopy_belief_propagation_mtx_files_edge_cuda_openmpi(const char *edge_file_name, const char *node_file_name,
        const struct joint_probability * edge_prob, size_t num_src, size_t num_dest,
                                                                    FILE *out, int my_rank, int n_ranks, int num_devices) {
    // each node runs this....
    Graph_t graph = NULL;
    clock_t start;
    clock_t end;
    double time_elapsed;
    int num_iterations;
    size_t num_vertices, num_edges;
    num_vertices = 0;
    num_edges = 0;

    // set up structs
    register_belief();
    register_joint_probability();

    if(my_rank == 0) {
        graph = build_graph_from_mtx(edge_file_name, node_file_name, edge_prob, num_src, num_dest);
        assert(graph != NULL);
        num_vertices = graph->current_num_vertices;
        num_edges = graph->current_num_edges;
        set_up_src_nodes_to_edges_no_hsearch(graph);
        set_up_dest_nodes_to_edges_no_hsearch(graph);
        init_previous_edge(graph);
        start = clock();
    }

    MPICHECK(MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    assert(num_vertices > 0);
    assert(num_edges > 0);
    if(my_rank > 0) {
        graph = create_graph(num_vertices, num_edges, edge_prob, num_src, num_dest);
        graph->current_num_edges = num_edges;
        graph->current_num_vertices = num_vertices;
    }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    // copy rest of graph
    MPICHECK(MPI_Bcast(&(graph->max_degree), 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->edges_src_index), num_edges, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->edges_dest_index), num_edges, MPI_INT, 0, MPI_COMM_WORLD));

    MPICHECK(MPI_Bcast(&(graph->edges_messages), num_edges, belief_struct, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->edges_messages_size), 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->edges_messages_previous), num_edges, MPI_FLOAT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->edges_messages_current), num_edges, MPI_FLOAT, 0, MPI_COMM_WORLD));

    MPICHECK(MPI_Bcast(&(graph->node_states), num_vertices, belief_struct, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->node_states_size), 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->node_states_previous), num_vertices, MPI_FLOAT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->node_states_current), num_vertices, MPI_FLOAT, 0, MPI_COMM_WORLD));

    MPICHECK(MPI_Bcast(&(graph->src_nodes_to_edges_node_list), num_vertices, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->src_nodes_to_edges_edge_list), num_edges, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->dest_nodes_to_edges_edge_list), num_edges, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&(graph->dest_nodes_to_edges_node_list), num_vertices, MPI_INT, 0, MPI_COMM_WORLD));

    // wait until everyone is set up
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    num_iterations = loopy_propagate_until_cuda_edge_openmpi(graph, PRECISION, NUM_ITERATIONS, my_rank, n_ranks, num_devices);

    if(my_rank == 0) {
        end = clock();
        time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;

        fprintf(out, "%s-%s,loopy-edge-openmpi,%ld,%ld,%d,%d,%lf,%d,%lf,%d,%lf\n", edge_file_name, node_file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, graph->max_in_degree, graph->avg_in_degree, graph->max_out_degree, graph->avg_out_degree, num_iterations+1, time_elapsed);
        fflush(out);
    }

    graph_destroy(graph);
    MPI_Type_free(&joint_probability_struct);
    MPI_Type_free(&belief_struct);
}


/**
 * Checks that the CUDA kernel completed
 * @param file The source code file
 * @param line The line within the source code file that executes the kernel
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

