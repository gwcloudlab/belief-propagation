#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "math.h"

#include "graph.h"

static char src_key[CHARS_IN_KEY], dest_key[CHARS_IN_KEY];

float fmaxf(float a, float b) {
    if(a >= b) {
        return a;
    }
    return b;
}
/**
 * Allocate and init the graph
 * @param num_vertices The number of vertices to allocate
 * @param num_edges The number of edges to allocate
 * @return An initialized graph
 */
Graph_t
create_graph(int num_vertices, int num_edges, const struct joint_probability *joint_probability, int joint_probability_dim_x, int joint_probability_dim_y)
{
	Graph_t g;

	g = (Graph_t)malloc(sizeof(struct graph));
	assert(g);
	g->edges_src_index = (int *)malloc(sizeof(int) * num_edges);
	assert(g->edges_src_index);
	g->edges_dest_index = (int *)malloc(sizeof(int) * num_edges);
	assert(g->edges_dest_index);

	assert(joint_probability_dim_x > 0);
	assert(joint_probability_dim_x <= MAX_STATES);

	assert(joint_probability_dim_y > 0);
	assert(joint_probability_dim_y <= MAX_STATES);

	g->edge_joint_probability_dim_x = joint_probability_dim_x;
	g->edge_joint_probability_dim_y = joint_probability_dim_y;

	for(int x = 0; x < joint_probability_dim_x; ++x) {
	    for(int y = 0; y < joint_probability_dim_y; ++y) {
	        g->edge_joint_probability.data[x][y] = joint_probability->data[x][y];
	    }
	}

    g->edges_messages = (struct belief *)malloc(sizeof(struct belief) * num_edges);
    assert(g->edges_messages);
    g->edges_messages_current = (float *)malloc(sizeof(float) * num_edges);
    assert(g->edges_messages_current);
    g->edges_messages_previous = (float *)malloc(sizeof(float) * num_edges);
    assert(g->edges_messages_previous);
    g->edges_messages_size = (int *)malloc(sizeof(int) * num_edges);
    assert(g->edges_messages_size);

    g->node_states = (struct belief *)malloc(sizeof(struct belief) * num_vertices);
    assert(g->node_states);
    g->node_states_previous = (float *)malloc(sizeof(float) * num_vertices);
    assert(g->node_states_previous);
    g->node_states_current = (float *)malloc(sizeof(float) * num_vertices);
	assert(g->node_states_current);
	g->node_states_size = (int *)malloc(sizeof(int) * num_vertices);
	assert(g->node_states_current);

	g->src_nodes_to_edges_node_list = (int *)malloc(sizeof(int) * num_vertices);
	assert(g->src_nodes_to_edges_node_list);
	g->src_nodes_to_edges_edge_list = (int *)malloc(sizeof(int) * num_edges);
	assert(g->src_nodes_to_edges_edge_list);
	g->dest_nodes_to_edges_node_list = (int *)malloc(sizeof(int) * num_vertices);
	assert(g->dest_nodes_to_edges_node_list);
	g->dest_nodes_to_edges_edge_list = (int *)malloc(sizeof(int) * num_edges);
	assert(g->dest_nodes_to_edges_edge_list);
	g->node_names = (char *)malloc(sizeof(char) * CHAR_BUFFER_SIZE * num_vertices);
	assert(g->node_names);
	g->visited = (char *)calloc(sizeof(char), (size_t)num_vertices);
	assert(g->visited);
	g->observed_nodes = (char *)calloc(sizeof(char), (size_t)num_vertices);
	assert(g->observed_nodes);
	g->variable_names = (char *)calloc(sizeof(char), (size_t)num_vertices * CHAR_BUFFER_SIZE * MAX_STATES);
	assert(g->variable_names);
    g->levels_to_nodes = (int *)malloc(sizeof(int) * 2 * num_vertices);
    assert(g->levels_to_nodes != NULL);
	g->work_queue_edges = NULL;
	g->work_queue_nodes = NULL;
	g->work_queue_scratch = NULL;

    g->node_hash_table_created = 0;
    g->edge_tables_created = 0;

    g->num_levels = 0;
	g->total_num_vertices = num_vertices;
	g->total_num_edges = num_edges;
	g->current_num_vertices = 0;
	g->current_num_edges = 0;
    g->diameter = -1;
    g->max_degree = 0;
	return g;
}

/**
 * Initializes the node at the given index
 * @param graph The graph to add the node to
 * @param node_index The index to add it at
 * @param num_variables The number of beliefs the node has
 */
void initialize_node(Graph_t graph, int node_index, int num_variables){
	int i;
	struct belief *new_belief;

	new_belief = &graph->node_states[node_index];
	assert(new_belief);
	graph->node_states_size[node_index] = num_variables;
	for(i = 0; i < num_variables; ++i){
		new_belief->data[i] = DEFAULT_STATE;
	}
}

/**
 * Initializes the edge on the graph
 * @param graph The graph to add the edge to
 * @param edge_index The index to add the edge at
 * @param src_index The index of the source node
 * @param dest_index The index of the destination node
 * @param dim_x The first dimension of the joint probability matrix
 * @param dim_y The second the dimension of the joint probability matrix
 * @param joint_probabilities The joint probability of the edge
 */
void init_edge(Graph_t graph, int edge_index, int src_index, int dest_index, int dim_x){
	int i, j;
    struct belief *edges_messages;

	assert(src_index < graph->total_num_vertices);
	assert(dest_index < graph->total_num_vertices);
	assert(edge_index < graph->total_num_edges);

	assert(dim_x <= MAX_STATES);

	graph->edges_src_index[edge_index] = src_index;
	graph->edges_dest_index[edge_index] = dest_index;

    edges_messages = &graph->edges_messages[edge_index];
    assert(edges_messages);
    graph->edges_messages_size[edge_index] = dim_x;

    for(i = 0; i < dim_x; ++i){
		graph->edges_messages[edge_index].data[i] = 0;
        graph->edges_messages_previous[edge_index] = INFINITY;
        graph->edges_messages_current[edge_index] = INFINITY;
    }
}

/**
 * Sets the belief of the node
 * @param graph The graph holding the node
 * @param node_index The node's index
 * @param num_variables The number of variables of the node
 * @param state The state of the variables to set
 * @param size The number of variables for the belief
 */
void node_set_state(Graph_t graph, int node_index, int num_variables, struct belief *state){
	int i;

	graph->node_states_size[node_index] = num_variables;
    for(i = 0; i < num_variables; ++i){
        graph->node_states[node_index].data[i] = state->data[i];
    }
}

/**
 * Adds a node to the graph
 * @param g The graph
 * @param num_variables The number of variables (beliefs) of the node
 * @param name The name of the node
 */
void graph_add_node(Graph_t g, int num_variables, const char * name) {
    int node_index;

    node_index = g->current_num_vertices;

    initialize_node(g, node_index, num_variables);
    strncpy(&g->node_names[node_index * CHAR_BUFFER_SIZE], name, CHAR_BUFFER_SIZE);

    g->current_num_vertices += 1;
}

/**
 * Adds a node to the graph and sets its beliefs
 * @param g The graph
 * @param num_variables The number of variables (beliefs) of the node
 * @param name The name of the graph
 * @param belief The states to set the node to
 * @param size The number of states for the belief
 */
void graph_add_and_set_node_state(Graph_t g, int num_variables, const char * name, struct belief *belief){
	int node_index;

	node_index = g->current_num_vertices;

	g->observed_nodes[node_index] = 1;
	graph_add_node(g, num_variables, name);
	node_set_state(g, node_index, num_variables, belief);
}

/**
 * Sets the belief of the node
 * @param g The graph
 * @param node_index The index of the node
 * @param num_states The number of states (beliefs) of the node
 * @param belief The belief to set the node to
 * @param size The number of states for the node
 */
void graph_set_node_state(Graph_t g, int node_index, int num_states, struct belief *belief){

	assert(node_index < g->current_num_vertices);

	assert(num_states <= g->node_states_size[node_index]);

	g->observed_nodes[node_index] = 1;

	node_set_state(g, node_index, num_states, belief);
}

/**
 * Adds the edge to the graph
 * @param graph The graph holding the edge
 * @param src_index The index of the source node of the edge
 * @param dest_index The index of the destination node of the edge
 * @param dim_x The first dimension of the edge's joint probability matrix
 * @param dim_y The second dimension of the edge's joint probability matrix
 * @param joint_probabilities The joint probability matrix
 */
void graph_add_edge(Graph_t graph, int src_index, int dest_index, int dim_x, int dim_y) {
	int edge_index;
    ENTRY src_e, *src_ep;
    ENTRY dest_e, *dest_ep;
    struct htable_entry *src_entry, *dest_entry;

	edge_index = graph->current_num_edges;
    assert(edge_index < graph->total_num_edges);

	assert(graph->node_states_size[src_index] == dim_x);
	assert(graph->node_states_size[dest_index] == dim_y);

    init_edge(graph, edge_index, src_index, dest_index, dim_x);
    if(graph->edge_tables_created == 0){
        graph->src_node_to_edge_table = (struct hsearch_data *)calloc(sizeof(struct hsearch_data), 1);
		assert(graph->src_node_to_edge_table);
        graph->dest_node_to_edge_table = (struct hsearch_data *)calloc(sizeof(struct hsearch_data), 1);
		assert(graph->dest_node_to_edge_table);
        assert(hcreate_r((size_t)graph->current_num_vertices, graph->src_node_to_edge_table) != 0);
        assert(hcreate_r((size_t)graph->current_num_vertices, graph->dest_node_to_edge_table) != 0);

        graph->edge_tables_created = 1;
    }

	sprintf(src_key, "%d", src_index);
    src_e.key = src_key;
	src_e.data = NULL;
    hsearch_r(src_e, FIND, &src_ep, graph->src_node_to_edge_table);
    if(src_ep == NULL){
        src_entry = (struct htable_entry *)calloc(sizeof(struct htable_entry), 1);
        src_entry->count = 0;
    }
    else{
        src_entry = (struct htable_entry *)src_ep->data;
    }
    //printf("Src Index: %d\n", src_index);
    assert(src_entry != NULL);
    assert(src_entry->count < MAX_DEGREE);
    src_entry->indices[src_entry->count] = edge_index;
    src_entry->count += 1;
    src_e.data = src_entry;

	sprintf(dest_key, "%d", dest_index);
    dest_e.key = dest_key;
	dest_e.data = NULL;
    hsearch_r(dest_e, FIND, &dest_ep, graph->dest_node_to_edge_table);
    if(dest_ep == NULL){
        dest_entry = (struct htable_entry *)calloc(sizeof(struct htable_entry), 1);
        dest_entry->count = 0;
    }
    else{
        dest_entry = (struct htable_entry *)dest_ep->data;
    }
    assert(dest_entry != NULL);
    assert(dest_entry->count < MAX_DEGREE);
    dest_entry->indices[dest_entry->count] = edge_index;
    dest_entry->count += 1;
    dest_e.data = dest_entry;


    assert( hsearch_r(src_e, ENTER, &src_ep, graph->src_node_to_edge_table) != 0);
    assert( hsearch_r(dest_e, ENTER, &dest_ep, graph->dest_node_to_edge_table) != 0);

	graph->current_num_edges += 1;
}

/**
 * Fills in the hash table mapping node names to nodes
 * @param graph The graph holding the hash table
 */
void fill_in_node_hash_table(Graph_t graph){
	long i;
	ENTRY e, *ep;

	if(graph->node_hash_table_created == 0){
		// insert node names into hash
		graph->node_hash_table = (struct hsearch_data *)calloc(sizeof(struct hsearch_data), 1);
		assert( hcreate_r((size_t)graph->current_num_vertices, graph->node_hash_table) != 0 );
		for(i = 0; i < graph->current_num_vertices; ++i){
			e.key = &(graph->node_names[i * CHAR_BUFFER_SIZE]);
			e.data = (void *)i;
			assert( hsearch_r(e, ENTER, &ep, graph->node_hash_table) != 0);

		}
		graph->node_hash_table_created = 1;
	}
}

/**
 * Search the graph's hash table for the node given its name
 * @param name The name of the ndoe
 * @param graph The graph
 * @return The index of the node
 */
long find_node_by_name(char * name, Graph_t graph){
	long i;
	ENTRY e, *ep;

	fill_in_node_hash_table(graph);

	e.key = name;
    e.data = NULL;
	assert( hsearch_r(e, FIND, &ep, graph->node_hash_table) != 0);
	assert(ep != NULL);

	i = (long)ep->data;
	assert(i < graph->current_num_vertices);


	return i;
}

static void set_up_nodes_to_edges(const int *edges_index, int * nodes_to_edges_nodes_list,
                                  int * nodes_to_edges_edges_list, Graph_t graph){
    int i, j, edge_index, num_vertices, num_edges, current_degree;
    ENTRY entry, *ep;
    struct htable_entry *metadata;
    char *search_key;
	struct hsearch_data *htab;

    assert(graph->current_num_vertices == graph->total_num_vertices);
    assert(graph->current_num_edges <= graph->total_num_edges);

    edge_index = 0;

    num_vertices = graph->total_num_vertices;
    num_edges = graph->current_num_edges;


	htab = (struct hsearch_data *)calloc(1, sizeof(struct hsearch_data));
    hcreate_r((size_t )num_vertices, htab);
    // fill hash table
    for(j = 0; j < num_edges; ++j){
        // search by node name
        i = edges_index[j];
        search_key = &(graph->node_names[i * CHAR_BUFFER_SIZE]);
        entry.key = search_key;
        entry.data = NULL;
        hsearch_r(entry, FIND, &ep, htab);

        // grab metadata if it exists or create it
        if(ep == NULL) {
            metadata = (struct htable_entry *)calloc(sizeof(struct htable_entry), 1);
            assert(metadata > 0);
			metadata->count = 0;
        }
        else {
            metadata = ep->data;
        }
        // add current edge to list
        metadata->indices[metadata->count] = j;
        metadata->count += 1;
        // ensure we're not going over
        assert(metadata->count < num_edges + 1);
        // insert
        entry.data = metadata;
        hsearch_r(entry, ENTER, &ep, htab);
        assert(ep > 0);
    }
    // fill in array
    for(i = 0; i < num_vertices; ++i){
        nodes_to_edges_nodes_list[i] = edge_index;

        current_degree = 0;

        search_key = &(graph->node_names[i * CHAR_BUFFER_SIZE]);
        entry.key = search_key;
        entry.data = NULL;
        hsearch_r(entry, FIND, &ep, htab);
        if(ep > 0) {
            metadata = ep->data;
            assert(metadata);
            assert(metadata->indices);
            assert(metadata->count >= 0);

            for (j = 0; j < metadata->count; ++j) {
                nodes_to_edges_edges_list[edge_index] = metadata->indices[j];
                edge_index += 1;
                current_degree += 1;
            }

            //cleanup
            free(metadata);

            if (current_degree > graph->max_degree) {
                graph->max_degree = current_degree;
            }
        }
    }
    hdestroy_r(htab);
	free(htab);
}

void set_up_src_nodes_to_edges(Graph_t graph){
	set_up_nodes_to_edges(graph->edges_src_index, graph->src_nodes_to_edges_node_list,
						  graph->src_nodes_to_edges_edge_list, graph);
}

/**
 * Sets up the parallel arrays holding the mapping of destination nodes to their edges
 * @param graph The graph to add the arrays to
 */
void set_up_dest_nodes_to_edges(Graph_t graph){
    set_up_nodes_to_edges(graph->edges_dest_index, graph->dest_nodes_to_edges_node_list,
						  graph->dest_nodes_to_edges_edge_list, graph);
}

void graph_destroy_htables(Graph_t g) {
    int i;
    ENTRY src_e, dest_e, *src_ep, *dest_ep;
    src_ep = NULL;
    dest_ep = NULL;
    if(g->node_hash_table_created != 0){
        hdestroy_r(g->node_hash_table);
        free(g->node_hash_table);
    }
    if(g->edge_tables_created != 0){
        for(i = 0; i < g->current_num_vertices; ++i){
            sprintf(src_key, "%d", i);
            src_e.key = src_key;
            src_e.data = NULL;
            if(hsearch_r(src_e, FIND, &src_ep, g->src_node_to_edge_table) != 0) {
                if (src_ep != NULL) {
                    free(src_ep->data);
                    src_ep->data = NULL;
                }
            }
            src_ep = NULL;

            sprintf(dest_key, "%d", i);
            dest_e.key = dest_key;
            dest_e.data = NULL;
            if(hsearch_r(dest_e, FIND, &dest_ep, g->dest_node_to_edge_table) != 0) {
                if(dest_ep != NULL){
                    free(dest_ep->data);
                }
            }
            dest_ep = NULL;
        }
        hdestroy_r(g->src_node_to_edge_table);
        hdestroy_r(g->dest_node_to_edge_table);
        free(g->src_node_to_edge_table);
        free(g->dest_node_to_edge_table);
    }

    g->node_hash_table_created = 0;
}

/**
 * Frees all allocated memory for the graph and its associated members
 * @param g The graph
 */
void graph_destroy(Graph_t g) {
	graph_destroy_htables(g);

	free(g->edges_src_index);
	free(g->edges_dest_index);

	free(g->edges_messages);
	free(g->edges_messages_current);
	free(g->edges_messages_previous);
	free(g->edges_messages_size);

	free(g->src_nodes_to_edges_node_list);
	free(g->src_nodes_to_edges_edge_list);
	free(g->dest_nodes_to_edges_node_list);
	free(g->dest_nodes_to_edges_edge_list);

	free(g->node_names);
	free(g->visited);
	free(g->observed_nodes);
	free(g->variable_names);
	free(g->levels_to_nodes);

	free(g->node_states);
	free(g->node_states_current);
	free(g->node_states_previous);
	free(g->node_states_size);

	if(g->work_queue_nodes != NULL) {
		free(g->work_queue_nodes);
	}
	if(g->work_queue_edges != NULL) {
		free(g->work_queue_edges);
	}
	if(g->work_queue_scratch != NULL) {
		free(g->work_queue_scratch);
	}

	free(g);
}

/**
 * Propagates beliefs by level for regular BP
 * @param g The graph to propagate beliefs
 */
void propagate_using_levels_start(Graph_t g){
	int i, k, node_index, edge_index, level_start_index, level_end_index, start_index, end_index, num_vertices;

	num_vertices = g->current_num_vertices;

	level_start_index = g->levels_to_nodes[0];
	if(1 == g->num_levels){
		level_end_index = 2 * num_vertices;
	}
	else{
		level_end_index = g->levels_to_nodes[1];
	}
	for(k = level_start_index; k < level_end_index; ++k){
		node_index = g->levels_to_nodes[k];
		//set as visited
		g->visited[node_index] = 1;

		//send messages
		start_index = g->src_nodes_to_edges_node_list[node_index];
		if(node_index + 1 == num_vertices){
			end_index = g->current_num_edges;
		}
		else {
			end_index = g->src_nodes_to_edges_node_list[node_index + 1];
		}
		for(i = start_index; i < end_index; ++i){
			g->visited[node_index] = 1;
			edge_index = g->src_nodes_to_edges_edge_list[i];

			send_message(&g->node_states[node_index], edge_index, &(g->edge_joint_probability),
					g->edge_joint_probability_dim_x, g->edge_joint_probability_dim_y,
					g->edges_messages, g->edges_messages_previous, g->edges_messages_current);

			/*printf("sending belief on edge\n");
			print_edge(g, edge_index);
			printf("belief: [");
			for(j = 0; j < g->node_num_vars[node_index]; ++j){
				printf("%.6lf\t", g->node_states[MAX_STATES * node_index + j]);
			}
			printf("]\n");



			printf("edge belief is:\n[");
			for(j = 0; j < g->edges_x_dim[edge_index]; ++j){
				printf("%.6lf\t", g->edges_messages[MAX_STATES * edge_index + j]);
			}
			printf("]\n");*/
		}
	}
}

/**
 * Updates the edge's buffer belief using the given state
 * @param states The source belief to send
 * @param edge_index The index of the edge
 * @param edge_joint_probabilities The array of joint probabilities in the graph
 * @param edge_messages The array of edge buffer beliefs
 */
void send_message(const struct belief * __restrict__ states, int edge_index,
		const struct joint_probability * __restrict__ edge_joint_probability,
				const int num_src,
				const int num_dest,
				struct belief *edge_messages,
				float * __restrict__ edges_messages_previous,
				float * __restrict__ edges_messages_current){
	int i, j;
	float sum;

	sum = 0.0;
	for(i = 0; i < num_src; ++i){
		edge_messages[edge_index].data[i] = 0.0;
		for(j = 0; j < num_dest; ++j){
            edge_messages[edge_index].data[i] += edge_joint_probability->data[i][j] * states->data[j];
		}
		sum += edge_messages[edge_index].data[i];
	}
	if(sum <= 0.0){
		sum = 1.0;
	}
    edges_messages_previous[edge_index] = edges_messages_current[edge_index];
    edges_messages_current[edge_index] = sum;
	for (i = 0; i < num_src; ++i) {
		edge_messages[edge_index].data[i] /= sum;
	}
}

/**
 * Combines beliefs
 * @param dest The destination belief to update
 * @param src The source belief
 * @param num_variables The number of variables (beliefs) within the message
 * @param offset The index offset for the source lookup
 */
#pragma acc routine
static inline void combine_message(struct belief * __restrict__ dest, const struct belief * __restrict__ src, int num_variables, int offset){
	int i;

#pragma omp simd safelen(AVG_STATES)
#pragma simd vectorlength(AVG_STATES)
	for(i = 0; i < num_variables; ++i){
		if(src[offset].data[i] == src[offset].data[i]) { // ensure no nan's
			dest->data[i] *= src[offset].data[i];
		}
	}
}

/**
 * Combines page rank contributions
 * @param dest The destination contributions
 * @param src The source contributions
 * @param num_variables The number of variables (contributions) within the message
 * @param offset The index offset for the source lookup
 */
#pragma acc routine
static inline void combine_page_rank_message(struct belief * __restrict__ dest, const struct belief * __restrict__ src, int num_variables, int offset){
    int i;

#pragma omp simd safelen(AVG_STATES)
#pragma simd vectorlength(AVG_STATES)
    for(i = 0; i < num_variables; ++i){
        if(src[offset].data[i] == src[offset].data[i]) { // ensure no nan's
            dest->data[i] += src[offset].data[i];
        }
    }
}

/**
 * Combines Viterbi beliefs
 * @param dest The destination belief to update
 * @param src The source belief
 * @param num_variables The number of variables (beliefs) within the message
 * @param offset The index offset for the source lookup
 */
#pragma acc routine
static inline void combine_viterbi_message(struct belief * __restrict__ dest, const struct belief * __restrict__ src, int num_variables, int offset){
    int i;

#pragma omp simd safelen(AVG_STATES)
#pragma simd vectorlength(AVG_STATES)
    for(i = 0; i < num_variables; ++i){
        if(src[offset].data[i] == src[offset].data[i]) { // ensure no nan's
            dest->data[i] = fmaxf(dest->data[i], src->data[i]);
        }
    }
}

/**
 * Propagates a node's belief using levels in normal BP
 * @param g The graph
 * @param current_node_index The index of the node within the graph
 */
static void propagate_node_using_levels(Graph_t g, int current_node_index){
	int i, num_variables, start_index, end_index, num_vertices, edge_index;
	int * dest_nodes_to_edges_nodes;
	int * dest_nodes_to_edges_edges;
	int * src_nodes_to_edges_nodes;
	int * src_nodes_to_edges_edges;
    struct belief buffer;

	num_variables = g->node_states_size[current_node_index];

	// mark as visited
	g->visited[current_node_index] = 1;

	num_vertices = g->current_num_vertices;
	dest_nodes_to_edges_nodes = g->dest_nodes_to_edges_node_list;
	dest_nodes_to_edges_edges = g->dest_nodes_to_edges_edge_list;
	src_nodes_to_edges_nodes = g->src_nodes_to_edges_node_list;
	src_nodes_to_edges_edges = g->src_nodes_to_edges_edge_list;

	// init buffer
	for(i = 0; i < num_variables; ++i){
		buffer.data[i] = 1.0;
	}

	// get the edges feeding into this node
	start_index = dest_nodes_to_edges_nodes[current_node_index];
	if(current_node_index + 1 == num_vertices){
		end_index = g->current_num_edges;
	}
	else{
		end_index = dest_nodes_to_edges_nodes[current_node_index + 1];
	}
	for(i = start_index; i < end_index; ++i){
		edge_index = dest_nodes_to_edges_edges[i];

		combine_message(&buffer, g->edges_messages, num_variables, edge_index);
	}

	//send belief
	start_index = src_nodes_to_edges_nodes[current_node_index];
	if(current_node_index + 1 == num_vertices){
		end_index = g->current_num_edges;
	}
	else {
		end_index = src_nodes_to_edges_nodes[current_node_index + 1];
	}

	for(i = start_index; i < end_index; ++i){
		edge_index = src_nodes_to_edges_edges[i];
		//ensure node hasn't been visited yet
		if(g->visited[g->edges_dest_index[edge_index]] == 0){
			/*printf("sending belief on edge\n");
			print_edge(g, edge_index);
			printf("belief: [");
			for(j = 0; j < num_variables; ++j){
				printf("%.6lf\t", message_buffer[j]);
			}
			printf("]\n");*/
			send_message(&buffer, edge_index, &(g->edge_joint_probability), g->edge_joint_probability_dim_x, g->edge_joint_probability_dim_y,
					g->edges_messages, g->edges_messages_previous, g->edges_messages_current);
		}
	}
}

/**
 * Propagates beliefs of the nodes at the current level
 * @param g The graph
 * @param current_level The current level within the tree
 */
void propagate_using_levels(Graph_t g, int current_level) {
	int i, start_index, end_index;

	start_index = g->levels_to_nodes[current_level];
	if(current_level + 1 == g->num_levels){
		end_index = 2 * g->current_num_vertices;
	}
	else{
		end_index = g->levels_to_nodes[current_level + 1];
	}
	//#pragma omp parallel for shared(g, start_index, end_index) private(i)
	for(i = start_index; i < end_index; ++i){
		propagate_node_using_levels(g, g->levels_to_nodes[i]);
	}
}

/**
 * Marginalizes the node at the given index
 * @param g The graph holding all nodes
 * @param node_index The index of the node within the graph
 */
static void marginalize_node(Graph_t g, int node_index){
	int i, num_variables, start_index, end_index, edge_index;
	float sum;

	int * dest_nodes_to_edges_nodes;
	int * dest_nodes_to_edges_edges;

    struct belief new_belief;

	dest_nodes_to_edges_nodes = g->dest_nodes_to_edges_node_list;
	dest_nodes_to_edges_edges = g->dest_nodes_to_edges_edge_list;

	num_variables = g->node_states_size[node_index];

	for(i = 0; i < num_variables; ++i){
		new_belief.data[i] = 1.0;
	}

	start_index = dest_nodes_to_edges_nodes[node_index];
	if(node_index + 1 == g->current_num_vertices){
		end_index = g->current_num_edges;
	}
	else {
		end_index = dest_nodes_to_edges_nodes[node_index + 1];
	}

	for(i = start_index; i < end_index; ++i){
		edge_index = dest_nodes_to_edges_edges[i];

		combine_message(&new_belief, g->edges_messages, num_variables, edge_index);
	}
	if(start_index < end_index){
		for(i = 0; i < num_variables; ++i){
			g->node_states[node_index].data[i] = new_belief.data[i];
		}
	}
	sum = 0.0;
	for(i = 0; i < num_variables; ++i){
		sum += g->node_states[node_index].data[i];
	}
	if(sum <= 0.0){
		sum = 1.0;
	}

	for(i = 0; i < num_variables; ++i){
		g->node_states[node_index].data[i] = g->node_states[node_index].data[i] / sum;
	}
}

/**
 * Marginalizes all nodes within the graph
 * @param g The graph
 */
void marginalize(Graph_t g){
	int i, num_nodes;

	num_nodes = g->current_num_vertices;

	for(i = 0; i < num_nodes; ++i){
		marginalize_node(g, i);
	}
}

/**
 * Resets the visited flags in the graph
 * @param g The graph
 */
void reset_visited(Graph_t g){
	int i, num_nodes;

	num_nodes = g->current_num_vertices;
	for(i = 0; i < num_nodes; ++i){
		g->visited[i] = 0;
	}
}

/**
 * Prints a given node
 * @param graph The graph holding all nodes
 * @param node_index The index of the node
 */
void print_node(Graph_t graph, int node_index){
	int i, num_vars, variable_name_index;

	num_vars = graph->node_states_size[node_index];

	printf("Node %s [\n", &graph->node_names[node_index * CHAR_BUFFER_SIZE]);
	for(i = 0; i < num_vars; ++i){
		variable_name_index = node_index * CHAR_BUFFER_SIZE * MAX_STATES + i * CHAR_BUFFER_SIZE;
		printf("%s:\t%.6lf\n", &graph->variable_names[variable_name_index], graph->node_states[node_index].data[i]);
	}
	printf("]\n");
}

/**
 * Prints the edge in the graph
 * @param graph The graph
 * @param edge_index The index of the edge
 */
void print_edge(Graph_t graph, int edge_index){
	int i, j, dim_x, dim_y, src_index, dest_index;


	dim_x = graph->edge_joint_probability_dim_x;
	dim_y = graph->edge_joint_probability_dim_y;
	src_index = graph->edges_src_index[edge_index];
	dest_index = graph->edges_dest_index[edge_index];

	printf("Edge  %s -> %s [\n", &graph->node_names[src_index * CHAR_BUFFER_SIZE], &graph->node_names[dest_index * CHAR_BUFFER_SIZE]);
	printf("Joint probability matrix: [\n");
	for(i = 0; i < dim_x; ++i){
		printf("[");
		for(j = 0; j < dim_y; ++j){
			printf("\t%.6lf",  graph->edge_joint_probability.data[i][j]);
		}
		printf("\t]\n");
	}
	printf("]\nMessage:\n[");
	for(i = 0; i < dim_x; ++i){
		printf("\t%.6lf", graph->edges_messages[edge_index].data[i]);
	}
	printf("\t]\n]\n");
}

/**
 * Prints all nodes in the graph
 * @param g The graph
 */
void print_nodes(Graph_t g){
	int i, num_nodes;

	num_nodes = g->current_num_vertices;

	for(i = 0; i < num_nodes; ++i){
		print_node(g, i);
	}
}

/**
 * Prints all edges in the graph
 * @param g The graph
 */
void print_edges(Graph_t g){
	int i, num_edges;

	num_edges = g->current_num_edges;

	for(i = 0; i < num_edges; ++i){
		print_edge(g, i);
	}
}

/**
 * Pritns the source nodes to edges mapping
 * @param g The graph
 */
void print_src_nodes_to_edges(Graph_t g){
	int i, j, start_index, end_index, num_vertices, edge_index;
	int * src_node_to_edges_nodes;
	int * src_node_to_edges_edges;

	printf("src index -> edge index\n");


	src_node_to_edges_nodes = g->src_nodes_to_edges_node_list;
	src_node_to_edges_edges = g->src_nodes_to_edges_edge_list;
	num_vertices = g->total_num_vertices;

	for(i = 0; i < num_vertices; ++i){
		printf("Node -----\n");
		print_node(g, i);
		printf("Edges-------\n");
		start_index = src_node_to_edges_nodes[i];
		if(i + 1 == num_vertices){
			end_index = g->current_num_edges;
		}
		else{
			end_index = src_node_to_edges_nodes[i+1];
		}
		for(j = start_index; j < end_index; ++j){
			edge_index = src_node_to_edges_edges[j];
			print_edge(g, edge_index);
		}
		printf("---------\n");
	}
}

/**
 * Prints the destination nodes to edges mapping
 * @param g The graph
 */
void print_dest_nodes_to_edges(Graph_t g){
	int i, j, start_index, end_index, num_vertices, edge_index;
	int * dest_node_to_edges_nodes;
	int * dest_node_to_edges_edges;

	printf("dest index -> edge index\n");

	dest_node_to_edges_nodes = g->dest_nodes_to_edges_node_list;
	dest_node_to_edges_edges = g->dest_nodes_to_edges_edge_list;
	num_vertices = g->total_num_vertices;

	for(i = 0; i < num_vertices; ++i){
		printf("Node -----\n");
		print_node(g, i);
		printf("Edges-------\n");
		start_index = dest_node_to_edges_nodes[i];
		if(i + 1 == num_vertices){
			end_index = g->current_num_edges;
		}
		else{
			end_index = dest_node_to_edges_nodes[i+1];
		}
		for(j = start_index; j < end_index; ++j){
			edge_index = dest_node_to_edges_edges[j];
			print_edge(g, edge_index);
		}
		printf("---------\n");
	}
}

/**
 * Sets up the beliefs of the array of previous beliefs
 * @param graph The graph holding the data
 */
void init_previous_edge(Graph_t graph){
	int i, j, num_vertices, start_index, end_index, edge_index;
	int * src_node_to_edges_nodes;
	int * src_node_to_edges_edges;
	struct belief *previous_messages;
	float * current;
	float * previous;

	num_vertices = graph->current_num_vertices;
	src_node_to_edges_nodes = graph->src_nodes_to_edges_node_list;
	src_node_to_edges_edges = graph->src_nodes_to_edges_edge_list;
	previous_messages = graph->edges_messages;
	current = graph->edges_messages_current;
	previous = graph->edges_messages_previous;

	for(i = 0; i < num_vertices; ++i){
		start_index = src_node_to_edges_nodes[i];
		if(i + 1 >= num_vertices){
			end_index = graph->current_num_edges;
		}
		else
		{
			end_index = src_node_to_edges_nodes[i + 1];
		}
		for(j = start_index; j < end_index; ++j){
			edge_index = src_node_to_edges_edges[j];

			send_message(&graph->node_states[i], edge_index, &(graph->edge_joint_probability), graph->edge_joint_probability_dim_x,
					graph->edge_joint_probability_dim_y, previous_messages, previous, current);
		}
	}
}

/**
 * Initializes the frontier for loopy BP
 * @param graph The graph
 * @param start_index The start index of the frontier
 * @param end_index The end index of the frontier
 * @param max_count The max size of the frontier
 */
void fill_in_leaf_nodes_in_index(Graph_t graph, const int * start_index, int * end_index, int max_count){
	int i, diff, edge_start_index, edge_end_index;

    graph->levels_to_nodes[0] = *start_index;
    for(i = 0; i < graph->current_num_vertices; ++i){
        edge_start_index = graph->dest_nodes_to_edges_node_list[i];
        if(i + 1 == graph->current_num_vertices){
            edge_end_index = graph->current_num_edges;
        }
        else{
            edge_end_index = graph->dest_nodes_to_edges_node_list[i + 1];
        }
		diff = edge_end_index - edge_start_index;

        if(diff <= max_count){
            graph->levels_to_nodes[*end_index] = i;
            *end_index += 1;
        }
    }
}

/**
 * Visits a node for regular BP; same as visiting a node using BFS
 * @param graph The graph
 * @param buffer_index The current index in the frontier
 * @param end_index The end of the frontier
 */
void visit_node(Graph_t graph, int buffer_index, int * end_index){
	int node_index, edge_start_index, edge_end_index, edge_index, i, j, dest_node_index;
	char visited;

    node_index = graph->levels_to_nodes[buffer_index];
    if(graph->visited[node_index] == 0){
        graph->visited[node_index] = 1;
        edge_start_index = graph->src_nodes_to_edges_node_list[node_index];
        if(node_index + 1 == graph->current_num_vertices){
            edge_end_index = graph->current_num_edges;
        }
        else{
            edge_end_index = graph->src_nodes_to_edges_node_list[node_index + 1];
        }
        for(i = edge_start_index; i < edge_end_index; ++i){
			edge_index = graph->src_nodes_to_edges_edge_list[i];
            dest_node_index = graph->edges_dest_index[edge_index];
            visited = 0;
			for(j = graph->current_num_vertices; j < *end_index; ++j){
				if(graph->levels_to_nodes[j] == dest_node_index){
					visited = 1;
					break;
				}
			}
			if(visited == 0){
				graph->levels_to_nodes[*end_index] = dest_node_index;
				*end_index += 1;
			}
        }
    }
}

/**
 * Organize nodes by their levels within the tree
 * @param graph The graph
 */
void init_levels_to_nodes(Graph_t graph){
	int start_index, end_index, copy_end_index, i;

    reset_visited(graph);

    start_index = graph->current_num_vertices;
    end_index = start_index;

    fill_in_leaf_nodes_in_index(graph, &start_index, &end_index, 2);
    while(end_index < 2 * graph->current_num_vertices){
        copy_end_index = end_index;
        for(i = start_index; i < copy_end_index; ++i){
            visit_node(graph, i, &end_index);
        }
        graph->num_levels += 1;
        graph->levels_to_nodes[graph->num_levels] = copy_end_index;
    }

	graph->num_levels += 1;

    reset_visited(graph);
}

/**
 * Prints the levels and the nodes at each level
 * @param graph The graph holding the nodes and level information
 */
void print_levels_to_nodes(Graph_t graph){
	int i, j, start_index, end_index;

    for(i = 0; i < graph->num_levels; ++i){
        printf("Level: %d\n", i);
        printf("---------------\n");
        start_index = graph->levels_to_nodes[i];
        if(i + 1 == graph->num_levels){
            end_index = graph->current_num_vertices * 2;
        }
        else{
            end_index = graph->levels_to_nodes[i + 1];
        }
        printf("Nodes:-----------\n");
        for(j = start_index; j < end_index; ++j){
            print_node(graph, graph->levels_to_nodes[j]);
        }
        printf("-------------------\n");
    }
}

/**
 * Initializes the message to the current node beliefs
 * @param message_buffer The beliefs to reset
 * @param node_states The current node states
 */
#pragma acc routine
static void initialize_message_buffer(struct belief * __restrict__ message_buffer, const struct belief * __restrict__ node_states, int node_index, int num_variables){
	int j;

	//reset buffer
	for(j = 0; j < num_variables; ++j){
		message_buffer->data[j] = node_states[node_index].data[j];
	}
}

/**
 * Reads the incoming messages and combines them with the buffer
 * @param message_buffer The destination beliefs
 * @param dest_node_to_edges_nodes Parallel array; maps destination nodes to their edges; first half; nodes mapped to their starting indices in dest_node_to_edges_edges
 * @param dest_node_to_edges_edges Parallel array; maps destination nodes to their edges; second half; edges indexed by the nodes
 * @param previous_messages The previous messages in the graph
 * @param num_edges The number of edges in the graph
 * @param num_vertices The number of nodes in the graph
 * @param num_variables The number of variables for noode
 * @param i The index of the current node in the graph
 */
#pragma acc routine
static void read_incoming_messages(struct belief * __restrict__ message_buffer,
								   const int * __restrict__ dest_node_to_edges_nodes,
								   const int * __restrict__ dest_node_to_edges_edges,
								   const struct belief * __restrict__ previous_messages,
                                   int num_edges, int num_vertices,
								   int num_variables, int i){
	int start_index, end_index, j, edge_index;

	start_index = dest_node_to_edges_nodes[i];
	if(i + 1 >= num_vertices){
		end_index = num_edges;
	}
	else{
		end_index = dest_node_to_edges_nodes[i + 1];
	}

	for(j = start_index; j < end_index; ++j){
		edge_index = dest_node_to_edges_edges[j];

		combine_message(message_buffer, previous_messages, num_variables, edge_index);
	}
}

/**
 * Sends a message from the buffer to the edge's message buffer
 * @param buffer The incoming message
 * @param edge_index The index of the edge in the graph
 * @param joint_probabilities The joint probability tables in the graph
 * @param edge_messages The array of message buffers for the edges
 */
#pragma acc routine
static void send_message_for_edge(const struct belief * __restrict__ buffer, int edge_index,
								  const struct joint_probability * __restrict__ edge_joint_probability,
								  const int num_src,
								  const int num_dest,
								  struct belief * __restrict__ edge_messages,
								  		float *edge_messages_previous,
								  		float *edge_messages_current) {
	int i, j;
	float sum, partial_sum;


	sum = 0.0;
	for(i = 0; i < num_src; ++i){
		partial_sum = 0.0;
        #pragma omp simd safelen(AVG_STATES)
        #pragma simd vectorlength(AVG_STATES)
		for(j = 0; j < num_dest; ++j){
			partial_sum += edge_joint_probability->data[i][j] * buffer->data[j];
		}
		edge_messages[edge_index].data[i] = partial_sum;
		sum += partial_sum;
	}
	if(sum <= 0.0){
		sum = 1.0;
	}
    edge_messages_previous[edge_index] = edge_messages_current[edge_index];
    edge_messages_current[edge_index] = sum;
    #pragma omp simd safelen(AVG_STATES)
    #pragma simd vectorlength(AVG_STATES)
	for (i = 0; i < num_src; ++i) {
        edge_messages[edge_index].data[i] = edge_messages[edge_index].data[i] / sum;
	}
}

/**
 * Sends the belief across the edge's joint probability table
 * @param belief The incoming belief
 * @param src_index The index of the incoming node
 * @param edge_index The index of the edge
 * @param edge_messages The array of edge message buffers
 *
 */
#pragma acc routine
static void send_message_for_edge_iteration(const struct belief * __restrict__ belief, int src_index, int edge_index,
                                            const struct joint_probability * __restrict__ edge_joint_probability,
											const int num_src,
											const int num_dest,
											struct belief * __restrict__ edge_messages,
											float * __restrict__ edge_messages_previous,
											float * __restrict__ edge_messages_current){
    int i, j;
    float sum, partial_sum;

    sum = 0.0;
    for(i = 0; i < num_src; ++i){
        partial_sum = 0.0;
        #pragma omp simd safelen(AVG_STATES)
        #pragma simd vectorlength(AVG_STATES)
        for(j = 0; j < num_dest; ++j){
            partial_sum += edge_joint_probability->data[i][j] * belief[src_index].data[j];
        }
        edge_messages[edge_index].data[i] = partial_sum;
        sum += partial_sum;
    }
    if(sum <= 0.0){
        sum = 1.0;
    }

    edge_messages_previous[edge_index] = edge_messages_current[edge_index];
    edge_messages_current[edge_index] = sum;

    #pragma omp simd safelen(AVG_STATES)
    #pragma simd vectorlength(AVG_STATES)
    for (i = 0; i < num_src; ++i) {
        edge_messages[edge_index].data[i] /= sum;
    }
}

/**
 * Sends a message for the given node
 * @param src_node_to_edges_nodes Parallel array; maps source nodes to their edges; first half; maps nodes to their starting index in src_node_to_edges_edges
 * @param src_node_to_edges_edges Paralell array; maps source nodes to their edges; second half; lists edges by source node
 * @param message_buffer The edge message buffer to update the node beliefs with
 * @param num_edges The number of edges in the graph
 * @param joint_probabilities The array of edge joint probability matrices
 * @param edge_message The array of current edge-buffered beliefs
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param i The current index of the node
 */
#pragma acc routine
static void send_message_for_node(const int * __restrict__ src_node_to_edges_nodes,
								  const int * __restrict__ src_node_to_edges_edges,
								  const struct belief * __restrict__ message_buffer, int num_edges,
								  const struct joint_probability * __restrict__ edge_joint_probability,
								  const int edge_joint_probability_dim_x,
								  const int edge_joint_probability_dim_y,
								  struct belief *edge_messages,
								  float * __restrict__ edge_messages_previous,
								  float * __restrict__ edge_messages_current,
								  int num_vertices, int i){
	int start_index, end_index, j, edge_index;

	start_index = src_node_to_edges_nodes[i];
	if(i + 1 >= num_vertices){
		end_index = num_edges;
	}
	else {
		end_index = src_node_to_edges_nodes[i + 1];
	}
    
	for(j = start_index; j < end_index; ++j){
		edge_index = src_node_to_edges_edges[j];
		/*printf("Sending on edge\n");
        print_edge(graph, edge_index);*/
		send_message_for_edge(message_buffer, edge_index, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y,
				edge_messages, edge_messages_previous, edge_messages_current);
	}
}

/**
 * Marginalizes the nodes in loopy BP
 * @param graph The graph
 * @param current_messages The current edge-buffered messages
 * @param num_vertices The number of nodes in the graph
 */
static void marginalize_loopy_nodes(Graph_t graph, const struct belief * __restrict__ current_messages, int num_vertices) {
	int j;

	int i, num_variables, start_index, end_index, edge_index, current_num_vertices, current_num_edges;
	float sum;
	struct belief *states;
	int *states_size;
	struct belief new_belief;

	int * dest_nodes_to_edges_nodes;
	int * dest_nodes_to_edges_edges;

	dest_nodes_to_edges_nodes = graph->dest_nodes_to_edges_node_list;
	dest_nodes_to_edges_edges = graph->dest_nodes_to_edges_edge_list;
	current_num_vertices = graph->current_num_vertices;
	current_num_edges = graph->current_num_edges;
	states = graph->node_states;
	states_size = graph->node_states_size;

#pragma omp parallel for default(none) shared(states, states_size, num_vertices, current_num_vertices, current_num_edges, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, current_messages) private(i, j, num_variables, start_index, end_index, edge_index, sum, new_belief)
	for(j = 0; j < num_vertices; ++j) {

		num_variables = states_size[j];

		for (i = 0; i < num_variables; ++i) {
			new_belief.data[i] = 1.0;
		}


		start_index = dest_nodes_to_edges_nodes[j];
		if (j + 1 == current_num_vertices) {
			end_index = current_num_edges;
		} else {
			end_index = dest_nodes_to_edges_nodes[j + 1];
		}

		for (i = start_index; i < end_index; ++i) {
			edge_index = dest_nodes_to_edges_edges[i];

			combine_message(&new_belief, current_messages, num_variables, edge_index);

		}
		if (start_index < end_index) {
            #pragma omp simd safelen(AVG_STATES)
            #pragma simd vectorlength(AVG_STATES)
			for (i = 0; i < num_variables; ++i) {
				states[j].data[i] *= new_belief.data[i];
			}
		}
		sum = 0.0;
        #pragma omp simd safelen(AVG_STATES)
        #pragma simd vectorlength(AVG_STATES)
        for (i = 0; i < num_variables; ++i) {
			sum += states[j].data[i];
		}
		if (sum <= 0.0) {
			sum = 1.0;
		}

        #pragma omp simd safelen(AVG_STATES)
        #pragma simd vectorlength(AVG_STATES)
		for (i = 0; i < num_variables; ++i) {
            states[j].data[i] /= sum;
		}
	}

/*
#pragma omp parallel for default(none) shared(graph, num_vertices, current) private(i)
	for(i = 0; i < num_vertices; ++i){
		marginalize_node(graph, i, current);
	}*/

}

/**
 * Marginalizes the nodes in loopy BP
 * @param graph The graph
 * @param current_messages The current edge-buffered messages
 * @param num_vertices The number of nodes in the graph
 */
static void marginalize_page_rank_nodes(Graph_t graph, const struct belief * __restrict__ current_messages, int num_vertices) {
    int j;

    int i, num_variables, start_index, end_index, edge_index, current_num_vertices, current_num_edges;
    float factor;
    struct belief *states;
    int *states_size;
    struct belief new_belief;

    int * dest_nodes_to_edges_nodes;
    int * dest_nodes_to_edges_edges;

    dest_nodes_to_edges_nodes = graph->dest_nodes_to_edges_node_list;
    dest_nodes_to_edges_edges = graph->dest_nodes_to_edges_edge_list;
    current_num_vertices = graph->current_num_vertices;
    current_num_edges = graph->current_num_edges;
    states = graph->node_states;
    states_size = graph->node_states_size;

#pragma omp parallel for default(none) shared(states, states_size, num_vertices, current_num_vertices, current_num_edges, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, current_messages) private(i, j, num_variables, start_index, end_index, edge_index, new_belief, factor)
    for(j = 0; j < num_vertices; ++j) {

        num_variables = states_size[j];

        for (i = 0; i < num_variables; ++i) {
            new_belief.data[i] = 0.0;
        }


        start_index = dest_nodes_to_edges_nodes[j];
        if (j + 1 == current_num_vertices) {
            end_index = current_num_edges;
        } else {
            end_index = dest_nodes_to_edges_nodes[j + 1];
        }

        for (i = start_index; i < end_index; ++i) {
            edge_index = dest_nodes_to_edges_edges[i];

            combine_page_rank_message(&new_belief, current_messages, num_variables, edge_index);
        }
        if (start_index < end_index) {
            factor = (1 - DAMPENING_FACTOR) / (end_index - start_index);
            for (i = 0; i < num_variables; ++i) {
                states[j].data[i] = factor + DAMPENING_FACTOR * new_belief.data[i];
            }
        }
    }

}

/**
 * Get the argmax for the nodes in loopy BP
 * @param graph The graph
 * @param current_messages The current edge-buffered messages
 * @param num_vertices The number of nodes in the graph
 */
static void argmax_loopy_nodes(Graph_t graph, const struct belief * __restrict__ current_messages, int num_vertices) {
    int j;

    int i, num_variables, start_index, end_index, edge_index, current_num_vertices, current_num_edges;
    struct belief *states;
    int * states_size;
    struct belief new_belief;

    int * dest_nodes_to_edges_nodes;
    int * dest_nodes_to_edges_edges;

    dest_nodes_to_edges_nodes = graph->dest_nodes_to_edges_node_list;
    dest_nodes_to_edges_edges = graph->dest_nodes_to_edges_edge_list;
    current_num_vertices = graph->current_num_vertices;
    current_num_edges = graph->current_num_edges;
    states = graph->node_states;
    states_size = graph->node_states_size;

#pragma omp parallel for default(none) shared(states, states_size, num_vertices, current_num_vertices, current_num_edges, dest_nodes_to_edges_nodes, dest_nodes_to_edges_edges, current_messages) private(i, j, num_variables, start_index, end_index, edge_index, new_belief)
    for(j = 0; j < num_vertices; ++j) {

        num_variables = states_size[j];

        for (i = 0; i < num_variables; ++i) {
            new_belief.data[i] = -1.0f;
        }


        start_index = dest_nodes_to_edges_nodes[j];
        if (j + 1 == current_num_vertices) {
            end_index = current_num_edges;
        } else {
            end_index = dest_nodes_to_edges_nodes[j + 1];
        }

        for (i = start_index; i < end_index; ++i) {
            edge_index = dest_nodes_to_edges_edges[i];

            combine_viterbi_message(&new_belief, current_messages, num_variables, edge_index);

        }
        if (start_index < end_index) {
#pragma omp simd safelen(AVG_STATES)
#pragma simd vectorlength(AVG_STATES)
            for (i = 0; i < num_variables; ++i) {
                states[j].data[i] = fmaxf(states[j].data[i], new_belief.data[i]);
            }
        }
    }

/*
#pragma omp parallel for default(none) shared(graph, num_vertices, current) private(i)
	for(i = 0; i < num_vertices; ++i){
		marginalize_node(graph, i, current);
	}*/

}

/**
 * Combines beliefs for edge-based loopy BP
 * @param edge_index The index of the current edge
 * @param current_messages The array of current messages buffered on the edge
 * @param dest_node_index The index of the destination node of the edge
 * @param belief The array of nodes in the graph
 */
#pragma acc routine
static void combine_loopy_edge(int edge_index, const struct belief * __restrict__ current_messages,
							   int dest_node_index, struct belief *belief, const int *belief_size){
    int i, num_variables;
	num_variables = belief_size[dest_node_index];

    for(i = 0; i < num_variables; ++i){
		#pragma omp atomic
		#pragma acc atomic
        belief[dest_node_index].data[i] *= current_messages[edge_index].data[i];
    }
}

/**
 * Marginalizes the nodes for OpenACC
 * @param node_states The current nodes and their beliefs in the graph
 * @param node_index The index of the current node
 * @param edge_messages The current messages buffered at the edges
 * @param dest_nodes_to_edges_nodes Parallel array; maps destination nodes to their edges; first half; the nodes to their starting index in dest_nodes_to_edges_edges
 * @param dest_nodes_to_edges_edges Parallel array; maps destination nodes to their edges; second half; lists edges by node
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 */
#pragma acc routine
static void marginalize_node_acc(struct belief * __restrict__ node_states, const int * __restrict__ node_states_size, int node_index,
								 const struct belief * __restrict__ edge_messages,
								 const int * __restrict__ dest_nodes_to_edges_nodes,
								 const int * __restrict__ dest_nodes_to_edges_edges,
								 int num_vertices, int num_edges){
	int i;
    int num_variables, start_index, end_index, edge_index;
	float sum;
    struct belief new_belief;

	num_variables = node_states_size[node_index];

	for(i = 0; i < num_variables; ++i){
		new_belief.data[i] = 1.0;
	}

	start_index = dest_nodes_to_edges_nodes[node_index];
	if(node_index + 1 == num_vertices){
		end_index = num_edges;
	}
	else {
		end_index = dest_nodes_to_edges_nodes[node_index + 1];
	}

	for(i = start_index; i < end_index; ++i){
		edge_index = dest_nodes_to_edges_edges[i];

		combine_message(&new_belief, edge_messages, num_variables, edge_index);

	}
	if(start_index < end_index){
		for(i = 0; i < num_variables; ++i){
			node_states[node_index].data[i] *= new_belief.data[i];
		}
	}
	sum = 0.0;
	for(i = 0; i < num_variables; ++i){
		sum += node_states[node_index].data[i];
	}
	if(sum <= 0.0){
		sum = 1.0;
	}

	for(i = 0; i < num_variables; ++i){
        node_states[node_index].data[i] /= sum;
	}
}

/**
 * Computes the page rank of the nodes for OpenACC
 * @param node_states The current nodes and their beliefs in the graph
 * @param node_index The index of the current node
 * @param edge_messages The current messages buffered at the edges
 * @param dest_nodes_to_edges_nodes Parallel array; maps destination nodes to their edges; first half; the nodes to their starting index in dest_nodes_to_edges_edges
 * @param dest_nodes_to_edges_edges Parallel array; maps destination nodes to their edges; second half; lists edges by node
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 */
#pragma acc routine
static void marginalize_page_rank_nodes_acc(struct belief * __restrict__ node_states, const int * __restrict__ node_states_size, int node_index,
                                            struct belief * __restrict__ edge_messages,
                                            const int * __restrict__ dest_nodes_to_edges_nodes,
                                            const int * __restrict__ dest_nodes_to_edges_edges,
                                            int num_vertices, int num_edges) {
    int i;
    int num_variables, start_index, end_index, edge_index;
    float factor;
    struct belief new_belief;

    num_variables = node_states_size[node_index];

    for(i = 0; i < num_variables; ++i){
        new_belief.data[i] = 0.0;
    }

    start_index = dest_nodes_to_edges_nodes[node_index];
    if(node_index + 1 == num_vertices){
        end_index = num_edges;
    }
    else {
        end_index = dest_nodes_to_edges_nodes[node_index + 1];
    }

    for(i = start_index; i < end_index; ++i){
        edge_index = dest_nodes_to_edges_edges[i];

        combine_page_rank_message(&new_belief, edge_messages, num_variables, edge_index);
    }
    if (start_index < end_index) {
        factor = (1 - DAMPENING_FACTOR) / (end_index - start_index);
        for (i = 0; i < num_variables; ++i) {
            node_states[node_index].data[i] = factor + DAMPENING_FACTOR * new_belief.data[i];
        }
    }
}

/**
 * Computes the argmax the nodes for OpenACC
 * @param node_states The current nodes and their beliefs in the graph
 * @param node_index The index of the current node
 * @param edge_messages The current messages buffered at the edges
 * @param dest_nodes_to_edges_nodes Parallel array; maps destination nodes to their edges; first half; the nodes to their starting index in dest_nodes_to_edges_edges
 * @param dest_nodes_to_edges_edges Parallel array; maps destination nodes to their edges; second half; lists edges by node
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 */
#pragma acc routine
static void argmax_node_acc(struct belief * __restrict__ node_states, const int * __restrict__ node_states_size, int node_index,
                                 const struct belief * __restrict__ edge_messages,
                                 const int * __restrict__ dest_nodes_to_edges_nodes,
                                 const int * __restrict__ dest_nodes_to_edges_edges,
                                 int num_vertices, int num_edges){
    int i;
    int num_variables, start_index, end_index, edge_index;
    struct belief new_belief;

    num_variables = node_states_size[node_index];

    for(i = 0; i < num_variables; ++i){
        new_belief.data[i] = -1.0f;
    }

    start_index = dest_nodes_to_edges_nodes[node_index];
    if(node_index + 1 == num_vertices){
        end_index = num_edges;
    }
    else {
        end_index = dest_nodes_to_edges_nodes[node_index + 1];
    }

    for(i = start_index; i < end_index; ++i){
        edge_index = dest_nodes_to_edges_edges[i];

        combine_viterbi_message(&new_belief, edge_messages, num_variables, edge_index);

    }
    if(start_index < end_index){
        for(i = 0; i < num_variables; ++i){
            node_states[node_index].data[i] = fmaxf(node_states[node_index].data[i], new_belief.data[i]);
        }
    }
}

/**
 * Performs one iteration of loopy BP on the graph
 * @param graph The graph
 */
void loopy_propagate_one_iteration(Graph_t graph){
	int i;
    int num_variables, num_vertices, num_edges, num_work_queue_items, current_index;
	int * dest_node_to_edges_nodes;
	int * dest_node_to_edges_edges;
	int * src_node_to_edges_nodes;
	int * src_node_to_edges_edges;
	int * work_queue_nodes;
	struct belief *node_states;
	int *node_states_size;
	struct joint_probability * edge_joint_probability;
	int edge_joint_probability_dim_x;
	int edge_joint_probability_dim_y;
	struct belief *current_edge_messages;
	float *edges_messages_current;
	float *edges_messages_previous;

	current_edge_messages = graph->edges_messages;
	edges_messages_current = graph->edges_messages_current;
	edges_messages_previous = graph->edges_messages_previous;

	edge_joint_probability = &(graph->edge_joint_probability);
	edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
	edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;

	struct belief buffer;

	num_vertices = graph->current_num_vertices;
	dest_node_to_edges_nodes = graph->dest_nodes_to_edges_node_list;
	dest_node_to_edges_edges = graph->dest_nodes_to_edges_edge_list;
	src_node_to_edges_nodes = graph->src_nodes_to_edges_node_list;
	src_node_to_edges_edges = graph->src_nodes_to_edges_edge_list;
    num_edges = graph->current_num_edges;
	node_states = graph->node_states;
	node_states_size = graph->node_states_size;

	work_queue_nodes = graph->work_queue_nodes;
	num_work_queue_items = graph->num_work_items_nodes;

#pragma omp parallel for default(none) shared(node_states, num_vertices, dest_node_to_edges_nodes, dest_node_to_edges_edges, src_node_to_edges_nodes, src_node_to_edges_edges, num_edges, current_edge_messages, edge_joint_probability, work_queue_nodes, num_work_queue_items, node_states_size, edge_joint_probability_dim_x, edge_joint_probability_dim_y, edges_messages_current, edges_messages_previous) private(buffer, i, num_variables, current_index) //schedule(dynamic, 16)
    for(i = 0; i < num_work_queue_items; ++i){
		current_index = work_queue_nodes[i];

		num_variables = node_states_size[current_index];

		initialize_message_buffer(&buffer, node_states, current_index, num_variables);

		//read incoming messages
		read_incoming_messages(&buffer, dest_node_to_edges_nodes, dest_node_to_edges_edges, current_edge_messages, num_edges, num_vertices, num_variables, current_index);

/*
		printf("Message at node\n");
		print_node(graph, i);
		printf("[\t");
		for(j = 0; j < num_variables; ++j){
			printf("%.6lf\t", message_buffer[j]);
		}
		printf("\t]\n");*/


		//send belief
		send_message_for_node(src_node_to_edges_nodes, src_node_to_edges_edges, &buffer, num_edges, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y, current_edge_messages, edges_messages_previous, edges_messages_current, num_vertices, current_index);

	}

	marginalize_loopy_nodes(graph, current_edge_messages, num_vertices);
	update_work_queue_nodes(graph, PRECISION_ITERATION);
}


/**
 * Performs one iteration of edge-optimized loopy BP
 * @param graph The graph
 */
void loopy_propagate_edge_one_iteration(Graph_t graph){
    int i;
	int num_edges, num_nodes, src_node_index, dest_node_index, current_index, num_work_items_edges;
    struct belief *node_states;
    int *node_states_size;
    struct joint_probability *edge_joint_probability;
    int edge_joint_probability_dim_x;
    int edge_joint_probability_dim_y;
    struct belief *current_edge_messages;
    float *edges_messages_previous;
    float *edges_messages_current;

	int * edges_src_index;
	int * edges_dest_index;
	int * work_queue_edges;

    current_edge_messages = graph->edges_messages;
    edges_messages_previous = graph->edges_messages_previous;
    edges_messages_current = graph->edges_messages_current;

	edge_joint_probability = &(graph->edge_joint_probability);
	edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
	edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;

    num_edges = graph->current_num_edges;
	num_nodes = graph->current_num_vertices;
    node_states = graph->node_states;
    node_states_size = graph->node_states_size;
	edges_src_index = graph->edges_src_index;
	edges_dest_index = graph->edges_dest_index;

	num_work_items_edges = graph->num_work_items_edges;
	work_queue_edges = graph->work_queue_edges;

    #pragma omp parallel for default(none) shared(node_states, edge_joint_probability, current_edge_messages, edges_messages_previous, edges_messages_current, edges_src_index, num_edges, work_queue_edges, num_work_items_edges, edge_joint_probability_dim_x, edge_joint_probability_dim_y) private(src_node_index, i, current_index)
    for(i = 0; i < num_work_items_edges; ++i){
		current_index = work_queue_edges[i];

        src_node_index = edges_src_index[current_index];
        send_message_for_edge_iteration(node_states, src_node_index, current_index, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y, current_edge_messages, edges_messages_previous, edges_messages_current);
    }

    #pragma omp parallel for default(none) shared(current_edge_messages, node_states, node_states_size, edges_dest_index, num_edges, work_queue_edges, num_work_items_edges) private(dest_node_index, i, current_index)
    for(i = 0; i < num_work_items_edges; ++i){
		current_index = work_queue_edges[i];

        dest_node_index = edges_dest_index[current_index];
		combine_loopy_edge(current_index, current_edge_messages, dest_node_index, node_states, node_states_size);
    }
	/*
#pragma omp parallel for default(none) shared(node_states, num_vars, num_nodes) private(i)
	for(i = 0; i < num_nodes; ++i){
		marginalize_loopy_node_edge(node_states, num_vars[i]);
	}*/
	marginalize_loopy_nodes(graph, current_edge_messages, num_nodes);
	update_work_queue_edges(graph, PRECISION_ITERATION);
}

/**
 * Performs one iteration of page rank on the graph
 * @param graph The graph
 */
void page_rank_one_iteration(Graph_t graph){
    int i;
    int num_variables, num_vertices, num_edges;
    int * dest_node_to_edges_nodes;
    int * dest_node_to_edges_edges;
    int * src_node_to_edges_nodes;
    int * src_node_to_edges_edges;
    struct belief *node_states;
    int *node_states_size;
    struct joint_probability *edge_joint_probability;
    int edge_joint_probability_dim_x;
    int edge_joint_probability_dim_y;
    struct belief *current_edge_messages;
    float * edges_messages_previous;
    float * edges_messages_current;

    current_edge_messages = graph->edges_messages;
    edges_messages_current = graph->edges_messages_current;
    edges_messages_previous = graph->edges_messages_previous;

	edge_joint_probability = &(graph->edge_joint_probability);
	edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
	edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;

    struct belief buffer;

    num_vertices = graph->current_num_vertices;
    dest_node_to_edges_nodes = graph->dest_nodes_to_edges_node_list;
    dest_node_to_edges_edges = graph->dest_nodes_to_edges_edge_list;
    src_node_to_edges_nodes = graph->src_nodes_to_edges_node_list;
    src_node_to_edges_edges = graph->src_nodes_to_edges_edge_list;
    num_edges = graph->current_num_edges;
    node_states = graph->node_states;
    node_states_size = graph->node_states_size;

#pragma omp parallel for default(none) shared(node_states, node_states_size, num_vertices, dest_node_to_edges_nodes, dest_node_to_edges_edges, src_node_to_edges_nodes, src_node_to_edges_edges, num_edges, current_edge_messages, edges_messages_current, edges_messages_previous, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y) private(buffer, i, num_variables) //schedule(dynamic, 16)
    for(i = 0; i < num_vertices; ++i){
        num_variables = node_states_size[i];

        initialize_message_buffer(&buffer, node_states, i, num_variables);

        //read incoming messages
        read_incoming_messages(&buffer, dest_node_to_edges_nodes, dest_node_to_edges_edges, current_edge_messages, num_edges, num_vertices, num_variables, i);

/*
		printf("Message at node\n");
		print_node(graph, i);
		printf("[\t");
		for(j = 0; j < num_variables; ++j){
			printf("%.6lf\t", message_buffer[j]);
		}
		printf("\t]\n");*/


        //send belief
        send_message_for_node(src_node_to_edges_nodes, src_node_to_edges_edges, &buffer, num_edges, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y, current_edge_messages, edges_messages_previous, edges_messages_current, num_vertices, i);

    }

    marginalize_page_rank_nodes(graph, current_edge_messages, num_vertices);
}


/**
 * Performs one iteration of edge-optimized PageRank
 * @param graph The graph
 */
void page_rank_edge_one_iteration(Graph_t graph){
    int i;
    int num_edges, num_nodes, src_node_index, dest_node_index;
    struct belief *node_states;
    int *node_states_size;
    struct joint_probability *edge_joint_probability;
    int edge_joint_probability_dim_x;
    int edge_joint_probability_dim_y;
    struct belief *current_edge_messages;
    float *edge_messages_current;
    float *edge_messages_previous;

    int * edges_src_index;
    int * edges_dest_index;

    current_edge_messages = graph->edges_messages;
    edge_messages_current = graph->edges_messages_current;
    edge_messages_previous = graph->edges_messages_previous;

	edge_joint_probability = &(graph->edge_joint_probability);
	edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
	edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;

    num_edges = graph->current_num_edges;
    num_nodes = graph->current_num_vertices;
    node_states = graph->node_states;
    node_states_size = graph->node_states_size;
    edges_src_index = graph->edges_src_index;
    edges_dest_index = graph->edges_dest_index;

#pragma omp parallel for default(none) shared(node_states, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y, current_edge_messages, edge_messages_current, edge_messages_previous, edges_src_index, num_edges) private(src_node_index, i)
    for(i = 0; i < num_edges; ++i){
        src_node_index = edges_src_index[i];
        send_message_for_edge_iteration(node_states, src_node_index, i, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y, current_edge_messages, edge_messages_previous, edge_messages_current);
    }

#pragma omp parallel for default(none) shared(current_edge_messages, node_states, node_states_size, edges_dest_index, num_edges) private(dest_node_index, i)
    for(i = 0; i < num_edges; ++i){
        dest_node_index = edges_dest_index[i];
        combine_loopy_edge(i, current_edge_messages, dest_node_index, node_states, node_states_size);
    }
    /*
#pragma omp parallel for default(none) shared(node_states, num_vars, num_nodes) private(i)
    for(i = 0; i < num_nodes; ++i){
        marginalize_loopy_node_edge(node_states, num_vars[i]);
    }*/
    marginalize_page_rank_nodes(graph, current_edge_messages, num_nodes);
}

/**
 * Performs one iteration of Viterbi on the graph
 * @param graph The graph
 */
void viterbi_one_iteration(Graph_t graph){
    int i;
    int num_variables, num_vertices, num_edges;
    int * dest_node_to_edges_nodes;
    int * dest_node_to_edges_edges;
    int * src_node_to_edges_nodes;
    int * src_node_to_edges_edges;
    struct belief *node_states;
    int *node_states_size;
    struct joint_probability *edge_joint_probability;
    int edge_joint_probability_dim_x;
    int edge_joint_probability_dim_y;
    struct belief *current_edge_messages;
    float *edges_messages_current;
    float *edges_messages_previous;

    current_edge_messages = graph->edges_messages;
    edges_messages_previous = graph->edges_messages_previous;
    edges_messages_current = graph->edges_messages_current;

	edge_joint_probability = &(graph->edge_joint_probability);
	edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
	edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;

    struct belief buffer;

    num_vertices = graph->current_num_vertices;
    dest_node_to_edges_nodes = graph->dest_nodes_to_edges_node_list;
    dest_node_to_edges_edges = graph->dest_nodes_to_edges_edge_list;
    src_node_to_edges_nodes = graph->src_nodes_to_edges_node_list;
    src_node_to_edges_edges = graph->src_nodes_to_edges_edge_list;
    num_edges = graph->current_num_edges;
    node_states = graph->node_states;
    node_states_size = graph->node_states_size;

#pragma omp parallel for default(none) shared(node_states, node_states_size, num_vertices, dest_node_to_edges_nodes, dest_node_to_edges_edges, src_node_to_edges_nodes, src_node_to_edges_edges, num_edges, current_edge_messages, edges_messages_current, edges_messages_previous, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y) private(buffer, i, num_variables) //schedule(dynamic, 16)
    for(i = 0; i < num_vertices; ++i){
        num_variables = node_states_size[i];

        initialize_message_buffer(&buffer, node_states, i, num_variables);

        //read incoming messages
        read_incoming_messages(&buffer, dest_node_to_edges_nodes, dest_node_to_edges_edges, current_edge_messages, num_edges, num_vertices, num_variables, i);

/*
		printf("Message at node\n");
		print_node(graph, i);
		printf("[\t");
		for(j = 0; j < num_variables; ++j){
			printf("%.6lf\t", message_buffer[j]);
		}
		printf("\t]\n");*/


        //send belief
        send_message_for_node(src_node_to_edges_nodes, src_node_to_edges_edges, &buffer, num_edges, edge_joint_probability,
							  edge_joint_probability_dim_x, edge_joint_probability_dim_y, current_edge_messages, edges_messages_previous, edges_messages_current, num_vertices, i);

    }

    argmax_loopy_nodes(graph, current_edge_messages, num_vertices);
}


/**
 * Performs one iteration of edge-optimized PageRank
 * @param graph The graph
 */
void viterbi_edge_one_iteration(Graph_t graph){
    int i;
    int num_edges, num_nodes, src_node_index, dest_node_index;
    struct belief *node_states;
    int *node_states_size;
    struct joint_probability *edge_joint_probability;
    int edge_joint_probability_dim_x;
    int edge_joint_probability_dim_y;
    struct belief *current_edge_messages;
    float *edge_messages_current;
    float *edge_messages_previous;

    int * edges_src_index;
    int * edges_dest_index;

    current_edge_messages = graph->edges_messages;
    edge_messages_previous = graph->edges_messages_previous;
    edge_messages_current = graph->edges_messages_current;

	edge_joint_probability = &(graph->edge_joint_probability);
	edge_joint_probability_dim_x = graph->edge_joint_probability_dim_x;
	edge_joint_probability_dim_y = graph->edge_joint_probability_dim_y;

    num_edges = graph->current_num_edges;
    num_nodes = graph->current_num_vertices;
    node_states = graph->node_states;
    node_states_size = graph->node_states_size;
    edges_src_index = graph->edges_src_index;
    edges_dest_index = graph->edges_dest_index;

#pragma omp parallel for default(none) shared(node_states, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y, current_edge_messages, edge_messages_previous, edge_messages_current, edges_src_index, num_edges) private(src_node_index, i)
    for(i = 0; i < num_edges; ++i){
        src_node_index = edges_src_index[i];
        send_message_for_edge_iteration(node_states, src_node_index, i, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y, current_edge_messages, edge_messages_previous, edge_messages_current);
    }

#pragma omp parallel for default(none) shared(current_edge_messages, node_states, node_states_size, edges_dest_index, num_edges) private(dest_node_index, i)
    for(i = 0; i < num_edges; ++i){
        dest_node_index = edges_dest_index[i];
        combine_loopy_edge(i, current_edge_messages, dest_node_index, node_states, node_states_size);
    }
    /*
#pragma omp parallel for default(none) shared(node_states, num_vars, num_nodes) private(i)
    for(i = 0; i < num_nodes; ++i){
        marginalize_loopy_node_edge(node_states, num_vars[i]);
    }*/
    argmax_loopy_nodes(graph, current_edge_messages, num_nodes);
}

/**
 * Runs edge-optimized loopy BP until convergence or max iterations reached
 * @param graph The graph
 * @param convergence The convergence threshold below which processing will stop
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations executed
 */
int loopy_propagate_until_edge(Graph_t graph, float convergence, int max_iterations){
    int j, k;
    int i, num_edges, num_nodes, num_variables;
    float delta, diff, previous_delta, sum;
    struct belief *states;
    int *states_size;
    float *edge_messages_current;
    float *edge_messages_previous;

    edge_messages_current = graph->edges_messages_current;
    edge_messages_previous = graph->edges_messages_previous;

    num_edges = graph->current_num_edges;
    num_nodes = graph->current_num_vertices;

    previous_delta = -1.0f;
    delta = 0.0f;

	init_work_queue_edges(graph);

    for(i = 0; i < max_iterations; ++i){
        //printf("Current iteration: %d\n", i+1);
        loopy_propagate_edge_one_iteration(graph);

        delta = 0.0f;

#pragma omp parallel for default(none) shared(edge_messages_previous, edge_messages_current, num_edges)  private(j, diff) reduction(+:delta)
        for(j = 0; j < num_edges; ++j){
            diff = edge_messages_previous[j] - edge_messages_current[j];
            //printf("Previous: %f\n", previous_edge_messages[j].data[k]);
            //printf("Current : %f\n", current_edge_messages[j].data[k]);
            if(diff != diff){
                diff = 0.0f;
            }
            delta += fabsf(diff);
        }

        //printf("Current delta: %.6lf\n", delta);
        //printf("Previous delta: %.6lf\n", previous_delta);
        if(delta < convergence || fabsf(delta - previous_delta) < convergence){
            break;
        }
		if(i < max_iterations - 1) {
			previous_delta = delta;
		}
    }
    if(i == max_iterations){
        printf("No Convergence: previous: %f vs current: %f\n", previous_delta, delta);
    }

    states = graph->node_states;
    states_size = graph->node_states_size;

#pragma omp parallel for default(none) shared(states, states_size, num_nodes) private(sum, num_variables, k)
    for(j = 0; j < num_nodes; ++j){
        sum = 0.0;
        num_variables = states_size[j];
#pragma omp simd safelen(AVG_STATES)
#pragma simd vectorlength(AVG_STATES)
        for (k = 0; k < num_variables; ++k) {
            sum += states[j].data[k];
        }
        if (sum <= 0.0) {
            sum = 1.0;
        }

#pragma omp simd safelen(AVG_STATES)
#pragma simd vectorlength(AVG_STATES)
        for (k = 0; k < num_variables; ++k) {
            states[j].data[k] /= sum;
        }
    }
    return i;
}


/**
 * Runs edge-optimized PageRank until convergence or max iterations reached
 * @param graph The graph
 * @param convergence The convergence threshold below which processing will stop
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations executed
 */
int page_rank_until_edge(Graph_t graph, float convergence, int max_iterations){
    int j;
    int i, num_edges;
    float delta, diff, previous_delta;
    float *edge_messages_current;
    float *edge_messages_previous;

    edge_messages_current = graph->edges_messages_current;
    edge_messages_previous = graph->edges_messages_previous;

    num_edges = graph->current_num_edges;

    previous_delta = -1.0f;
    delta = 0.0f;

    for(i = 0; i < max_iterations; ++i){
        //printf("Current iteration: %d\n", i+1);
        page_rank_edge_one_iteration(graph);

        delta = 0.0f;

#pragma omp parallel for default(none) shared(edge_messages_previous, edge_messages_current, num_edges)  private(j, diff) reduction(+:delta)
        for(j = 0; j < num_edges; ++j){
            diff = edge_messages_previous[j] - edge_messages_current[j];
            //printf("Previous: %f\n", previous_edge_messages[j].data[k]);
            //printf("Current : %f\n", current_edge_messages[j].data[k]);
            if(diff != diff){
                diff = 0.0f;
            }
            delta += fabsf(diff);
        }

        //printf("Current delta: %.6lf\n", delta);
        //printf("Previous delta: %.6lf\n", previous_delta);
        if(delta < convergence || fabsf(delta - previous_delta) < convergence){
            break;
        }
        if(i < max_iterations - 1) {
            previous_delta = delta;
        }
    }
    if(i == max_iterations){
        printf("No Convergence: previous: %f vs current: %f\n", previous_delta, delta);
    }
    return i;
}

/**
 * Runs node-optimized BP until either convergence threshold met or max iterations
 * @param graph The graph
 * @param convergence The covergence threshold below which the graph will cease processing
 * @param max_iterations The maximum number of iterations
 * @return The actual number of iterations executed
 */
int loopy_propagate_until(Graph_t graph, float convergence, int max_iterations){
	int j;
    int i, num_edges;
	float delta, diff, previous_delta;
	float *edges_messages_current;
	float *edges_messages_previous;

	edges_messages_current = graph->edges_messages_current;
	edges_messages_previous = graph->edges_messages_previous;

	num_edges = graph->current_num_edges;

	previous_delta = -1.0f;
	delta = 0.0;

	init_work_queue_nodes(graph);

	for(i = 0; i < max_iterations; ++i){
		//printf("Current iteration: %d\n", i+1);
		loopy_propagate_one_iteration(graph);

		delta = 0.0;

#pragma omp parallel for default(none) shared(edges_messages_previous, edges_messages_current, num_edges)  private(j, diff) reduction(+:delta)
		for(j = 0; j < num_edges; ++j){
            diff = edges_messages_previous[j] - edges_messages_current[j];
            //printf("Previous Edge[%d][%d]: %f\n", j, k, previous_edge_messages[j].data[k]);
            //printf("Current Edge[%d][%d]: %f\n", j, k, current_edge_messages[j].data[k]);
            if(diff != diff){
                diff = 0.0;
            }
            delta += fabsf(diff);
		}

		if(delta < convergence || fabsf(delta - previous_delta) < convergence){
			break;
		}
		if(i < max_iterations - 1) {
			previous_delta = delta;
		}
	}
	if(i == max_iterations){
		printf("No Convergence: previous: %f vs current: %f\n", previous_delta, delta);
	}
//	assert(i > 0);
	return i;
}

/**
 * Runs PageRank until either convergence threshold met or max iterations
 * @param graph The graph
 * @param convergence The covergence threshold below which the graph will cease processing
 * @param max_iterations The maximum number of iterations
 * @return The actual number of iterations executed
 */
int page_rank_until(Graph_t graph, float convergence, int max_iterations){
    int j;
    int i, num_edges;
    float delta, diff, previous_delta;
    float *edges_message_previous;
    float *edges_message_current;

	edges_message_current = graph->edges_messages_current;
	edges_message_previous = graph->edges_messages_previous;

    num_edges = graph->current_num_edges;

    previous_delta = -1.0f;
    delta = 0.0;

    for(i = 0; i < max_iterations; ++i){
        //printf("Current iteration: %d\n", i+1);
        page_rank_one_iteration(graph);

        delta = 0.0;

#pragma omp parallel for default(none) shared(edges_message_previous, edges_message_current, num_edges)  private(j, diff) reduction(+:delta)
        for(j = 0; j < num_edges; ++j){
            diff = edges_message_previous[j] - edges_message_current[j];
            //printf("Previous Edge[%d][%d]: %f\n", j, k, previous_edge_messages[j].data[k]);
            //printf("Current Edge[%d][%d]: %f\n", j, k, current_edge_messages[j].data[k]);
            if(diff != diff){
                diff = 0.0;
            }
            delta += fabsf(diff);
        }

        if(delta < convergence || fabsf(delta - previous_delta) < convergence){
            break;
        }
        if(i < max_iterations - 1) {
            previous_delta = delta;
        }
    }
    if(i == max_iterations){
        printf("No Convergence: previous: %f vs current: %f\n", previous_delta, delta);
    }
//    assert(i > 0);
    return i;
}


/**
 * Runs Viterbi until either convergence threshold met or max iterations
 * @param graph The graph
 * @param convergence The covergence threshold below which the graph will cease processing
 * @param max_iterations The maximum number of iterations
 * @return The actual number of iterations executed
 */
int viterbi_until(Graph_t graph, float convergence, int max_iterations){
    int j, k, num_variables;
    int i, num_edges, num_nodes;
    float delta, diff, previous_delta, sum;

    float *edges_messages_current;
    float *edges_messages_previous;
    struct belief *states;
    int *states_size;

    edges_messages_current = graph->edges_messages_current;
	edges_messages_previous = graph->edges_messages_previous;

    num_edges = graph->current_num_edges;
    num_nodes = graph->current_num_vertices;

    previous_delta = -1.0f;
    delta = 0.0;

    for(i = 0; i < max_iterations; ++i){
        //printf("Current iteration: %d\n", i+1);
        viterbi_one_iteration(graph);

        delta = 0.0;

#pragma omp parallel for default(none) shared(edges_messages_previous, edges_messages_current, num_edges)  private(j, diff) reduction(+:delta)
        for(j = 0; j < num_edges; ++j){
            diff = edges_messages_previous[j] - edges_messages_current[j];
            //printf("Previous Edge[%d][%d]: %f\n", j, k, previous_edge_messages[j].data[k]);
            //printf("Current Edge[%d][%d]: %f\n", j, k, current_edge_messages[j].data[k]);
            if(diff != diff){
                diff = 0.0;
            }
            delta += fabsf(diff);
        }

        if(delta < convergence || fabsf(delta - previous_delta) < convergence){
            break;
        }
        if(i < max_iterations - 1) {
            previous_delta = delta;
        }
    }
    if(i == max_iterations){
        printf("No Convergence: previous: %f vs current: %f\n", previous_delta, delta);
    }

    states = graph->node_states;
    states_size = graph->node_states_size;

    #pragma omp parallel for default(none) shared(states, states_size, num_nodes) private(sum, num_variables, k)
    for(j = 0; j < num_nodes; ++j){
        sum = 0.0;
        num_variables = states_size[j];
        #pragma omp simd safelen(AVG_STATES)
        #pragma simd vectorlength(AVG_STATES)
        for (k = 0; k < num_variables; ++k) {
            sum += states[j].data[k];
        }
        if (sum <= 0.0) {
            sum = 1.0;
        }

        #pragma omp simd safelen(AVG_STATES)
        #pragma simd vectorlength(AVG_STATES)
        for (k = 0; k < num_variables; ++k) {
            states[j].data[k] /= sum;
        }
    }
//    assert(i > 0);
    return i;
}

static void update_work_queue_nodes_acc(int * __restrict__ num_work_items_nodes,
										int * __restrict__ work_queue_scratch, int * __restrict__ work_queue_nodes,
										const struct belief * __restrict__ node_states,
										float * __restrict__ node_states_previous, float * __restrict__ node_states_current,
										int num_vertices, float convergence) {
	int current_index, i;

	current_index = 0;
#pragma omp parallel for default(none) shared(current_index, num_work_items_nodes, work_queue_scratch, convergence, work_queue_nodes, node_states, node_states_previous, node_states_current) private(i)
#pragma acc kernels copyin(work_queue_nodes[0:num_vertices], node_states[0:num_vertices], node_states_current[0:num_vertices], node_states_previous[0:num_vertices]) copyout(work_queue_scratch[0:num_vertices])
	for(i = 0; i < *num_work_items_nodes; ++i) {
		if(fabs(node_states_current[work_queue_nodes[i]] - node_states_previous[work_queue_nodes[i]]) >= convergence) {
#pragma omp critical
#pragma acc atomic capture
			{
				work_queue_scratch[current_index] = work_queue_nodes[i];
				current_index++;
			}
		}
	}
	memcpy(work_queue_nodes, work_queue_scratch, (size_t)num_vertices);
	*num_work_items_nodes = current_index;
}


/**
 * Runs loopy BP for OpenACC
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 * @param dest_node_to_edges_nodes Parallel array; maps destination nodes to their edges; first half; maps nodes to their starting indices in dest_node_to_edges_edges
 * @param dest_node_to_edges_edges Parallel array; maps destination nodes to their edges; second half; lists edges by nodes
 * @param src_node_to_edges_nodes Parallel array; maps source nodes to their edges; first half; maps nodes to their starting indices in src_node_to_edges_edges
 * @param src_node_to_edges_edges Parallel array; maps source nodes to their edges; second half; lists edges by nodes
 * @param node_states The current node states
 * @param previous_messages The previous messages in the graph
 * @param current_messages The current messages in the graph
 * @param joint_probabilities The joint probability tables of the edges
 * @param work_items_nodes The work queue for the nodes
 * @param work_queue_scratch The scratch space for adjusting the queue
 * @param num_work_items_nodes The number of items in the work queue
 * @param max_iterations The maximum number of iterations to run for
 * @param convergence The convergence threshold
 * @return The actual number of iterations used
 */
static int loopy_propagate_iterations_acc(int num_vertices, int num_edges,
										   const int * __restrict__ dest_node_to_edges_nodes,
										   const int * __restrict__ dest_node_to_edges_edges,
										   const int * __restrict__ src_node_to_edges_nodes,
										   const int * __restrict__ src_node_to_edges_edges,
										   struct belief * __restrict__ node_states,
										  const int * __restrict__ node_states_size,
										   float * node_states_previous,
										   float * node_states_current,
										   struct belief * __restrict__ current_messages,
										   float * __restrict__ messages_previous,
										   float * __restrict__ messages_current,
										   const struct joint_probability * __restrict__ edge_joint_probability,
										   const int edge_joint_probability_dim_x,
										   const int edge_joint_probability_dim_y,
										   int * __restrict__ work_items_nodes, int * __restrict__ work_queue_scratch,
										   int num_work_items_nodes,
										   int max_iterations,
										   float convergence){
	int j, k, current_index;
    int i, num_variables, num_iter;
	float delta, previous_delta, diff;
	struct belief *curr_messages;

	curr_messages = current_messages;


	struct belief belief_buffer;

	num_iter = 0;

	previous_delta = -1.0f;
	delta = 0.0f;

    for(i = 0; i < max_iterations; i+= BATCH_SIZE) {
#pragma acc data present_or_copy(node_states[0:(num_vertices)], node_states_previous[0:(num_vertices)], node_states_current[0:(num_vertices)], curr_messages[0:(num_edges)], messages_current[0:(num_edges)], messages_previous[0:(num_edges)], work_items_nodes[0:(num_work_items_nodes)]) present_or_copyin(dest_node_to_edges_nodes[0:num_vertices], dest_node_to_edges_edges[0:num_edges], src_node_to_edges_nodes[0:num_vertices], src_node_to_edges_edges[0:num_edges], edge_joint_probability[0:1], node_states_size[0:(num_vertices)])
        {
            //printf("Current iteration: %d\n", i+1);
            for (j = 0; j < BATCH_SIZE; ++j) {
#pragma acc kernels
                for (k = 0; k < num_work_items_nodes; ++k) {
					current_index = work_items_nodes[k];

                    num_variables = node_states_size[current_index];

                    initialize_message_buffer(&belief_buffer, node_states, current_index, num_variables);

                    //read incoming messages
                    read_incoming_messages(&belief_buffer, dest_node_to_edges_nodes, dest_node_to_edges_edges, curr_messages, num_edges, num_vertices,
                                           num_variables, current_index);

/*
		printf("Message at node\n");
		print_node(graph, i);
		printf("[\t");
		for(j = 0; j < num_variables; ++j){
			printf("%.6lf\t", message_buffer[j]);
		}
		printf("\t]\n");*/


                    //send belief
                    send_message_for_node(src_node_to_edges_nodes, src_node_to_edges_edges, &belief_buffer, num_edges, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y,
                                          curr_messages, messages_previous, messages_current, num_vertices, current_index);

                }

#pragma acc kernels
                for (k = 0; k < num_work_items_nodes; ++k) {
					current_index = work_items_nodes[k];

                    marginalize_node_acc(node_states, node_states_size, current_index, curr_messages, dest_node_to_edges_nodes, dest_node_to_edges_edges, num_vertices,
                                         num_edges);
                }
            }

            //update_work_queue_nodes_acc(&num_work_items_nodes, work_queue_scratch, work_items_nodes, node_states, node_states_previous, node_states_current, num_vertices, convergence);


            delta = 0.0f;
#pragma acc kernels
            for (j = 0; j < num_vertices; ++j) {
                diff = node_states_previous[j] - node_states_current[j];
                if (diff != diff) {
                    diff = 0.0f;
                }
                delta += fabsf(diff);
            }
        }
        if(delta < convergence || fabsf(delta - previous_delta) < convergence){
            break;
        }
        previous_delta = delta;
        num_iter += BATCH_SIZE;
    }
	if(i == max_iterations) {
		printf("No Convergence: previous: %f vs current: %f\n", previous_delta, delta);
	}


	return num_iter;
}


/**
 * Runs PageRank for OpenACC
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 * @param dest_node_to_edges_nodes Parallel array; maps destination nodes to their edges; first half; maps nodes to their starting indices in dest_node_to_edges_edges
 * @param dest_node_to_edges_edges Parallel array; maps destination nodes to their edges; second half; lists edges by nodes
 * @param src_node_to_edges_nodes Parallel array; maps source nodes to their edges; first half; maps nodes to their starting indices in src_node_to_edges_edges
 * @param src_node_to_edges_edges Parallel array; maps source nodes to their edges; second half; lists edges by nodes
 * @param node_states The current node states
 * @param previous_messages The previous messages in the graph
 * @param current_messages The current messages in the graph
 * @param joint_probabilities The joint probability tables of the edges
 * @param max_iterations The maximum number of iterations to run for
 * @param convergence The convergence threshold
 * @return The actual number of iterations used
 */
static int page_rank_iterations_acc(int num_vertices, int num_edges,
                                                   const int * __restrict__ dest_node_to_edges_nodes,
                                                   const int * __restrict__ dest_node_to_edges_edges,
                                                   const int * __restrict__ src_node_to_edges_nodes,
                                                   int *src_node_to_edges_edges,
                                                   struct belief * __restrict__ node_states,
													const int * __restrict__ node_states_size,
													struct belief * __restrict__ current_messages,
                                                   float * __restrict__ messages_previous,
                                                   float * __restrict__ messages_current,
                                                   const struct joint_probability * __restrict__ edge_joint_probability,
                                                   const int edge_joint_probability_dim_x,
                                                   const int edge_joint_probability_dim_y,
                                                   int max_iterations,
                                                   float convergence){
    int j, k;
    int i, num_variables, num_iter;
    float delta, previous_delta, diff;
    struct belief *curr_messages;

    curr_messages = current_messages;


    struct belief belief_buffer;

    num_iter = 0;

    previous_delta = -1.0f;
    delta = 0.0f;

    for(i = 0; i < max_iterations; i+= BATCH_SIZE) {
#pragma acc data present_or_copy(node_states[0:(num_vertices)], curr_messages[0:(num_edges)], messages_current[0:(num_edges)], messages_previous[0:(num_edges)]) present_or_copyin(dest_node_to_edges_nodes[0:num_vertices], dest_node_to_edges_edges[0:num_edges], src_node_to_edges_nodes[0:num_vertices], src_node_to_edges_edges[0:num_edges], edge_joint_probability[0:1], node_states_size[0:(num_vertices)])
        {
            //printf("Current iteration: %d\n", i+1);
            for (j = 0; j < BATCH_SIZE; ++j) {
#pragma acc kernels
                for (k = 0; k < num_vertices; ++k) {
                    num_variables = node_states_size[k];

                    initialize_message_buffer(&belief_buffer, node_states, k, num_variables);

                    //read incoming messages
                    read_incoming_messages(&belief_buffer, dest_node_to_edges_nodes, dest_node_to_edges_edges, curr_messages, num_edges, num_vertices,
                                           num_variables, k);

/*
		printf("Message at node\n");
		print_node(graph, i);
		printf("[\t");
		for(j = 0; j < num_variables; ++j){
			printf("%.6lf\t", message_buffer[j]);
		}
		printf("\t]\n");*/


                    //send belief
                    send_message_for_node(src_node_to_edges_nodes, src_node_to_edges_edges, &belief_buffer, num_edges, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y,
                                          curr_messages, messages_previous, messages_current, num_vertices, k);

                }

#pragma acc kernels
                for (k = 0; k < num_vertices; ++k) {
                    marginalize_page_rank_nodes_acc(node_states, node_states_size, k, curr_messages, dest_node_to_edges_nodes, dest_node_to_edges_edges, num_vertices,
                                         num_edges);
                }
            }


            delta = 0.0f;
#pragma acc kernels
            for (j = 0; j < num_edges; ++j) {
                diff = messages_previous[j] - messages_current[j];
                if (diff != diff) {
                    diff = 0.0f;
                }
                delta += fabsf(diff);
            }
        }
        if(delta < convergence || fabsf(delta - previous_delta) < convergence){
            break;
        }
        previous_delta = delta;
        num_iter += BATCH_SIZE;
    }
    if(i == max_iterations) {
        printf("No Convergence: previous: %f vs current: %f\n", previous_delta, delta);
    }


    return num_iter;
}

/**
 * Runs Viterbi for OpenACC
 * @param num_vertices The number of vertices (nodes) in the graph
 * @param num_edges The number of edges in the graph
 * @param dest_node_to_edges_nodes Parallel array; maps destination nodes to their edges; first half; maps nodes to their starting indices in dest_node_to_edges_edges
 * @param dest_node_to_edges_edges Parallel array; maps destination nodes to their edges; second half; lists edges by nodes
 * @param src_node_to_edges_nodes Parallel array; maps source nodes to their edges; first half; maps nodes to their starting indices in src_node_to_edges_edges
 * @param src_node_to_edges_edges Parallel array; maps source nodes to their edges; second half; lists edges by nodes
 * @param node_states The current node states
 * @param previous_messages The previous messages in the graph
 * @param current_messages The current messages in the graph
 * @param joint_probabilities The joint probability tables of the edges
 * @param max_iterations The maximum number of iterations to run for
 * @param convergence The convergence threshold
 * @return The actual number of iterations used
 */
static int viterbi_iterations_acc(int num_vertices, int num_edges,
                                             const int * __restrict__ dest_node_to_edges_nodes,
                                             const int * __restrict__ dest_node_to_edges_edges,
                                             const int * __restrict__ src_node_to_edges_nodes,
                                             const int * __restrict__ src_node_to_edges_edges,
                                             struct belief *node_states,
                                             		const int * __restrict__ node_states_size,
                                             		struct belief *current_messages,
                                             		float * __restrict__ messages_previous,
                                             		float * __restrict__ messages_current,
                                             struct joint_probability *edge_joint_probability,
                                             const int edge_joint_probability_dim_x,
                                             const int edge_joint_probability_dim_y,
                                             int max_iterations,
                                             float convergence){
    int j, k;
    int i, num_variables, num_iter;
    float delta, previous_delta, diff, sum;
    struct belief *curr_messages;

    curr_messages = current_messages;


    struct belief belief_buffer;

    num_iter = 0;

    previous_delta = -1.0f;
    delta = 0.0f;

    for(i = 0; i < max_iterations; i+= BATCH_SIZE) {
#pragma acc data present_or_copy(node_states[0:(num_vertices)], curr_messages[0:(num_edges)], messages_previous[0:(num_edges)], messages_current[0:(num_edges)]) present_or_copyin(dest_node_to_edges_nodes[0:num_vertices], dest_node_to_edges_edges[0:num_edges], src_node_to_edges_nodes[0:num_vertices], src_node_to_edges_edges[0:num_edges], edge_joint_probability[0:1], node_states_size[0:(num_vertices)])
        {
            //printf("Current iteration: %d\n", i+1);
            for (j = 0; j < BATCH_SIZE; ++j) {
#pragma acc kernels
                for (k = 0; k < num_vertices; ++k) {
                    num_variables = node_states_size[k];

                    initialize_message_buffer(&belief_buffer, node_states, k, num_variables);

                    //read incoming messages
                    read_incoming_messages(&belief_buffer, dest_node_to_edges_nodes, dest_node_to_edges_edges, curr_messages, num_edges, num_vertices,
                                           num_variables, k);

/*
		printf("Message at node\n");
		print_node(graph, i);
		printf("[\t");
		for(j = 0; j < num_variables; ++j){
			printf("%.6lf\t", message_buffer[j]);
		}
		printf("\t]\n");*/


                    //send belief
                    send_message_for_node(src_node_to_edges_nodes, src_node_to_edges_edges, &belief_buffer, num_edges, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y,
                                          curr_messages, messages_previous, messages_current, num_vertices, k);

                }

#pragma acc kernels
                for (k = 0; k < num_vertices; ++k) {
                    argmax_node_acc(node_states, node_states_size, k, curr_messages, dest_node_to_edges_nodes, dest_node_to_edges_edges, num_vertices,
                                                    num_edges);
                }
            }


            delta = 0.0f;
#pragma acc kernels
            for (j = 0; j < num_edges; ++j) {
                diff = messages_previous[j] - messages_current[j];
                if (diff != diff) {
                    diff = 0.0f;
                }
                delta += fabsf(diff);
            }
        }
        if(delta < convergence || fabsf(delta - previous_delta) < convergence){
            break;
        }
        previous_delta = delta;
        num_iter += BATCH_SIZE;
    }
    if(i == max_iterations) {
        printf("No Convergence: previous: %f vs current: %f\n", previous_delta, delta);
    }

    #pragma acc data present_or_copy(node_states[0:(num_vertices)], node_states_size[0:(num_vertices)])
    {
        #pragma acc kernels
        for(j = 0; j < num_vertices; ++j){
            sum = 0.0;
            num_variables = node_states_size[j];
            for (k = 0; k < num_variables; ++k) {
                sum += node_states[j].data[k];
            }
            if (sum <= 0.0) {
                sum = 1.0;
            }

            for (k = 0; k < num_variables; ++k) {
                node_states[j].data[k] /= sum;
            }
        }
    }

    return num_iter;
}

/**
 * Runs loopy BP for OpenACC until convergence or max_iterations met
 * @param graph The graph
 * @param convergence The convergence threshold
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations
 */
int loopy_propagate_until_acc(Graph_t graph, float convergence, int max_iterations){
	int iter;

	init_work_queue_nodes(graph);

	/*printf("===BEFORE====\n");
	print_nodes(graph);
	print_edges(graph);
*/
	iter = loopy_propagate_iterations_acc(graph->current_num_vertices, graph->current_num_edges,
	graph->dest_nodes_to_edges_node_list, graph->dest_nodes_to_edges_edge_list,
										  graph->src_nodes_to_edges_node_list, graph->src_nodes_to_edges_edge_list,
	graph->node_states, graph->node_states_size, graph->node_states_previous, graph->node_states_current,
	graph->edges_messages, graph->edges_messages_previous, graph->edges_messages_current, &(graph->edge_joint_probability),
										  graph->edge_joint_probability_dim_x, graph->edge_joint_probability_dim_y,
										  graph->work_queue_nodes, graph->work_queue_scratch,
										  graph->num_work_items_nodes,
										  max_iterations, convergence);

	/*printf("===AFTER====\n");
	print_nodes(graph);
	print_edges(graph);*/

	return iter;
}

/**
 * Runs PageRank for OpenACC until convergence or max_iterations met
 * @param graph The graph
 * @param convergence The convergence threshold
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations
 */
int page_rank_until_acc(Graph_t graph, float convergence, int max_iterations){
    int iter;

    /*printf("===BEFORE====\n");
    print_nodes(graph);
    print_edges(graph);
*/
    iter = page_rank_iterations_acc(graph->current_num_vertices, graph->current_num_edges,
                                          graph->dest_nodes_to_edges_node_list, graph->dest_nodes_to_edges_edge_list,
                                          graph->src_nodes_to_edges_node_list, graph->src_nodes_to_edges_edge_list,
                                          graph->node_states, graph->node_states_size,
                                          graph->edges_messages, graph->edges_messages_previous, graph->edges_messages_current,
                                          &(graph->edge_joint_probability), graph->edge_joint_probability_dim_x, graph->edge_joint_probability_dim_y,
                                          max_iterations, convergence);

    /*printf("===AFTER====\n");
    print_nodes(graph);
    print_edges(graph);*/

    return iter;
}

/**
 * Runs Viterbi for OpenACC until convergence or max_iterations met
 * @param graph The graph
 * @param convergence The convergence threshold
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations
 */
int viterbi_until_acc(Graph_t graph, float convergence, int max_iterations){
    int iter;

    /*printf("===BEFORE====\n");
    print_nodes(graph);
    print_edges(graph);
*/
    iter = viterbi_iterations_acc(graph->current_num_vertices, graph->current_num_edges,
                                    graph->dest_nodes_to_edges_node_list, graph->dest_nodes_to_edges_edge_list,
                                    graph->src_nodes_to_edges_node_list, graph->src_nodes_to_edges_edge_list,
                                    graph->node_states, graph->node_states_size,
                                    graph->edges_messages, graph->edges_messages_previous, graph->edges_messages_current,
                                    &(graph->edge_joint_probability), graph->edge_joint_probability_dim_x, graph->edge_joint_probability_dim_y,
                                    max_iterations, convergence);

    /*printf("===AFTER====\n");
    print_nodes(graph);
    print_edges(graph);*/

    return iter;
}

static void update_work_queue_edges_acc(int * __restrict__ num_work_items_edges, int * __restrict__ work_queue_edges,
										int * __restrict__ work_queue_scratch,
										const float * __restrict__ previous_state, const float * __restrict__ current_state,
										int num_edges,
										float convergence) {
	int i, current_index;

	current_index = 0;

#pragma omp parallel for default(none) shared(current_index, num_work_items_edges, work_queue_scratch, convergence, work_queue_edges, current_state, previous_state) private(i)
#pragma acc kernels copyin(work_queue_edges[0:num_edges], previous_state[0:num_edges], current_state[0:num_edges]) copyout(work_queue_scratch[0:num_edges])
	for(i = 0; i < *num_work_items_edges; ++i) {
		if(fabs(current_state[work_queue_edges[i]] - previous_state[work_queue_edges[i]]) >= convergence) {
#pragma omp critical
#pragma acc atomic capture
			{
				work_queue_scratch[current_index] = work_queue_edges[i];
				current_index++;
			}
		}
	}
	memcpy(work_queue_edges, work_queue_scratch, (size_t)num_edges);
	*num_work_items_edges = current_index;
}


/**
 * Runs edge-optimized loopy BP
 * @param num_vertices The number of vertices in the graph
 * @param num_edges The number of edges in the graph
 * @param node_states The current beliefs (states) of the nodes
 * @param previous_edge_messages The previous buffered edges in the graph
 * @param current_edge_messages The current buffered edges in the graph
 * @param joint_probabilities The edges' joint probability tables
 * @param edges_src_index The indices of the edges' source nodes
 * @param edges_dest_index The indices of the edges' destination nodes
 * @param dest_node_to_edges_node_list Parallel array; maps destination nodes to their edges; first half; maps nodes to their starting index in dest_node_to_edges_edge_list
 * @param dest_node_to_edges_edge_list Parallel array; maps destination nodes to their edges; second half; lists edges by nodes
 * @param max_iterations The maximum number of iterations to run for
 * @param convergence The convergence threshold
 * @return The actual number of iterations executed
 */
static int loopy_propagate_iterations_edges_acc(int num_vertices, int num_edges,
														 struct belief * __restrict__ node_states,
														 int * __restrict__ node_states_size,
														 struct belief * __restrict__ current_edge_messages,
														 float * __restrict__ edges_messages_previous,
														 float * __restrict__ edges_messages_current,
														 const struct joint_probability * __restrict__ edge_joint_probability,
														 const int edge_joint_probability_dim_x,
														 const int edge_joint_probability_dim_y,
														 const int * __restrict__ edges_src_index,
														 const int * __restrict__ edges_dest_index,
														 const int * __restrict__ dest_node_to_edges_node_list,
														 const int * __restrict__ dest_node_to_edges_edge_list,
														 int max_iterations, float convergence){
	int j, k;
    int i, num_iter, src_node_index, dest_node_index;
	float delta, previous_delta, diff;
	struct belief *curr_messages;

	curr_messages = current_edge_messages;


	num_iter = 0;

	previous_delta = -1.0f;
	delta = 0.0f;

	for(i = 0; i < max_iterations; i+= BATCH_SIZE) {
#pragma acc data present_or_copy(node_states[0:(num_vertices)], curr_messages[0:(num_edges)], edges_messages_previous[0:(num_edges)], edges_messages_current[0:(num_edges)]) present_or_copyin(dest_node_to_edges_node_list[0:num_vertices], dest_node_to_edges_edge_list[0:num_edges], edge_joint_probability[0:1], edges_src_index[0:(num_edges)], node_states_size[0:(num_vertices)])
		{
			//printf("Current iteration: %d\n", i+1);
			for (j = 0; j < BATCH_SIZE; ++j) {
#pragma acc kernels
				for(k = 0; k < num_edges; ++k){
					src_node_index = edges_src_index[k];
					send_message_for_edge_iteration(node_states, src_node_index, k, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y, curr_messages, edges_messages_previous, edges_messages_current);
				}

#pragma acc kernels
				for(k = 0; k < num_edges; ++k){
					dest_node_index = edges_dest_index[k];
					combine_loopy_edge(i, curr_messages, dest_node_index, node_states, node_states_size);
				}
#pragma acc kernels
                /*
				for(k = 0; k < num_vertices; ++k){
					marginalize_loopy_node_edge(node_states, num_vars[k]);
				}*/
				for (k = 0; k < num_vertices; ++k) {
					marginalize_node_acc(node_states, node_states_size, k, curr_messages, dest_node_to_edges_node_list, dest_node_to_edges_edge_list, num_vertices,
										 num_edges);
				}
			}


			delta = 0.0f;
#pragma acc kernels
			for (j = 0; j < num_edges; ++j) {
                diff = edges_messages_previous[j] - edges_messages_current[j];
                if (diff != diff) {
                    diff = 0.0f;
                }
                delta += fabsf(diff);
			}
		}
		if(delta < convergence || fabsf(delta - previous_delta) < convergence){
			break;
		}
		previous_delta = delta;
		num_iter += BATCH_SIZE;
	}
	if(i == max_iterations) {
		printf("No Convergence: previous: %f vs current: %f\n", previous_delta, delta);
	}


	return num_iter;
}


/**
 * Runs edge-optimized PageRank
 * @param num_vertices The number of vertices in the graph
 * @param num_edges The number of edges in the graph
 * @param node_states The current beliefs (states) of the nodes
 * @param previous_edge_messages The previous buffered edges in the graph
 * @param current_edge_messages The current buffered edges in the graph
 * @param joint_probabilities The edges' joint probability tables
 * @param edges_src_index The indices of the edges' source nodes
 * @param edges_dest_index The indices of the edges' destination nodes
 * @param dest_node_to_edges_node_list Parallel array; maps destination nodes to their edges; first half; maps nodes to their starting index in dest_node_to_edges_edge_list
 * @param dest_node_to_edges_edge_list Parallel array; maps destination nodes to their edges; second half; lists edges by nodes
 * @param max_iterations The maximum number of iterations to run for
 * @param convergence The convergence threshold
 * @return The actual number of iterations executed
 */
static int page_rank_iterations_edges_acc(int num_vertices, int num_edges,
                                                         struct belief * __restrict__ node_states,
                                                         const int * __restrict__ node_states_size,
                                                         struct belief * __restrict__ current_edge_messages,
                                                         float * __restrict__ edges_messages_previous,
                                                         float * __restrict__ edges_messages_current,
                                                         const struct joint_probability * __restrict__ edge_joint_probability,
                                                         const int edge_joint_probability_dim_x,
                                                         const int edge_joint_probability_dim_y,
                                                         const int * __restrict__ edges_src_index,
                                                         const int * __restrict__ edges_dest_index,
                                                         const int * __restrict__ dest_node_to_edges_node_list,
                                                         const int * __restrict__ dest_node_to_edges_edge_list,
                                                         int max_iterations, float convergence){
    int j, k;
    int i, num_iter, src_node_index, dest_node_index;
    float delta, previous_delta, diff;
    struct belief *curr_messages;

    curr_messages = current_edge_messages;


    num_iter = 0;

    previous_delta = -1.0f;
    delta = 0.0f;

    for(i = 0; i < max_iterations; i+= BATCH_SIZE) {
#pragma acc data present_or_copy(node_states[0:(num_vertices)], curr_messages[0:(num_edges)], edges_messages_previous[0:(num_edges)], edges_messages_current[0:(num_edges)]) present_or_copyin(dest_node_to_edges_node_list[0:num_vertices], dest_node_to_edges_edge_list[0:num_edges], edge_joint_probability[0:1], edges_src_index[0:num_edges], node_states_size[0:(num_vertices)])
        {
            //printf("Current iteration: %d\n", i+1);
            for (j = 0; j < BATCH_SIZE; ++j) {
#pragma acc kernels
                for(k = 0; k < num_edges; ++k){
                    src_node_index = edges_src_index[k];
                    send_message_for_edge_iteration(node_states, src_node_index, k, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y, curr_messages, edges_messages_previous, edges_messages_current);
                }

#pragma acc kernels
                for(k = 0; k < num_edges; ++k){
                    dest_node_index = edges_dest_index[k];
                    combine_loopy_edge(i, curr_messages, dest_node_index, node_states, node_states_size);
                }
#pragma acc kernels
                /*
				for(k = 0; k < num_vertices; ++k){
					marginalize_loopy_node_edge(node_states, num_vars[k]);
				}*/
                for (k = 0; k < num_vertices; ++k) {
                    marginalize_page_rank_nodes_acc(node_states, node_states_size, k, curr_messages, dest_node_to_edges_node_list, dest_node_to_edges_edge_list, num_vertices,
                                         num_edges);
                }
            }


            delta = 0.0f;
#pragma acc kernels
            for (j = 0; j < num_edges; ++j) {
                diff = edges_messages_previous[j] - edges_messages_current[j];
                if (diff != diff) {
                    diff = 0.0f;
                }
                delta += fabsf(diff);
            }
        }
        if(delta < convergence || fabsf(delta - previous_delta) < convergence){
            break;
        }
        previous_delta = delta;
        num_iter += BATCH_SIZE;
    }
    if(i == max_iterations) {
        printf("No Convergence: previous: %f vs current: %f\n", previous_delta, delta);
    }


    return num_iter;
}

/**
 * Runs edge-optimized Viterbi
 * @param num_vertices The number of vertices in the graph
 * @param num_edges The number of edges in the graph
 * @param node_states The current beliefs (states) of the nodes
 * @param previous_edge_messages The previous buffered edges in the graph
 * @param current_edge_messages The current buffered edges in the graph
 * @param joint_probabilities The edges' joint probability tables
 * @param edges_src_index The indices of the edges' source nodes
 * @param edges_dest_index The indices of the edges' destination nodes
 * @param dest_node_to_edges_node_list Parallel array; maps destination nodes to their edges; first half; maps nodes to their starting index in dest_node_to_edges_edge_list
 * @param dest_node_to_edges_edge_list Parallel array; maps destination nodes to their edges; second half; lists edges by nodes
 * @param max_iterations The maximum number of iterations to run for
 * @param convergence The convergence threshold
 * @return The actual number of iterations executed
 */
static int viterbi_iterations_edges_acc(int num_vertices, int num_edges,
                                                   struct belief * __restrict__ node_states,
                                                   const int * __restrict__ node_states_size,
                                                   struct belief * __restrict__ current_edge_messages,
                                                   float * __restrict__ edges_messages_previous,
                                                   float * __restrict__ edges_messages_current,
                                                   const struct joint_probability * __restrict__ edge_joint_probability,
                                                   const int edge_joint_probability_dim_x,
                                                   const int edge_joint_probability_dim_y,
                                                   const int * __restrict__ edges_src_index,
                                                   const int * __restrict__ edges_dest_index,
                                                   const int * __restrict__ dest_node_to_edges_node_list,
                                                   const int * __restrict__ dest_node_to_edges_edge_list,
                                                   int max_iterations, float convergence){
    int j, k;
    int i, num_iter, src_node_index, dest_node_index, num_variables;
    float delta, previous_delta, diff, sum;
    struct belief *curr_messages;

    curr_messages = current_edge_messages;


    num_iter = 0;

    previous_delta = -1.0f;
    delta = 0.0f;

    for(i = 0; i < max_iterations; i+= BATCH_SIZE) {
#pragma acc data present_or_copy(node_states[0:(num_vertices)], curr_messages[0:(num_edges)], edges_messages_previous[0:(num_edges)], edges_messages_current[0:(num_edges)]) present_or_copyin(dest_node_to_edges_node_list[0:num_vertices], dest_node_to_edges_edge_list[0:num_edges], edge_joint_probability, edges_src_index[0:(num_edges)], node_states_size[0:(num_vertices)])
        {
            //printf("Current iteration: %d\n", i+1);
            for (j = 0; j < BATCH_SIZE; ++j) {
#pragma acc kernels
                for(k = 0; k < num_edges; ++k){
                    src_node_index = edges_src_index[k];
                    send_message_for_edge_iteration(node_states, src_node_index, k, edge_joint_probability, edge_joint_probability_dim_x, edge_joint_probability_dim_y, curr_messages, edges_messages_previous, edges_messages_current);
                }

#pragma acc kernels
                for(k = 0; k < num_edges; ++k){
                    dest_node_index = edges_dest_index[k];
                    combine_loopy_edge(i, curr_messages, dest_node_index, node_states, node_states_size);
                }
#pragma acc kernels
                /*
				for(k = 0; k < num_vertices; ++k){
					marginalize_loopy_node_edge(node_states, num_vars[k]);
				}*/
                for (k = 0; k < num_vertices; ++k) {
                    argmax_node_acc(node_states, node_states_size, k, curr_messages, dest_node_to_edges_node_list, dest_node_to_edges_edge_list, num_vertices,
                                                    num_edges);
                }
            }


            delta = 0.0f;
#pragma acc kernels
            for (j = 0; j < num_edges; ++j) {
                diff = edges_messages_previous[j] - edges_messages_current[j];
                if (diff != diff) {
                    diff = 0.0f;
                }
                delta += fabsf(diff);
            }
        }
        if(delta < convergence || fabsf(delta - previous_delta) < convergence){
            break;
        }
        previous_delta = delta;
        num_iter += BATCH_SIZE;
    }
    if(i == max_iterations) {
        printf("No Convergence: previous: %f vs current: %f\n", previous_delta, delta);
    }

    #pragma acc data present_or_copy(node_states[0:(num_vertices)], node_states_size[0:(num_vertices)])
    {
        #pragma acc kernels
        for(j = 0; j < num_vertices; ++j){
            sum = 0.0;
            num_variables = node_states_size[j];
            for (k = 0; k < num_variables; ++k) {
                sum += node_states[j].data[k];
            }
            if (sum <= 0.0) {
                sum = 1.0;
            }

            for (k = 0; k < num_variables; ++k) {
                node_states[j].data[k] /= sum;
            }
        }
    }


    return num_iter;
}


/**
 * Runs edge-optimized loopy BP until convergence or max iterations executed
 * @param graph The graph
 * @param convergence The convergence threshold below which execution will halt
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations executed
 */
int loopy_propagate_until_edge_acc(Graph_t graph, float convergence, int max_iterations){
    int iter;

    /*printf("===BEFORE====\n");
    print_nodes(graph);
    print_edges(graph);
*/
    iter = loopy_propagate_iterations_edges_acc(graph->current_num_vertices, graph->current_num_edges,
    graph->node_states,
    graph->node_states_size,
    graph->edges_messages,
    graph->edges_messages_previous,
    graph->edges_messages_current,
    &(graph->edge_joint_probability),
    graph->edge_joint_probability_dim_x,
    graph->edge_joint_probability_dim_y,
    graph->edges_src_index, graph->edges_dest_index,
	graph->dest_nodes_to_edges_node_list, graph->dest_nodes_to_edges_edge_list,
    max_iterations, convergence);

    /*printf("===AFTER====\n");
    print_nodes(graph);
    print_edges(graph);*/

    return iter;
}


/**
 * Runs edge-optimized PageRank until convergence or max iterations executed
 * @param graph The graph
 * @param convergence The convergence threshold below which execution will halt
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations executed
 */
int page_rank_until_edge_acc(Graph_t graph, float convergence, int max_iterations){
    int iter;

    /*printf("===BEFORE====\n");
    print_nodes(graph);
    print_edges(graph);
*/
    iter = page_rank_iterations_edges_acc(graph->current_num_vertices, graph->current_num_edges,
                                                graph->node_states, graph->node_states_size,
                                                graph->edges_messages, graph->edges_messages_previous, graph->edges_messages_current,
                                                &(graph->edge_joint_probability),
                                                graph->edge_joint_probability_dim_x,
                                                graph->edge_joint_probability_dim_y,
                                                graph->edges_src_index, graph->edges_dest_index,
                                                graph->dest_nodes_to_edges_node_list, graph->dest_nodes_to_edges_edge_list,
                                                max_iterations, convergence);

    /*printf("===AFTER====\n");
    print_nodes(graph);
    print_edges(graph);*/

    return iter;
}

/**
 * Runs edge-optimized Viterbi until convergence or max iterations executed
 * @param graph The graph
 * @param convergence The convergence threshold below which execution will halt
 * @param max_iterations The maximum number of iterations to run for
 * @return The actual number of iterations executed
 */
int viterbi_until_edge_acc(Graph_t graph, float convergence, int max_iterations){
    int iter;

    /*printf("===BEFORE====\n");
    print_nodes(graph);
    print_edges(graph);
*/
    iter = page_rank_iterations_edges_acc(graph->current_num_vertices, graph->current_num_edges,
                                          graph->node_states, graph->node_states_size,
                                          graph->edges_messages, graph->edges_messages_previous, graph->edges_messages_current,
                                          &(graph->edge_joint_probability), graph->edge_joint_probability_dim_x, graph->edge_joint_probability_dim_y,
                                          graph->edges_src_index, graph->edges_dest_index,
                                          graph->dest_nodes_to_edges_node_list, graph->dest_nodes_to_edges_edge_list,
                                          max_iterations, convergence);

    /*printf("===AFTER====\n");
    print_nodes(graph);
    print_edges(graph);*/

    return iter;
}

/**
 * Calculates the diameter of the graph using Floyd-Warshall
 * @param graph The graph
 */
void calculate_diameter(Graph_t graph){
    // calculate diameter using floyd-warshall
    int ** dist;
	int ** g;
    int i, j, k, start_index, end_index;
	int curr_dist;

	dist = (int **)malloc(sizeof(int *) * graph->current_num_vertices);
	assert(dist);
	g = (int **)malloc(sizeof(int *) * graph->current_num_vertices);
	assert(g);
	for(i = 0; i < graph->current_num_vertices; ++i){
		dist[i] = (int *)malloc(sizeof(int) * graph->current_num_vertices);
		assert(dist[i]);
		g[i] = (int *)malloc(sizeof(int) * graph->current_num_vertices);
		assert(g[i]);
	}

	// fill in g based on edges
	for(i = 0; i < graph->current_num_vertices; ++i){
		for(j = 0; j < graph->current_num_vertices; ++j){
			g[i][j] = WEIGHT_INFINITY;
		}
	}
	for(i = 0; i < graph->current_num_vertices; ++i){
		 start_index = graph->src_nodes_to_edges_node_list[i];
		if(i + 1 == graph->current_num_vertices){
			end_index = graph->current_num_edges;
		}
		else{
			end_index = graph->src_nodes_to_edges_node_list[i+1];
		}
		for(j = start_index; j < end_index; ++j){
			k = graph->src_nodes_to_edges_edge_list[j];
			g[i][graph->edges_dest_index[k]] = 1;
		}
	}

	for(i = 0; i < graph->current_num_vertices; ++i){
		for(j = 0; j < graph->current_num_vertices; ++j){
			dist[i][j] = g[i][j];
		}
	}

	for(k = 0; k < graph->current_num_vertices; ++k){
		#pragma omp parallel for shared(dist, graph) private(curr_dist, i, j)
		for(i = 0; i < graph->current_num_vertices; ++i){
			for(j = 0; j < graph->current_num_vertices; ++j){
				curr_dist = dist[i][k] + dist[k][j];
				if(curr_dist < dist[i][j]){
					dist[i][j] = curr_dist;
				}
			}
		}
	}

	graph->diameter = -1;

	for(i = 0; i < graph->current_num_vertices; ++i){
		for(j = 0; j < graph->current_num_vertices; ++j){
			if(dist[i][j] != WEIGHT_INFINITY && dist[i][j] > graph->diameter){
				graph->diameter = dist[i][j];
			}
		}
		free(g[i]);
		free(dist[i]);
	}
	free(g);
	free(dist);
}

void prep_as_page_rank(Graph_t g){
    //joint probability is actually out-degree
    struct belief * node_beliefs;
    int * node_beliefs_size;
    int num_vertices;
    int i;

    node_beliefs = g->node_states;
    node_beliefs_size = g->node_states_size;

    num_vertices = g->current_num_vertices;

    for(i = 0; i < num_vertices; ++i) {
        // fix beliefs
        node_beliefs[i].data[0] = 1.0f;
        node_beliefs_size[i] = 1;
    }
}

void init_work_queue_nodes(Graph_t graph) {
	int i, num_work_items_nodes;
	int *work_queue_nodes;

	graph->num_work_items_nodes = graph->current_num_vertices;
	assert(graph->work_queue_nodes == NULL);
	graph->work_queue_nodes = (int*)malloc(sizeof(int) * graph->num_work_items_nodes);
	assert(graph->work_queue_nodes);

	assert(graph->work_queue_scratch == NULL);
	graph->work_queue_scratch = (int*)malloc(sizeof(int) * graph->current_num_vertices);
	assert(graph->work_queue_scratch);

	num_work_items_nodes = graph->num_work_items_nodes;
	work_queue_nodes = graph->work_queue_nodes;

#pragma omp parallel for default(none) shared(num_work_items_nodes, work_queue_nodes) private(i)
#pragma acc parallel private(i)
	for(i = 0; i < num_work_items_nodes; ++i) {
		work_queue_nodes[i] = i;
	}
}

void init_work_queue_edges(Graph_t graph) {
	int i, num_work_item_edges;
	int *work_queue_edges;

	graph->num_work_items_edges = graph->current_num_edges;
	assert(graph->work_queue_edges == NULL);
	graph->work_queue_edges = (int*)malloc(sizeof(int) * graph->num_work_items_edges);
	assert(graph->work_queue_edges);

	assert(graph->work_queue_scratch == NULL);
	graph->work_queue_scratch = (int*)malloc(sizeof(int) * graph->current_num_edges);
	assert(graph->work_queue_scratch);

	work_queue_edges = graph->work_queue_edges;
	num_work_item_edges = graph->num_work_items_edges;

#pragma omp parallel for default(none) shared(num_work_item_edges, work_queue_edges) private(i)
#pragma acc parallel private(i)
	for(i = 0; i < num_work_item_edges; ++i) {
        work_queue_edges[i] = i;
	}
}

void update_work_queue_nodes(Graph_t graph, float convergence) {
	int current_index, i, num_work_items_nodes, num_vertices;
	int *work_queue_nodes, *work_queue_scratch;

	float *current_states;
	float *previous_states;
	double diff = 0;

	current_index = 0;
	num_work_items_nodes = graph->num_work_items_nodes;
	work_queue_nodes = graph->work_queue_nodes;
	work_queue_scratch = graph->work_queue_scratch;
	current_states = graph->node_states_current;
	previous_states = graph->node_states_previous;
    num_vertices = graph->current_num_vertices;

#pragma omp parallel for default(none) shared(current_index, num_work_items_nodes, work_queue_scratch, convergence, work_queue_nodes, current_states, previous_states) private(i)
#pragma acc parallel private(i) copyin(work_queue_nodes[0:num_vertices], current_states[0:num_vertices], previous_states[0:num_vertices]) copyout(work_queue_scratch[0:num_vertices])
    for(i = 0; i < num_work_items_nodes; ++i) {
		if(fabs(current_states[work_queue_nodes[i]] - previous_states[work_queue_nodes[i]]) >= convergence) {
			#pragma omp critical
#pragma acc atomic capture
            {
                work_queue_scratch[current_index] = work_queue_nodes[i];
                current_index++;
            }
		}
	}
	memcpy(work_queue_nodes, work_queue_scratch, (size_t)num_vertices);
	graph->num_work_items_nodes = current_index;
}

void update_work_queue_edges(Graph_t graph, float convergence) {
	int current_index, i, num_work_items_edges, num_edges;
	int *work_queue_edges, *work_queue_scratch;
	float *previous_states;
	float *current_states;

	current_index = 0;
	num_work_items_edges = graph->num_work_items_edges;
	work_queue_edges = graph->work_queue_edges;
	work_queue_scratch = graph->work_queue_scratch;
	current_states = graph->edges_messages_current;
	previous_states = graph->edges_messages_previous;

    num_edges = graph->current_num_edges;

#pragma omp parallel for default(none) shared(current_index, num_work_items_edges, work_queue_scratch, convergence, work_queue_edges, current_states, previous_states) private(i)
#pragma acc parallel private(i) copyin(work_queue_edges[0:num_edges], current_states[0:num_edges], previous_states[0:num_edges]) copyout(work_queue_scratch[0:num_edges])
    for(i = 0; i < num_work_items_edges; ++i) {
		if(fabs(current_states[work_queue_edges[i]] - previous_states[work_queue_edges[i]]) >= convergence) {
#pragma omp critical
#pragma acc atomic capture
			{
				work_queue_scratch[current_index] = work_queue_edges[i];
				current_index++;
			}
		}
	}
	memcpy(work_queue_edges, work_queue_scratch, (size_t)num_edges);
	graph->num_work_items_edges = current_index;
}

float difference(struct belief *a, int a_size, struct belief *b, int b_size) {
    float diff = 0.0f;
    if(a_size > MAX_STATES || b_size > MAX_STATES) {
        return diff;
    }
    for(int i = 0; i < a_size && i < b_size; ++i) {
        diff += fabsf(a->data[i] - b->data[i]);
    }
    return diff;
}

void set_joint_probability_yahoo_web(struct joint_probability * edge_joint_probability, int * dim_x, int * dim_y) {
	assert(MAX_STATES >= 2);
	assert(edge_joint_probability);
	assert(dim_x);
	assert(dim_y);
	*dim_x = 2;
	*dim_y = 2;
	edge_joint_probability->data[0][0] = 0.95f;
	edge_joint_probability->data[0][1] = 0.05f;
	edge_joint_probability->data[1][0] = 0.5f;
	edge_joint_probability->data[1][1] = 0.5f;
}

void set_joint_probability_twitter(struct joint_probability * edge_joint_probability, int * dim_x, int * dim_y) {
	assert(MAX_STATES >= 3);
	assert(edge_joint_probability);
	assert(dim_x);
	assert(dim_y);
	*dim_x = 3;
	*dim_y = 3;
	edge_joint_probability->data[0][0] = 0.1f;
	edge_joint_probability->data[0][1] = 0.05f;
	edge_joint_probability->data[0][2] = 0.85f;
	edge_joint_probability->data[1][0] = 0.1f;
	edge_joint_probability->data[1][1] = 0.45f;
	edge_joint_probability->data[1][2] = 0.45f;
	edge_joint_probability->data[2][0] = 0.35f;
	edge_joint_probability->data[2][1] = 0.05f;
	edge_joint_probability->data[2][2] = 0.6f;
}

void set_joint_probability_vc(struct joint_probability * edge_joint_probability, int * dim_x, int * dim_y) {
	assert(MAX_STATES >= 3);
	assert(edge_joint_probability);
	assert(dim_x);
	assert(dim_y);
	*dim_x = 3;
	*dim_y = 3;
	edge_joint_probability->data[0][0] = 0.266666667f;
	edge_joint_probability->data[0][1] = -0.033333333f;
	edge_joint_probability->data[0][2] = 0.366666667f;
	edge_joint_probability->data[1][0] = 0.033333333f;
	edge_joint_probability->data[1][1] = -0.333333333f;
	edge_joint_probability->data[1][2] = 0.366666667f;
	edge_joint_probability->data[2][0] = -0.233333333f;
	edge_joint_probability->data[2][1] = 0.366666667f;
	edge_joint_probability->data[2][2] = -0.133333333f;
}