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
        TAILQ_INIT(&(src_entry->indices));
        src_entry->count = 0;
    }
    else{
        src_entry = (struct htable_entry *)src_ep->data;
    }
    //printf("Src Index: %d\n", src_index);
    assert(src_entry != NULL);
    /*
    src_entry->indices[src_entry->count] = edge_index;
    src_entry->count += 1;*/
    add_index(src_entry, edge_index);
    src_e.data = src_entry;

	sprintf(dest_key, "%d", dest_index);
    dest_e.key = dest_key;
	dest_e.data = NULL;
    hsearch_r(dest_e, FIND, &dest_ep, graph->dest_node_to_edge_table);
    if(dest_ep == NULL){
        dest_entry = (struct htable_entry *)calloc(sizeof(struct htable_entry), 1);
        TAILQ_INIT(&(dest_entry->indices));
        dest_entry->count = 0;
    }
    else{
        dest_entry = (struct htable_entry *)dest_ep->data;
    }
    assert(dest_entry != NULL);
    /*dest_entry->indices[dest_entry->count] = edge_index;
    dest_entry->count += 1;*/
    add_index(dest_entry, edge_index);
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

void add_index(struct htable_entry * entry, int index) {
    struct htable_index * index_entry = (struct htable_index *)malloc(sizeof(struct htable_entry));
    assert(index_entry);
    index_entry->index = index;
    TAILQ_INSERT_TAIL(&(entry->indices), index_entry, next_index);
    entry->count += 1;
}

void delete_indices(struct htable_entry * entry) {
    struct htable_index *index;
    while((index = TAILQ_FIRST(&(entry->indices)))) {
        TAILQ_REMOVE(&(entry->indices), index, next_index);
        free(index);
    }
}

static void set_up_nodes_to_edges(const int *edges_index, int * nodes_to_edges_nodes_list,
                                  int * nodes_to_edges_edges_list, Graph_t graph){
    int i, j, edge_index, num_vertices, num_edges, current_degree;
    ENTRY entry, *ep;
    struct htable_entry *metadata;
    char *search_key;
	struct hsearch_data *htab;
	struct htable_index *index;

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
			TAILQ_INIT(&(metadata->indices));
        }
        else {
            metadata = ep->data;
        }
        // add current edge to list
        add_index(metadata, j);
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

            assert(metadata->count >= 0);

            TAILQ_FOREACH(index, &(metadata->indices), next_index) {
                nodes_to_edges_edges_list[edge_index] = index->index;
                edge_index += 1;
                current_degree += 1;
            }

            //cleanup
            delete_indices(metadata);
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
                if (src_ep != NULL && src_ep->data != NULL) {
                    struct htable_entry * metadata = (struct htable_entry *)src_ep->data;
                    delete_indices(metadata);
                    free(metadata);
                    src_ep->data = NULL;
                }
            }
            src_ep = NULL;

            sprintf(dest_key, "%d", i);
            dest_e.key = dest_key;
            dest_e.data = NULL;
            if(hsearch_r(dest_e, FIND, &dest_ep, g->dest_node_to_edge_table) != 0) {
                if(dest_ep != NULL && dest_ep->data != NULL){
                    struct htable_entry * metadata = (struct htable_entry *)dest_ep->data;
                    delete_indices(metadata);
                    free(metadata);
                    dest_ep->data = NULL;
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

void set_joint_probability_32(struct joint_probability * edge_joint_probability, int * dim_x, int * dim_y) {
	assert(MAX_STATES >= 32);
	assert(edge_joint_probability);
	assert(dim_x);
	assert(dim_y);
	*dim_x = 32;
	*dim_y = 32;

    // row 0
    edge_joint_probability->data[0][0] = 0.0576182695772f;
    edge_joint_probability->data[0][1] = 0.0195189057242f;
    edge_joint_probability->data[0][2] = 0.0467379035556f;
    edge_joint_probability->data[0][3] = 0.0434962931057f;
    edge_joint_probability->data[0][4] = 5.65677018341e-05f;
    edge_joint_probability->data[0][5] = 0.0527002003068f;
    edge_joint_probability->data[0][6] = 0.0340889699495f;
    edge_joint_probability->data[0][7] = 0.0324737799034f;
    edge_joint_probability->data[0][8] = 0.0589681419368f;
    edge_joint_probability->data[0][9] = 0.011159250271f;
    edge_joint_probability->data[0][10] = 0.0296761524237f;
    edge_joint_probability->data[0][11] = 0.0207752399758f;
    edge_joint_probability->data[0][12] = 0.0471880198311f;
    edge_joint_probability->data[0][13] = 0.0250808435379f;
    edge_joint_probability->data[0][14] = 0.0597305034115f;
    edge_joint_probability->data[0][15] = 0.00807061687886f;
    edge_joint_probability->data[0][16] = 0.00489865484545f;
    edge_joint_probability->data[0][17] = 0.0528476808692f;
    edge_joint_probability->data[0][18] = 0.0096378989193f;
    edge_joint_probability->data[0][19] = 0.0165787558987f;
    edge_joint_probability->data[0][20] = 0.0396624911251f;
    edge_joint_probability->data[0][21] = 0.00402215754707f;
    edge_joint_probability->data[0][22] = 0.0040912841261f;
    edge_joint_probability->data[0][23] = 0.0410141864149f;
    edge_joint_probability->data[0][24] = 0.0576084885092f;
    edge_joint_probability->data[0][25] = 0.0196241423753f;
    edge_joint_probability->data[0][26] = 0.0593653654057f;
    edge_joint_probability->data[0][27] = 0.00471732607566f;
    edge_joint_probability->data[0][28] = 0.00791283607315f;
    edge_joint_probability->data[0][29] = 0.0592117449952f;
    edge_joint_probability->data[0][30] = 0.0463057556999f;
    edge_joint_probability->data[0][31] = 0.0251615730289f;

// row 1
    edge_joint_probability->data[1][0] = 0.0101169637922f;
    edge_joint_probability->data[1][1] = 0.0353198528975f;
    edge_joint_probability->data[1][2] = 0.0166420643722f;
    edge_joint_probability->data[1][3] = 0.0165518640814f;
    edge_joint_probability->data[1][4] = 0.0263969294056f;
    edge_joint_probability->data[1][5] = 0.0352392373276f;
    edge_joint_probability->data[1][6] = 0.0127780628135f;
    edge_joint_probability->data[1][7] = 0.0187152577095f;
    edge_joint_probability->data[1][8] = 0.00568724115761f;
    edge_joint_probability->data[1][9] = 0.0154193432439f;
    edge_joint_probability->data[1][10] = 0.0487050173609f;
    edge_joint_probability->data[1][11] = 0.0405647291963f;
    edge_joint_probability->data[1][12] = 0.0429063022512f;
    edge_joint_probability->data[1][13] = 0.0290844627672f;
    edge_joint_probability->data[1][14] = 0.00865030290901f;
    edge_joint_probability->data[1][15] = 0.0130300252622f;
    edge_joint_probability->data[1][16] = 0.034108844154f;
    edge_joint_probability->data[1][17] = 0.0587117446713f;
    edge_joint_probability->data[1][18] = 0.0562211356875f;
    edge_joint_probability->data[1][19] = 0.00124449534132f;
    edge_joint_probability->data[1][20] = 0.0402817316639f;
    edge_joint_probability->data[1][21] = 0.00268088250725f;
    edge_joint_probability->data[1][22] = 0.044472899564f;
    edge_joint_probability->data[1][23] = 0.054203619817f;
    edge_joint_probability->data[1][24] = 0.0508309682552f;
    edge_joint_probability->data[1][25] = 0.042174723678f;
    edge_joint_probability->data[1][26] = 0.00106541373675f;
    edge_joint_probability->data[1][27] = 0.0522661980784f;
    edge_joint_probability->data[1][28] = 0.0260317939526f;
    edge_joint_probability->data[1][29] = 0.047477680841f;
    edge_joint_probability->data[1][30] = 0.0559548135111f;
    edge_joint_probability->data[1][31] = 0.0564653979928f;

// row 2
    edge_joint_probability->data[2][0] = 0.025654929039f;
    edge_joint_probability->data[2][1] = 0.0278093531647f;
    edge_joint_probability->data[2][2] = 0.0529104905219f;
    edge_joint_probability->data[2][3] = 0.0268283662869f;
    edge_joint_probability->data[2][4] = 0.0221997897064f;
    edge_joint_probability->data[2][5] = 0.00623147764857f;
    edge_joint_probability->data[2][6] = 0.00124058177433f;
    edge_joint_probability->data[2][7] = 0.0466391395484f;
    edge_joint_probability->data[2][8] = 0.0245770397963f;
    edge_joint_probability->data[2][9] = 0.0173249009552f;
    edge_joint_probability->data[2][10] = 0.0393719429286f;
    edge_joint_probability->data[2][11] = 0.0531394665244f;
    edge_joint_probability->data[2][12] = 0.0362740908173f;
    edge_joint_probability->data[2][13] = 0.00387252919234f;
    edge_joint_probability->data[2][14] = 0.0403203987107f;
    edge_joint_probability->data[2][15] = 0.0143406979674f;
    edge_joint_probability->data[2][16] = 0.0474311886442f;
    edge_joint_probability->data[2][17] = 0.055942329993f;
    edge_joint_probability->data[2][18] = 0.0070488340352f;
    edge_joint_probability->data[2][19] = 0.0448485442631f;
    edge_joint_probability->data[2][20] = 0.00771562095345f;
    edge_joint_probability->data[2][21] = 0.0255348087173f;
    edge_joint_probability->data[2][22] = 0.0447399182944f;
    edge_joint_probability->data[2][23] = 0.0122878305496f;
    edge_joint_probability->data[2][24] = 0.0204958260907f;
    edge_joint_probability->data[2][25] = 0.0406391505704f;
    edge_joint_probability->data[2][26] = 0.00701178939361f;
    edge_joint_probability->data[2][27] = 0.0585284914292f;
    edge_joint_probability->data[2][28] = 0.0487227171325f;
    edge_joint_probability->data[2][29] = 0.0545351251562f;
    edge_joint_probability->data[2][30] = 0.0652473549943f;
    edge_joint_probability->data[2][31] = 0.0205352752005f;

// row 3
    edge_joint_probability->data[3][0] = 0.0382826679338f;
    edge_joint_probability->data[3][1] = 0.0474666163501f;
    edge_joint_probability->data[3][2] = 0.00508701821821f;
    edge_joint_probability->data[3][3] = 0.0475700633814f;
    edge_joint_probability->data[3][4] = 0.0448306337338f;
    edge_joint_probability->data[3][5] = 0.00901171010088f;
    edge_joint_probability->data[3][6] = 0.023942651898f;
    edge_joint_probability->data[3][7] = 0.0263581386047f;
    edge_joint_probability->data[3][8] = 0.00743740141014f;
    edge_joint_probability->data[3][9] = 0.0477318741721f;
    edge_joint_probability->data[3][10] = 0.0449084158637f;
    edge_joint_probability->data[3][11] = 0.0221982319845f;
    edge_joint_probability->data[3][12] = 0.011858311319f;
    edge_joint_probability->data[3][13] = 0.0178668275182f;
    edge_joint_probability->data[3][14] = 0.0449469302106f;
    edge_joint_probability->data[3][15] = 0.0182301680633f;
    edge_joint_probability->data[3][16] = 0.0306676338964f;
    edge_joint_probability->data[3][17] = 0.0404881632381f;
    edge_joint_probability->data[3][18] = 0.0201832012853f;
    edge_joint_probability->data[3][19] = 0.0470268874319f;
    edge_joint_probability->data[3][20] = 0.0183105913958f;
    edge_joint_probability->data[3][21] = 0.00252579907849f;
    edge_joint_probability->data[3][22] = 0.0338435633996f;
    edge_joint_probability->data[3][23] = 0.0481789121327f;
    edge_joint_probability->data[3][24] = 0.0263704947799f;
    edge_joint_probability->data[3][25] = 0.052216043913f;
    edge_joint_probability->data[3][26] = 0.0442264060171f;
    edge_joint_probability->data[3][27] = 0.0181626124283f;
    edge_joint_probability->data[3][28] = 0.0180785350125f;
    edge_joint_probability->data[3][29] = 0.0435396124517f;
    edge_joint_probability->data[3][30] = 0.045792280752f;
    edge_joint_probability->data[3][31] = 0.0526616020248f;

// row 4
    edge_joint_probability->data[4][0] = 0.043501568004f;
    edge_joint_probability->data[4][1] = 0.0402443840498f;
    edge_joint_probability->data[4][2] = 0.0533769004686f;
    edge_joint_probability->data[4][3] = 0.064252813549f;
    edge_joint_probability->data[4][4] = 0.0368187774981f;
    edge_joint_probability->data[4][5] = 0.0160231221352f;
    edge_joint_probability->data[4][6] = 0.0692214363952f;
    edge_joint_probability->data[4][7] = 0.0180441823828f;
    edge_joint_probability->data[4][8] = 0.0670322572524f;
    edge_joint_probability->data[4][9] = 0.00871347260104f;
    edge_joint_probability->data[4][10] = 0.031848054693f;
    edge_joint_probability->data[4][11] = 0.0217689208462f;
    edge_joint_probability->data[4][12] = 0.00832198091938f;
    edge_joint_probability->data[4][13] = 0.00919834273179f;
    edge_joint_probability->data[4][14] = 0.0691122809941f;
    edge_joint_probability->data[4][15] = 0.00969269911382f;
    edge_joint_probability->data[4][16] = 0.0294183487903f;
    edge_joint_probability->data[4][17] = 0.0192729269301f;
    edge_joint_probability->data[4][18] = 0.018946680438f;
    edge_joint_probability->data[4][19] = 0.0262378796397f;
    edge_joint_probability->data[4][20] = 0.0583774968604f;
    edge_joint_probability->data[4][21] = 0.0117922018737f;
    edge_joint_probability->data[4][22] = 0.0565808133014f;
    edge_joint_probability->data[4][23] = 0.00163353407991f;
    edge_joint_probability->data[4][24] = 0.00321866452672f;
    edge_joint_probability->data[4][25] = 0.0434377646763f;
    edge_joint_probability->data[4][26] = 0.0376426481422f;
    edge_joint_probability->data[4][27] = 0.000726741832098f;
    edge_joint_probability->data[4][28] = 0.0621436227604f;
    edge_joint_probability->data[4][29] = 0.00416347817308f;
    edge_joint_probability->data[4][30] = 0.000238515144071f;
    edge_joint_probability->data[4][31] = 0.0589974891971f;

// row 5
    edge_joint_probability->data[5][0] = 0.053403002907f;
    edge_joint_probability->data[5][1] = 0.00111396686655f;
    edge_joint_probability->data[5][2] = 0.00520629440767f;
    edge_joint_probability->data[5][3] = 0.0112695258701f;
    edge_joint_probability->data[5][4] = 0.0482905719984f;
    edge_joint_probability->data[5][5] = 0.0534363188329f;
    edge_joint_probability->data[5][6] = 0.0374352491898f;
    edge_joint_probability->data[5][7] = 0.0348947788997f;
    edge_joint_probability->data[5][8] = 0.0174181236886f;
    edge_joint_probability->data[5][9] = 0.0390127446559f;
    edge_joint_probability->data[5][10] = 0.050048248924f;
    edge_joint_probability->data[5][11] = 0.0107117564651f;
    edge_joint_probability->data[5][12] = 0.0354759307364f;
    edge_joint_probability->data[5][13] = 0.0363059322534f;
    edge_joint_probability->data[5][14] = 0.00568554528547f;
    edge_joint_probability->data[5][15] = 0.0294433614373f;
    edge_joint_probability->data[5][16] = 0.0331071231309f;
    edge_joint_probability->data[5][17] = 0.026044949613f;
    edge_joint_probability->data[5][18] = 0.00648974497442f;
    edge_joint_probability->data[5][19] = 0.0387631746171f;
    edge_joint_probability->data[5][20] = 0.0287807377798f;
    edge_joint_probability->data[5][21] = 0.0254347624611f;
    edge_joint_probability->data[5][22] = 0.000531696791045f;
    edge_joint_probability->data[5][23] = 0.0489292168774f;
    edge_joint_probability->data[5][24] = 0.0331903965792f;
    edge_joint_probability->data[5][25] = 0.0151822574546f;
    edge_joint_probability->data[5][26] = 0.0526301799003f;
    edge_joint_probability->data[5][27] = 0.0534257171709f;
    edge_joint_probability->data[5][28] = 0.047439378541f;
    edge_joint_probability->data[5][29] = 0.0488053396093f;
    edge_joint_probability->data[5][30] = 0.0368727136456f;
    edge_joint_probability->data[5][31] = 0.0352212584358f;

// row 6
    edge_joint_probability->data[6][0] = 0.0231622525449f;
    edge_joint_probability->data[6][1] = 0.0458673938149f;
    edge_joint_probability->data[6][2] = 0.0564581770936f;
    edge_joint_probability->data[6][3] = 0.0529536159158f;
    edge_joint_probability->data[6][4] = 0.048152844689f;
    edge_joint_probability->data[6][5] = 0.0216591399228f;
    edge_joint_probability->data[6][6] = 0.0228100810466f;
    edge_joint_probability->data[6][7] = 0.0473110889463f;
    edge_joint_probability->data[6][8] = 0.032592871327f;
    edge_joint_probability->data[6][9] = 0.0213179839297f;
    edge_joint_probability->data[6][10] = 0.0564449744034f;
    edge_joint_probability->data[6][11] = 0.040668135166f;
    edge_joint_probability->data[6][12] = 0.0115693961952f;
    edge_joint_probability->data[6][13] = 0.00911943313092f;
    edge_joint_probability->data[6][14] = 0.0211844473852f;
    edge_joint_probability->data[6][15] = 0.0230214369023f;
    edge_joint_probability->data[6][16] = 0.0545486841336f;
    edge_joint_probability->data[6][17] = 0.0127397081308f;
    edge_joint_probability->data[6][18] = 0.0131462868626f;
    edge_joint_probability->data[6][19] = 0.050695420602f;
    edge_joint_probability->data[6][20] = 0.0534172958604f;
    edge_joint_probability->data[6][21] = 0.0457947528883f;
    edge_joint_probability->data[6][22] = 0.0217124169313f;
    edge_joint_probability->data[6][23] = 0.0223892755056f;
    edge_joint_probability->data[6][24] = 0.0530758113441f;
    edge_joint_probability->data[6][25] = 0.010877869988f;
    edge_joint_probability->data[6][26] = 0.0328325981556f;
    edge_joint_probability->data[6][27] = 0.0508283875962f;
    edge_joint_probability->data[6][28] = 0.00769793296287f;
    edge_joint_probability->data[6][29] = 0.0104713355215f;
    edge_joint_probability->data[6][30] = 0.000525510914291f;
    edge_joint_probability->data[6][31] = 0.0249534401893f;

// row 7
    edge_joint_probability->data[7][0] = 0.000924389671609f;
    edge_joint_probability->data[7][1] = 0.0347020923607f;
    edge_joint_probability->data[7][2] = 0.0122908021731f;
    edge_joint_probability->data[7][3] = 0.0615642384269f;
    edge_joint_probability->data[7][4] = 0.0323998450509f;
    edge_joint_probability->data[7][5] = 0.0292961522606f;
    edge_joint_probability->data[7][6] = 0.0294999136449f;
    edge_joint_probability->data[7][7] = 0.0475122617473f;
    edge_joint_probability->data[7][8] = 0.00611994407966f;
    edge_joint_probability->data[7][9] = 0.00941769288695f;
    edge_joint_probability->data[7][10] = 0.0611196297697f;
    edge_joint_probability->data[7][11] = 0.0222989228093f;
    edge_joint_probability->data[7][12] = 0.0527221992644f;
    edge_joint_probability->data[7][13] = 0.0233822572411f;
    edge_joint_probability->data[7][14] = 0.0231907277673f;
    edge_joint_probability->data[7][15] = 0.0284881486003f;
    edge_joint_probability->data[7][16] = 0.0131707103966f;
    edge_joint_probability->data[7][17] = 0.0219783294403f;
    edge_joint_probability->data[7][18] = 0.0545388426279f;
    edge_joint_probability->data[7][19] = 0.058308599953f;
    edge_joint_probability->data[7][20] = 0.00930057462858f;
    edge_joint_probability->data[7][21] = 0.00246013944105f;
    edge_joint_probability->data[7][22] = 0.0624278294484f;
    edge_joint_probability->data[7][23] = 0.0180897364316f;
    edge_joint_probability->data[7][24] = 0.0302180003234f;
    edge_joint_probability->data[7][25] = 0.0164767028593f;
    edge_joint_probability->data[7][26] = 0.0323894410736f;
    edge_joint_probability->data[7][27] = 0.0111630552474f;
    edge_joint_probability->data[7][28] = 0.0403717314589f;
    edge_joint_probability->data[7][29] = 0.0623636006117f;
    edge_joint_probability->data[7][30] = 0.0615207396317f;
    edge_joint_probability->data[7][31] = 0.0302927486719f;

// row 8
    edge_joint_probability->data[8][0] = 0.0483644019567f;
    edge_joint_probability->data[8][1] = 0.00239694556789f;
    edge_joint_probability->data[8][2] = 0.0136400098561f;
    edge_joint_probability->data[8][3] = 0.0521108237164f;
    edge_joint_probability->data[8][4] = 0.0117607540442f;
    edge_joint_probability->data[8][5] = 0.00874243632404f;
    edge_joint_probability->data[8][6] = 0.0288909361948f;
    edge_joint_probability->data[8][7] = 0.0395783099159f;
    edge_joint_probability->data[8][8] = 0.0173096306531f;
    edge_joint_probability->data[8][9] = 0.0220264902883f;
    edge_joint_probability->data[8][10] = 0.0547468935402f;
    edge_joint_probability->data[8][11] = 0.0392503369442f;
    edge_joint_probability->data[8][12] = 0.0511104249822f;
    edge_joint_probability->data[8][13] = 0.0112581898645f;
    edge_joint_probability->data[8][14] = 0.0195090328878f;
    edge_joint_probability->data[8][15] = 0.0372828947441f;
    edge_joint_probability->data[8][16] = 0.00850503955155f;
    edge_joint_probability->data[8][17] = 0.0501156132619f;
    edge_joint_probability->data[8][18] = 0.0346072822038f;
    edge_joint_probability->data[8][19] = 0.023967885237f;
    edge_joint_probability->data[8][20] = 0.0438965883289f;
    edge_joint_probability->data[8][21] = 0.0341895277687f;
    edge_joint_probability->data[8][22] = 0.0433999454053f;
    edge_joint_probability->data[8][23] = 0.0273462524209f;
    edge_joint_probability->data[8][24] = 0.0373819153772f;
    edge_joint_probability->data[8][25] = 0.0225771413926f;
    edge_joint_probability->data[8][26] = 0.0514814590896f;
    edge_joint_probability->data[8][27] = 0.0394948777313f;
    edge_joint_probability->data[8][28] = 0.00256954458047f;
    edge_joint_probability->data[8][29] = 0.040407861293f;
    edge_joint_probability->data[8][30] = 0.0344853664793f;
    edge_joint_probability->data[8][31] = 0.0475951883983f;

// row 9
    edge_joint_probability->data[9][0] = 0.0099468039623f;
    edge_joint_probability->data[9][1] = 0.0350691765823f;
    edge_joint_probability->data[9][2] = 0.0315632758284f;
    edge_joint_probability->data[9][3] = 0.0508618463928f;
    edge_joint_probability->data[9][4] = 0.0270894478347f;
    edge_joint_probability->data[9][5] = 0.0419782242129f;
    edge_joint_probability->data[9][6] = 0.00769995735977f;
    edge_joint_probability->data[9][7] = 0.0270040303432f;
    edge_joint_probability->data[9][8] = 0.0606482948697f;
    edge_joint_probability->data[9][9] = 0.0135456651645f;
    edge_joint_probability->data[9][10] = 0.00824996695983f;
    edge_joint_probability->data[9][11] = 0.00620379472127f;
    edge_joint_probability->data[9][12] = 0.067274870082f;
    edge_joint_probability->data[9][13] = 0.0540754262359f;
    edge_joint_probability->data[9][14] = 0.0347745964683f;
    edge_joint_probability->data[9][15] = 0.0303809696849f;
    edge_joint_probability->data[9][16] = 0.0266731213354f;
    edge_joint_probability->data[9][17] = 0.0100233750496f;
    edge_joint_probability->data[9][18] = 0.0389655575171f;
    edge_joint_probability->data[9][19] = 0.0150713214319f;
    edge_joint_probability->data[9][20] = 0.0538840993092f;
    edge_joint_probability->data[9][21] = 0.0144518633887f;
    edge_joint_probability->data[9][22] = 0.0555859523638f;
    edge_joint_probability->data[9][23] = 0.0375553592425f;
    edge_joint_probability->data[9][24] = 0.0297487577247f;
    edge_joint_probability->data[9][25] = 0.018851796599f;
    edge_joint_probability->data[9][26] = 0.0643947892056f;
    edge_joint_probability->data[9][27] = 0.0128029710703f;
    edge_joint_probability->data[9][28] = 0.0428961235202f;
    edge_joint_probability->data[9][29] = 0.0467775060456f;
    edge_joint_probability->data[9][30] = 0.00321634081917f;
    edge_joint_probability->data[9][31] = 0.0227347186745f;

// row 10
    edge_joint_probability->data[10][0] = 0.0361693162871f;
    edge_joint_probability->data[10][1] = 0.0307018688309f;
    edge_joint_probability->data[10][2] = 0.0546030666961f;
    edge_joint_probability->data[10][3] = 0.00911531950977f;
    edge_joint_probability->data[10][4] = 0.00292261938896f;
    edge_joint_probability->data[10][5] = 0.0335891931971f;
    edge_joint_probability->data[10][6] = 0.0264693868522f;
    edge_joint_probability->data[10][7] = 0.0541186257274f;
    edge_joint_probability->data[10][8] = 0.00597340255902f;
    edge_joint_probability->data[10][9] = 0.0537728299394f;
    edge_joint_probability->data[10][10] = 0.0298310559715f;
    edge_joint_probability->data[10][11] = 0.0575998441342f;
    edge_joint_probability->data[10][12] = 0.0563842365048f;
    edge_joint_probability->data[10][13] = 0.025545148294f;
    edge_joint_probability->data[10][14] = 0.0641622913347f;
    edge_joint_probability->data[10][15] = 0.0142477408602f;
    edge_joint_probability->data[10][16] = 0.0282897065073f;
    edge_joint_probability->data[10][17] = 0.0217643580309f;
    edge_joint_probability->data[10][18] = 0.0427680747034f;
    edge_joint_probability->data[10][19] = 0.00230818267097f;
    edge_joint_probability->data[10][20] = 0.0278195834021f;
    edge_joint_probability->data[10][21] = 0.00694541437234f;
    edge_joint_probability->data[10][22] = 0.0524773335769f;
    edge_joint_probability->data[10][23] = 0.0203020916949f;
    edge_joint_probability->data[10][24] = 0.0348585522213f;
    edge_joint_probability->data[10][25] = 0.00264570011962f;
    edge_joint_probability->data[10][26] = 0.0553494573206f;
    edge_joint_probability->data[10][27] = 0.000281153155631f;
    edge_joint_probability->data[10][28] = 0.0178486866878f;
    edge_joint_probability->data[10][29] = 0.0223948888653f;
    edge_joint_probability->data[10][30] = 0.0508651929426f;
    edge_joint_probability->data[10][31] = 0.0578756776409f;

// row 11
    edge_joint_probability->data[11][0] = 0.0251642286351f;
    edge_joint_probability->data[11][1] = 0.016087293835f;
    edge_joint_probability->data[11][2] = 0.00693570418642f;
    edge_joint_probability->data[11][3] = 0.00927920549076f;
    edge_joint_probability->data[11][4] = 0.0501011911533f;
    edge_joint_probability->data[11][5] = 0.0184435415918f;
    edge_joint_probability->data[11][6] = 0.0400029200665f;
    edge_joint_probability->data[11][7] = 0.00688581032419f;
    edge_joint_probability->data[11][8] = 0.0163248215935f;
    edge_joint_probability->data[11][9] = 0.0269183983284f;
    edge_joint_probability->data[11][10] = 0.0391901529601f;
    edge_joint_probability->data[11][11] = 0.0267206400967f;
    edge_joint_probability->data[11][12] = 0.05889385081f;
    edge_joint_probability->data[11][13] = 0.0490802733116f;
    edge_joint_probability->data[11][14] = 0.0389544642808f;
    edge_joint_probability->data[11][15] = 0.00156428210653f;
    edge_joint_probability->data[11][16] = 0.0425810635116f;
    edge_joint_probability->data[11][17] = 0.0126199142859f;
    edge_joint_probability->data[11][18] = 0.0453125391945f;
    edge_joint_probability->data[11][19] = 0.0557319552944f;
    edge_joint_probability->data[11][20] = 0.0063863361863f;
    edge_joint_probability->data[11][21] = 0.0216537975668f;
    edge_joint_probability->data[11][22] = 0.0577441393288f;
    edge_joint_probability->data[11][23] = 0.0521353083396f;
    edge_joint_probability->data[11][24] = 0.0338397672675f;
    edge_joint_probability->data[11][25] = 0.0615796489104f;
    edge_joint_probability->data[11][26] = 0.010390721756f;
    edge_joint_probability->data[11][27] = 0.0327347318672f;
    edge_joint_probability->data[11][28] = 0.0063898692095f;
    edge_joint_probability->data[11][29] = 0.0534105777982f;
    edge_joint_probability->data[11][30] = 0.029398611308f;
    edge_joint_probability->data[11][31] = 0.0475442394045f;

// row 12
    edge_joint_probability->data[12][0] = 0.0226874785254f;
    edge_joint_probability->data[12][1] = 0.0663961566476f;
    edge_joint_probability->data[12][2] = 0.00183999478746f;
    edge_joint_probability->data[12][3] = 0.0286171181183f;
    edge_joint_probability->data[12][4] = 0.0667518496966f;
    edge_joint_probability->data[12][5] = 0.0157640233131f;
    edge_joint_probability->data[12][6] = 0.0189396003184f;
    edge_joint_probability->data[12][7] = 0.0296832609574f;
    edge_joint_probability->data[12][8] = 0.0349451888536f;
    edge_joint_probability->data[12][9] = 0.0388364435403f;
    edge_joint_probability->data[12][10] = 0.000922635617432f;
    edge_joint_probability->data[12][11] = 0.0287503396826f;
    edge_joint_probability->data[12][12] = 0.0372572396143f;
    edge_joint_probability->data[12][13] = 0.0341649263019f;
    edge_joint_probability->data[12][14] = 0.028616360043f;
    edge_joint_probability->data[12][15] = 0.0158437966369f;
    edge_joint_probability->data[12][16] = 0.00538852430313f;
    edge_joint_probability->data[12][17] = 0.00727624204236f;
    edge_joint_probability->data[12][18] = 0.0505600823459f;
    edge_joint_probability->data[12][19] = 0.0396088858454f;
    edge_joint_probability->data[12][20] = 0.0488339313505f;
    edge_joint_probability->data[12][21] = 0.0247855314192f;
    edge_joint_probability->data[12][22] = 0.0537002522801f;
    edge_joint_probability->data[12][23] = 0.0120905573782f;
    edge_joint_probability->data[12][24] = 0.0484984676549f;
    edge_joint_probability->data[12][25] = 0.0165189422175f;
    edge_joint_probability->data[12][26] = 0.0396246395908f;
    edge_joint_probability->data[12][27] = 0.0664884157908f;
    edge_joint_probability->data[12][28] = 0.00659390776963f;
    edge_joint_probability->data[12][29] = 0.0517256188594f;
    edge_joint_probability->data[12][30] = 0.0520623769015f;
    edge_joint_probability->data[12][31] = 0.00622721159653f;

// row 13
    edge_joint_probability->data[13][0] = 0.00900923623098f;
    edge_joint_probability->data[13][1] = 0.0162378506812f;
    edge_joint_probability->data[13][2] = 0.0143694224764f;
    edge_joint_probability->data[13][3] = 0.0384825679076f;
    edge_joint_probability->data[13][4] = 0.0433643177025f;
    edge_joint_probability->data[13][5] = 0.0621783537694f;
    edge_joint_probability->data[13][6] = 0.0360588548448f;
    edge_joint_probability->data[13][7] = 0.0184162584052f;
    edge_joint_probability->data[13][8] = 0.0269989910691f;
    edge_joint_probability->data[13][9] = 0.0247076020249f;
    edge_joint_probability->data[13][10] = 0.00482841090862f;
    edge_joint_probability->data[13][11] = 0.0460857830384f;
    edge_joint_probability->data[13][12] = 0.0163240322989f;
    edge_joint_probability->data[13][13] = 0.03107173709f;
    edge_joint_probability->data[13][14] = 0.0474507684886f;
    edge_joint_probability->data[13][15] = 0.0243942726008f;
    edge_joint_probability->data[13][16] = 0.0413616223117f;
    edge_joint_probability->data[13][17] = 0.033391702079f;
    edge_joint_probability->data[13][18] = 0.0182383277708f;
    edge_joint_probability->data[13][19] = 0.0254390175539f;
    edge_joint_probability->data[13][20] = 0.00173507908823f;
    edge_joint_probability->data[13][21] = 0.0593556869015f;
    edge_joint_probability->data[13][22] = 0.0361592394642f;
    edge_joint_probability->data[13][23] = 0.0333929761612f;
    edge_joint_probability->data[13][24] = 0.0167106746038f;
    edge_joint_probability->data[13][25] = 0.0158510053202f;
    edge_joint_probability->data[13][26] = 0.0333049893835f;
    edge_joint_probability->data[13][27] = 0.0468713407807f;
    edge_joint_probability->data[13][28] = 0.0173880731317f;
    edge_joint_probability->data[13][29] = 0.0637405926838f;
    edge_joint_probability->data[13][30] = 0.0426319878941f;
    edge_joint_probability->data[13][31] = 0.0544492253341f;

// row 14
    edge_joint_probability->data[14][0] = 0.0214727010466f;
    edge_joint_probability->data[14][1] = 0.0516222360629f;
    edge_joint_probability->data[14][2] = 0.0650056120291f;
    edge_joint_probability->data[14][3] = 0.0168671340563f;
    edge_joint_probability->data[14][4] = 0.0215340040847f;
    edge_joint_probability->data[14][5] = 0.0357731813709f;
    edge_joint_probability->data[14][6] = 0.0576430944871f;
    edge_joint_probability->data[14][7] = 0.0208506622006f;
    edge_joint_probability->data[14][8] = 0.0642368059864f;
    edge_joint_probability->data[14][9] = 0.00101196356453f;
    edge_joint_probability->data[14][10] = 0.0306987576722f;
    edge_joint_probability->data[14][11] = 0.00294197084779f;
    edge_joint_probability->data[14][12] = 0.0243893662961f;
    edge_joint_probability->data[14][13] = 0.0624532253494f;
    edge_joint_probability->data[14][14] = 0.000736087029993f;
    edge_joint_probability->data[14][15] = 0.0546297042167f;
    edge_joint_probability->data[14][16] = 0.0319153447278f;
    edge_joint_probability->data[14][17] = 0.0200101381533f;
    edge_joint_probability->data[14][18] = 0.0268878287023f;
    edge_joint_probability->data[14][19] = 0.0169651870509f;
    edge_joint_probability->data[14][20] = 0.00563627666948f;
    edge_joint_probability->data[14][21] = 0.0482019449939f;
    edge_joint_probability->data[14][22] = 0.0515794797911f;
    edge_joint_probability->data[14][23] = 0.0598741860924f;
    edge_joint_probability->data[14][24] = 0.0341838885278f;
    edge_joint_probability->data[14][25] = 0.0621201509473f;
    edge_joint_probability->data[14][26] = 0.00378250411903f;
    edge_joint_probability->data[14][27] = 0.0593989827857f;
    edge_joint_probability->data[14][28] = 0.00732743585208f;
    edge_joint_probability->data[14][29] = 0.00609877246076f;
    edge_joint_probability->data[14][30] = 0.0238402433092f;
    edge_joint_probability->data[14][31] = 0.0103111295158f;

// row 15
    edge_joint_probability->data[15][0] = 0.057705523724f;
    edge_joint_probability->data[15][1] = 0.0343825467408f;
    edge_joint_probability->data[15][2] = 0.0220026974039f;
    edge_joint_probability->data[15][3] = 0.0384060338603f;
    edge_joint_probability->data[15][4] = 0.0309720118994f;
    edge_joint_probability->data[15][5] = 0.0135538050955f;
    edge_joint_probability->data[15][6] = 0.0315913870188f;
    edge_joint_probability->data[15][7] = 0.0264396431539f;
    edge_joint_probability->data[15][8] = 0.0545372265861f;
    edge_joint_probability->data[15][9] = 0.037854217889f;
    edge_joint_probability->data[15][10] = 0.0079002533192f;
    edge_joint_probability->data[15][11] = 0.0207933647523f;
    edge_joint_probability->data[15][12] = 0.0417128793722f;
    edge_joint_probability->data[15][13] = 0.0171450037891f;
    edge_joint_probability->data[15][14] = 0.0437663465654f;
    edge_joint_probability->data[15][15] = 0.0455708123951f;
    edge_joint_probability->data[15][16] = 0.0156057135727f;
    edge_joint_probability->data[15][17] = 0.047135274573f;
    edge_joint_probability->data[15][18] = 0.0440012787707f;
    edge_joint_probability->data[15][19] = 0.0214139635376f;
    edge_joint_probability->data[15][20] = 0.0279094053923f;
    edge_joint_probability->data[15][21] = 0.0381300602135f;
    edge_joint_probability->data[15][22] = 0.0554856634938f;
    edge_joint_probability->data[15][23] = 0.0500527382132f;
    edge_joint_probability->data[15][24] = 0.0159071525786f;
    edge_joint_probability->data[15][25] = 0.0471723742367f;
    edge_joint_probability->data[15][26] = 0.0252851240887f;
    edge_joint_probability->data[15][27] = 0.00703223195001f;
    edge_joint_probability->data[15][28] = 0.0079936340456f;
    edge_joint_probability->data[15][29] = 0.00966395845434f;
    edge_joint_probability->data[15][30] = 0.0400563355311f;
    edge_joint_probability->data[15][31] = 0.0228213377832f;

// row 16
    edge_joint_probability->data[16][0] = 0.0409721504862f;
    edge_joint_probability->data[16][1] = 0.025000441763f;
    edge_joint_probability->data[16][2] = 0.0150793868888f;
    edge_joint_probability->data[16][3] = 0.0252465138055f;
    edge_joint_probability->data[16][4] = 0.00394595444465f;
    edge_joint_probability->data[16][5] = 0.0572253530675f;
    edge_joint_probability->data[16][6] = 0.0317584872793f;
    edge_joint_probability->data[16][7] = 0.049446699218f;
    edge_joint_probability->data[16][8] = 0.00408534100054f;
    edge_joint_probability->data[16][9] = 0.0158416056835f;
    edge_joint_probability->data[16][10] = 0.000990516269402f;
    edge_joint_probability->data[16][11] = 0.0631358193529f;
    edge_joint_probability->data[16][12] = 0.0444456578836f;
    edge_joint_probability->data[16][13] = 0.0408216038023f;
    edge_joint_probability->data[16][14] = 0.0214063138405f;
    edge_joint_probability->data[16][15] = 0.0156779534185f;
    edge_joint_probability->data[16][16] = 0.0222807564195f;
    edge_joint_probability->data[16][17] = 0.0245658614952f;
    edge_joint_probability->data[16][18] = 0.0248133472283f;
    edge_joint_probability->data[16][19] = 0.0293564110461f;
    edge_joint_probability->data[16][20] = 0.0426223887459f;
    edge_joint_probability->data[16][21] = 0.0172119664245f;
    edge_joint_probability->data[16][22] = 0.0103439230629f;
    edge_joint_probability->data[16][23] = 0.0541865599032f;
    edge_joint_probability->data[16][24] = 0.0545778119423f;
    edge_joint_probability->data[16][25] = 0.00460077559389f;
    edge_joint_probability->data[16][26] = 0.0272532449128f;
    edge_joint_probability->data[16][27] = 0.0446936494706f;
    edge_joint_probability->data[16][28] = 0.0145850344825f;
    edge_joint_probability->data[16][29] = 0.063414058694f;
    edge_joint_probability->data[16][30] = 0.0646614784048f;
    edge_joint_probability->data[16][31] = 0.0457529339693f;

// row 17
    edge_joint_probability->data[17][0] = 0.045808693621f;
    edge_joint_probability->data[17][1] = 0.0357876792631f;
    edge_joint_probability->data[17][2] = 0.0397350081414f;
    edge_joint_probability->data[17][3] = 0.0177255532098f;
    edge_joint_probability->data[17][4] = 0.0223033506017f;
    edge_joint_probability->data[17][5] = 0.00258439438384f;
    edge_joint_probability->data[17][6] = 0.0332702618459f;
    edge_joint_probability->data[17][7] = 0.0371050296495f;
    edge_joint_probability->data[17][8] = 0.0472327539773f;
    edge_joint_probability->data[17][9] = 0.0508840157809f;
    edge_joint_probability->data[17][10] = 0.0534166303339f;
    edge_joint_probability->data[17][11] = 0.0479685597114f;
    edge_joint_probability->data[17][12] = 0.0183978346832f;
    edge_joint_probability->data[17][13] = 0.0179000628919f;
    edge_joint_probability->data[17][14] = 0.0506124768415f;
    edge_joint_probability->data[17][15] = 0.0537977466043f;
    edge_joint_probability->data[17][16] = 0.00988808313752f;
    edge_joint_probability->data[17][17] = 0.0394915840876f;
    edge_joint_probability->data[17][18] = 0.0160757645162f;
    edge_joint_probability->data[17][19] = 0.0217284721925f;
    edge_joint_probability->data[17][20] = 0.0292496322871f;
    edge_joint_probability->data[17][21] = 0.0207799143361f;
    edge_joint_probability->data[17][22] = 0.0320247949395f;
    edge_joint_probability->data[17][23] = 0.0224745594636f;
    edge_joint_probability->data[17][24] = 0.0124661708277f;
    edge_joint_probability->data[17][25] = 0.00452359325803f;
    edge_joint_probability->data[17][26] = 0.0329498774489f;
    edge_joint_probability->data[17][27] = 0.0214142976004f;
    edge_joint_probability->data[17][28] = 0.050548031475f;
    edge_joint_probability->data[17][29] = 0.0538992372937f;
    edge_joint_probability->data[17][30] = 0.0188597768346f;
    edge_joint_probability->data[17][31] = 0.0390961587609f;

// row 18
    edge_joint_probability->data[18][0] = 0.0610013992125f;
    edge_joint_probability->data[18][1] = 0.0112902157262f;
    edge_joint_probability->data[18][2] = 0.0470937571811f;
    edge_joint_probability->data[18][3] = 0.00192936768842f;
    edge_joint_probability->data[18][4] = 0.0570986344259f;
    edge_joint_probability->data[18][5] = 0.0213186474707f;
    edge_joint_probability->data[18][6] = 0.0625590003627f;
    edge_joint_probability->data[18][7] = 0.00755199958651f;
    edge_joint_probability->data[18][8] = 0.00144682146133f;
    edge_joint_probability->data[18][9] = 0.0333278304099f;
    edge_joint_probability->data[18][10] = 0.0181472024008f;
    edge_joint_probability->data[18][11] = 0.0629042728147f;
    edge_joint_probability->data[18][12] = 0.0277889348661f;
    edge_joint_probability->data[18][13] = 0.0526353850622f;
    edge_joint_probability->data[18][14] = 0.0566441926252f;
    edge_joint_probability->data[18][15] = 0.0368859056561f;
    edge_joint_probability->data[18][16] = 0.000800300149368f;
    edge_joint_probability->data[18][17] = 0.0582755426863f;
    edge_joint_probability->data[18][18] = 0.0044165883851f;
    edge_joint_probability->data[18][19] = 0.0119817151932f;
    edge_joint_probability->data[18][20] = 0.0200583676462f;
    edge_joint_probability->data[18][21] = 0.00332991519797f;
    edge_joint_probability->data[18][22] = 0.0112134841041f;
    edge_joint_probability->data[18][23] = 0.0377295336413f;
    edge_joint_probability->data[18][24] = 0.0596028839768f;
    edge_joint_probability->data[18][25] = 0.00207125039239f;
    edge_joint_probability->data[18][26] = 0.0117992147394f;
    edge_joint_probability->data[18][27] = 0.0398874309104f;
    edge_joint_probability->data[18][28] = 0.0570105750919f;
    edge_joint_probability->data[18][29] = 0.0385587169655f;
    edge_joint_probability->data[18][30] = 0.0452557109767f;
    edge_joint_probability->data[18][31] = 0.0383852029928f;

// row 19
    edge_joint_probability->data[19][0] = 0.0580188428528f;
    edge_joint_probability->data[19][1] = 0.0569260388639f;
    edge_joint_probability->data[19][2] = 0.0127072645209f;
    edge_joint_probability->data[19][3] = 0.00973651941221f;
    edge_joint_probability->data[19][4] = 0.0493580959951f;
    edge_joint_probability->data[19][5] = 0.00836037466583f;
    edge_joint_probability->data[19][6] = 0.0397151713238f;
    edge_joint_probability->data[19][7] = 0.00816473943329f;
    edge_joint_probability->data[19][8] = 0.00395045619896f;
    edge_joint_probability->data[19][9] = 0.00254096098515f;
    edge_joint_probability->data[19][10] = 0.0374187805244f;
    edge_joint_probability->data[19][11] = 0.0291275500779f;
    edge_joint_probability->data[19][12] = 0.00601655288245f;
    edge_joint_probability->data[19][13] = 0.040750816804f;
    edge_joint_probability->data[19][14] = 0.0292143855274f;
    edge_joint_probability->data[19][15] = 0.00722114481441f;
    edge_joint_probability->data[19][16] = 0.0425457997067f;
    edge_joint_probability->data[19][17] = 0.0342500923954f;
    edge_joint_probability->data[19][18] = 0.0581417110983f;
    edge_joint_probability->data[19][19] = 0.0196315418772f;
    edge_joint_probability->data[19][20] = 0.028507128776f;
    edge_joint_probability->data[19][21] = 0.0469047660707f;
    edge_joint_probability->data[19][22] = 0.0569438193825f;
    edge_joint_probability->data[19][23] = 0.0356008772255f;
    edge_joint_probability->data[19][24] = 0.0020784486862f;
    edge_joint_probability->data[19][25] = 0.0212191812565f;
    edge_joint_probability->data[19][26] = 0.0251553624194f;
    edge_joint_probability->data[19][27] = 0.0461247076192f;
    edge_joint_probability->data[19][28] = 0.0415852156341f;
    edge_joint_probability->data[19][29] = 0.0418357688482f;
    edge_joint_probability->data[19][30] = 0.0438481088909f;
    edge_joint_probability->data[19][31] = 0.0563997752308f;

// row 20
    edge_joint_probability->data[20][0] = 0.0397908660775f;
    edge_joint_probability->data[20][1] = 0.0481167946831f;
    edge_joint_probability->data[20][2] = 0.0267985922663f;
    edge_joint_probability->data[20][3] = 0.041672553612f;
    edge_joint_probability->data[20][4] = 0.0333646096203f;
    edge_joint_probability->data[20][5] = 0.0546389677889f;
    edge_joint_probability->data[20][6] = 0.000926518995163f;
    edge_joint_probability->data[20][7] = 0.025715147114f;
    edge_joint_probability->data[20][8] = 0.00369962332529f;
    edge_joint_probability->data[20][9] = 0.0618050634373f;
    edge_joint_probability->data[20][10] = 0.048199707887f;
    edge_joint_probability->data[20][11] = 0.0102825039795f;
    edge_joint_probability->data[20][12] = 0.055408505653f;
    edge_joint_probability->data[20][13] = 0.0434684542604f;
    edge_joint_probability->data[20][14] = 0.027503349878f;
    edge_joint_probability->data[20][15] = 0.0490057602365f;
    edge_joint_probability->data[20][16] = 0.0164436560573f;
    edge_joint_probability->data[20][17] = 0.0503270736223f;
    edge_joint_probability->data[20][18] = 0.053737916995f;
    edge_joint_probability->data[20][19] = 0.0512299626488f;
    edge_joint_probability->data[20][20] = 0.00937302062195f;
    edge_joint_probability->data[20][21] = 0.00569155958049f;
    edge_joint_probability->data[20][22] = 0.00713211309899f;
    edge_joint_probability->data[20][23] = 0.0292148148476f;
    edge_joint_probability->data[20][24] = 0.0336312286845f;
    edge_joint_probability->data[20][25] = 0.00551192789731f;
    edge_joint_probability->data[20][26] = 0.04097596854f;
    edge_joint_probability->data[20][27] = 0.0208690512201f;
    edge_joint_probability->data[20][28] = 0.0175024452575f;
    edge_joint_probability->data[20][29] = 0.034572867528f;
    edge_joint_probability->data[20][30] = 0.0479636445297f;
    edge_joint_probability->data[20][31] = 0.00542573005609f;

// row 21
    edge_joint_probability->data[21][0] = 0.0169975174586f;
    edge_joint_probability->data[21][1] = 0.0524424261409f;
    edge_joint_probability->data[21][2] = 0.0516500515017f;
    edge_joint_probability->data[21][3] = 0.0570486057203f;
    edge_joint_probability->data[21][4] = 0.0121580897936f;
    edge_joint_probability->data[21][5] = 0.0309721757701f;
    edge_joint_probability->data[21][6] = 0.0418605934586f;
    edge_joint_probability->data[21][7] = 0.0566443696302f;
    edge_joint_probability->data[21][8] = 0.0117902080536f;
    edge_joint_probability->data[21][9] = 0.0023609740771f;
    edge_joint_probability->data[21][10] = 0.0170033192164f;
    edge_joint_probability->data[21][11] = 0.0449418833658f;
    edge_joint_probability->data[21][12] = 0.019366277387f;
    edge_joint_probability->data[21][13] = 0.00371292741754f;
    edge_joint_probability->data[21][14] = 0.0383052615358f;
    edge_joint_probability->data[21][15] = 0.00272590157995f;
    edge_joint_probability->data[21][16] = 0.0513558495224f;
    edge_joint_probability->data[21][17] = 0.0405659131333f;
    edge_joint_probability->data[21][18] = 0.0420839938237f;
    edge_joint_probability->data[21][19] = 0.0415253101381f;
    edge_joint_probability->data[21][20] = 0.00453296763921f;
    edge_joint_probability->data[21][21] = 0.0408229756182f;
    edge_joint_probability->data[21][22] = 0.0554392432558f;
    edge_joint_probability->data[21][23] = 0.0153592295772f;
    edge_joint_probability->data[21][24] = 0.0366935345204f;
    edge_joint_probability->data[21][25] = 0.00959194923125f;
    edge_joint_probability->data[21][26] = 0.0496071322515f;
    edge_joint_probability->data[21][27] = 0.0532842086522f;
    edge_joint_probability->data[21][28] = 0.0330043759897f;
    edge_joint_probability->data[21][29] = 0.000656898835577f;
    edge_joint_probability->data[21][30] = 0.00881797398012f;
    edge_joint_probability->data[21][31] = 0.0566778617241f;

// row 22
    edge_joint_probability->data[22][0] = 0.0572087545287f;
    edge_joint_probability->data[22][1] = 0.0205867334033f;
    edge_joint_probability->data[22][2] = 0.0540571387994f;
    edge_joint_probability->data[22][3] = 0.0621271547486f;
    edge_joint_probability->data[22][4] = 0.0531694437797f;
    edge_joint_probability->data[22][5] = 0.0207256506479f;
    edge_joint_probability->data[22][6] = 0.00645268876791f;
    edge_joint_probability->data[22][7] = 0.0151750472565f;
    edge_joint_probability->data[22][8] = 0.0083942016307f;
    edge_joint_probability->data[22][9] = 0.0563870518529f;
    edge_joint_probability->data[22][10] = 0.0632598243809f;
    edge_joint_probability->data[22][11] = 0.0579407741944f;
    edge_joint_probability->data[22][12] = 0.0432417334675f;
    edge_joint_probability->data[22][13] = 0.0272273575975f;
    edge_joint_probability->data[22][14] = 0.0255008653749f;
    edge_joint_probability->data[22][15] = 0.0201844656208f;
    edge_joint_probability->data[22][16] = 0.0530705751805f;
    edge_joint_probability->data[22][17] = 0.00712897450104f;
    edge_joint_probability->data[22][18] = 0.021292186093f;
    edge_joint_probability->data[22][19] = 0.00692191745314f;
    edge_joint_probability->data[22][20] = 0.00797272224849f;
    edge_joint_probability->data[22][21] = 0.00808716335269f;
    edge_joint_probability->data[22][22] = 0.061356080424f;
    edge_joint_probability->data[22][23] = 0.000353696803724f;
    edge_joint_probability->data[22][24] = 0.026291808453f;
    edge_joint_probability->data[22][25] = 0.00664201520544f;
    edge_joint_probability->data[22][26] = 0.0380807470675f;
    edge_joint_probability->data[22][27] = 0.0341462400466f;
    edge_joint_probability->data[22][28] = 0.0630816443329f;
    edge_joint_probability->data[22][29] = 0.0043371190783f;
    edge_joint_probability->data[22][30] = 0.046194054225f;
    edge_joint_probability->data[22][31] = 0.0234041694832f;

// row 23
    edge_joint_probability->data[23][0] = 0.0464357579419f;
    edge_joint_probability->data[23][1] = 0.00324365935862f;
    edge_joint_probability->data[23][2] = 0.0219472361568f;
    edge_joint_probability->data[23][3] = 0.0434102637156f;
    edge_joint_probability->data[23][4] = 0.00186618747773f;
    edge_joint_probability->data[23][5] = 0.0108749786393f;
    edge_joint_probability->data[23][6] = 0.0479946739643f;
    edge_joint_probability->data[23][7] = 0.0315719892559f;
    edge_joint_probability->data[23][8] = 0.0588729264168f;
    edge_joint_probability->data[23][9] = 0.021845783673f;
    edge_joint_probability->data[23][10] = 0.0443643170291f;
    edge_joint_probability->data[23][11] = 0.0277242670553f;
    edge_joint_probability->data[23][12] = 0.0471302282367f;
    edge_joint_probability->data[23][13] = 0.043406822164f;
    edge_joint_probability->data[23][14] = 0.0489224470442f;
    edge_joint_probability->data[23][15] = 0.000947358578508f;
    edge_joint_probability->data[23][16] = 0.031192231274f;
    edge_joint_probability->data[23][17] = 0.0109726631537f;
    edge_joint_probability->data[23][18] = 0.00814896125159f;
    edge_joint_probability->data[23][19] = 0.0460650769157f;
    edge_joint_probability->data[23][20] = 0.0402121000091f;
    edge_joint_probability->data[23][21] = 0.0250941448095f;
    edge_joint_probability->data[23][22] = 0.0119832673174f;
    edge_joint_probability->data[23][23] = 0.0122795313452f;
    edge_joint_probability->data[23][24] = 0.0389280260517f;
    edge_joint_probability->data[23][25] = 0.0496242040129f;
    edge_joint_probability->data[23][26] = 0.0549849073626f;
    edge_joint_probability->data[23][27] = 0.0348338014118f;
    edge_joint_probability->data[23][28] = 0.0561355595733f;
    edge_joint_probability->data[23][29] = 0.0496519166049f;
    edge_joint_probability->data[23][30] = 0.00560811503169f;
    edge_joint_probability->data[23][31] = 0.0237265971672f;

// row 24
    edge_joint_probability->data[24][0] = 0.0530005451514f;
    edge_joint_probability->data[24][1] = 0.00600383086577f;
    edge_joint_probability->data[24][2] = 0.0481487064208f;
    edge_joint_probability->data[24][3] = 0.00892295441776f;
    edge_joint_probability->data[24][4] = 0.0337216613932f;
    edge_joint_probability->data[24][5] = 0.0326960649997f;
    edge_joint_probability->data[24][6] = 0.0703900201929f;
    edge_joint_probability->data[24][7] = 0.0181115102448f;
    edge_joint_probability->data[24][8] = 0.00655302448237f;
    edge_joint_probability->data[24][9] = 0.0585190592466f;
    edge_joint_probability->data[24][10] = 0.0630197595004f;
    edge_joint_probability->data[24][11] = 0.0309933606201f;
    edge_joint_probability->data[24][12] = 0.0251797239258f;
    edge_joint_probability->data[24][13] = 0.0651784247492f;
    edge_joint_probability->data[24][14] = 0.0427964694453f;
    edge_joint_probability->data[24][15] = 0.0240240632415f;
    edge_joint_probability->data[24][16] = 0.0369717060564f;
    edge_joint_probability->data[24][17] = 0.030634642077f;
    edge_joint_probability->data[24][18] = 0.024645540139f;
    edge_joint_probability->data[24][19] = 0.00433880738371f;
    edge_joint_probability->data[24][20] = 0.0430316963015f;
    edge_joint_probability->data[24][21] = 0.0433434705552f;
    edge_joint_probability->data[24][22] = 0.0113592634027f;
    edge_joint_probability->data[24][23] = 0.0554872914223f;
    edge_joint_probability->data[24][24] = 0.0354123953352f;
    edge_joint_probability->data[24][25] = 0.00838945900028f;
    edge_joint_probability->data[24][26] = 0.0241532834623f;
    edge_joint_probability->data[24][27] = 0.0293520376134f;
    edge_joint_probability->data[24][28] = 0.0180851745062f;
    edge_joint_probability->data[24][29] = 0.0186619759286f;
    edge_joint_probability->data[24][30] = 0.00600985680222f;
    edge_joint_probability->data[24][31] = 0.0228642211165f;

// row 25
    edge_joint_probability->data[25][0] = 0.0259764095667f;
    edge_joint_probability->data[25][1] = 0.0298797030015f;
    edge_joint_probability->data[25][2] = 0.0271662116528f;
    edge_joint_probability->data[25][3] = 0.0442344173505f;
    edge_joint_probability->data[25][4] = 0.0438110684186f;
    edge_joint_probability->data[25][5] = 0.0503504082147f;
    edge_joint_probability->data[25][6] = 0.0392054171458f;
    edge_joint_probability->data[25][7] = 0.0174687454672f;
    edge_joint_probability->data[25][8] = 0.0482539907741f;
    edge_joint_probability->data[25][9] = 0.0218815917691f;
    edge_joint_probability->data[25][10] = 0.0343155755532f;
    edge_joint_probability->data[25][11] = 0.0205067841587f;
    edge_joint_probability->data[25][12] = 0.00899657977485f;
    edge_joint_probability->data[25][13] = 0.0473668328195f;
    edge_joint_probability->data[25][14] = 0.02104137025f;
    edge_joint_probability->data[25][15] = 0.0376652526506f;
    edge_joint_probability->data[25][16] = 0.045131100526f;
    edge_joint_probability->data[25][17] = 0.0198021025856f;
    edge_joint_probability->data[25][18] = 0.0401995560997f;
    edge_joint_probability->data[25][19] = 0.0454002784305f;
    edge_joint_probability->data[25][20] = 0.0423552268005f;
    edge_joint_probability->data[25][21] = 0.0257427557705f;
    edge_joint_probability->data[25][22] = 0.00958076977728f;
    edge_joint_probability->data[25][23] = 0.0463047598244f;
    edge_joint_probability->data[25][24] = 0.0446319549236f;
    edge_joint_probability->data[25][25] = 0.00712882270811f;
    edge_joint_probability->data[25][26] = 0.0375087951225f;
    edge_joint_probability->data[25][27] = 0.0320493901446f;
    edge_joint_probability->data[25][28] = 0.0175537354863f;
    edge_joint_probability->data[25][29] = 0.0207163193917f;
    edge_joint_probability->data[25][30] = 0.0338345486881f;
    edge_joint_probability->data[25][31] = 0.013939525153f;

// row 26
    edge_joint_probability->data[26][0] = 0.00547857574942f;
    edge_joint_probability->data[26][1] = 0.0624229152133f;
    edge_joint_probability->data[26][2] = 0.035029413883f;
    edge_joint_probability->data[26][3] = 0.041733047686f;
    edge_joint_probability->data[26][4] = 0.0396732841672f;
    edge_joint_probability->data[26][5] = 0.0235969499163f;
    edge_joint_probability->data[26][6] = 0.0111641860099f;
    edge_joint_probability->data[26][7] = 0.0134213004248f;
    edge_joint_probability->data[26][8] = 0.0555860620246f;
    edge_joint_probability->data[26][9] = 0.0618942907799f;
    edge_joint_probability->data[26][10] = 0.0363057562874f;
    edge_joint_probability->data[26][11] = 0.00676797580525f;
    edge_joint_probability->data[26][12] = 0.0666178121102f;
    edge_joint_probability->data[26][13] = 0.019879920901f;
    edge_joint_probability->data[26][14] = 0.0243653241582f;
    edge_joint_probability->data[26][15] = 0.0683125257235f;
    edge_joint_probability->data[26][16] = 0.012356892248f;
    edge_joint_probability->data[26][17] = 0.021245020497f;
    edge_joint_probability->data[26][18] = 0.0140963473297f;
    edge_joint_probability->data[26][19] = 0.0372093445931f;
    edge_joint_probability->data[26][20] = 0.00751938410238f;
    edge_joint_probability->data[26][21] = 0.018610606015f;
    edge_joint_probability->data[26][22] = 0.0421151240047f;
    edge_joint_probability->data[26][23] = 0.0540829072591f;
    edge_joint_probability->data[26][24] = 0.0290611187877f;
    edge_joint_probability->data[26][25] = 0.0347964294944f;
    edge_joint_probability->data[26][26] = 0.00972689430276f;
    edge_joint_probability->data[26][27] = 0.0278459665474f;
    edge_joint_probability->data[26][28] = 0.049560418939f;
    edge_joint_probability->data[26][29] = 0.0172968389471f;
    edge_joint_probability->data[26][30] = 0.0127951741999f;
    edge_joint_probability->data[26][31] = 0.0394321918928f;

// row 27
    edge_joint_probability->data[27][0] = 0.00640684222366f;
    edge_joint_probability->data[27][1] = 0.038555654859f;
    edge_joint_probability->data[27][2] = 0.0549551291964f;
    edge_joint_probability->data[27][3] = 0.0545802459566f;
    edge_joint_probability->data[27][4] = 0.0344865626822f;
    edge_joint_probability->data[27][5] = 0.0274809132584f;
    edge_joint_probability->data[27][6] = 0.00481510779629f;
    edge_joint_probability->data[27][7] = 0.063056513367f;
    edge_joint_probability->data[27][8] = 0.0341219582759f;
    edge_joint_probability->data[27][9] = 0.0134319325853f;
    edge_joint_probability->data[27][10] = 0.032074327237f;
    edge_joint_probability->data[27][11] = 0.0348978911399f;
    edge_joint_probability->data[27][12] = 0.0154324483208f;
    edge_joint_probability->data[27][13] = 0.0490470727399f;
    edge_joint_probability->data[27][14] = 0.00575263716283f;
    edge_joint_probability->data[27][15] = 0.0551825139682f;
    edge_joint_probability->data[27][16] = 0.0578983347262f;
    edge_joint_probability->data[27][17] = 0.0324794184741f;
    edge_joint_probability->data[27][18] = 0.0111248337837f;
    edge_joint_probability->data[27][19] = 0.0272444559804f;
    edge_joint_probability->data[27][20] = 0.0164795873959f;
    edge_joint_probability->data[27][21] = 0.0353752665519f;
    edge_joint_probability->data[27][22] = 0.0386478631543f;
    edge_joint_probability->data[27][23] = 0.0533851340543f;
    edge_joint_probability->data[27][24] = 0.0533751031985f;
    edge_joint_probability->data[27][25] = 0.0338872123481f;
    edge_joint_probability->data[27][26] = 0.0236317382662f;
    edge_joint_probability->data[27][27] = 0.0116671496768f;
    edge_joint_probability->data[27][28] = 0.0191437449705f;
    edge_joint_probability->data[27][29] = 0.0367460325263f;
    edge_joint_probability->data[27][30] = 0.0211147794402f;
    edge_joint_probability->data[27][31] = 0.00352159468338f;

// row 28
    edge_joint_probability->data[28][0] = 0.0456188098441f;
    edge_joint_probability->data[28][1] = 0.059148616483f;
    edge_joint_probability->data[28][2] = 0.0478563998367f;
    edge_joint_probability->data[28][3] = 0.0642656880631f;
    edge_joint_probability->data[28][4] = 0.0189190289484f;
    edge_joint_probability->data[28][5] = 0.00852157208883f;
    edge_joint_probability->data[28][6] = 0.0351038149999f;
    edge_joint_probability->data[28][7] = 0.0135244460992f;
    edge_joint_probability->data[28][8] = 0.0358542752061f;
    edge_joint_probability->data[28][9] = 0.0334768374806f;
    edge_joint_probability->data[28][10] = 0.0346303701805f;
    edge_joint_probability->data[28][11] = 0.0371688921414f;
    edge_joint_probability->data[28][12] = 0.0150725530565f;
    edge_joint_probability->data[28][13] = 0.042135448129f;
    edge_joint_probability->data[28][14] = 0.0016603342938f;
    edge_joint_probability->data[28][15] = 0.033256504561f;
    edge_joint_probability->data[28][16] = 0.0294198299556f;
    edge_joint_probability->data[28][17] = 0.0221975829476f;
    edge_joint_probability->data[28][18] = 0.0548683831663f;
    edge_joint_probability->data[28][19] = 0.021976627883f;
    edge_joint_probability->data[28][20] = 0.0117249665022f;
    edge_joint_probability->data[28][21] = 0.0571993097612f;
    edge_joint_probability->data[28][22] = 0.0124922814582f;
    edge_joint_probability->data[28][23] = 0.0324944154313f;
    edge_joint_probability->data[28][24] = 0.0380583150714f;
    edge_joint_probability->data[28][25] = 0.0521939322953f;
    edge_joint_probability->data[28][26] = 0.0597725380005f;
    edge_joint_probability->data[28][27] = 0.0257887754756f;
    edge_joint_probability->data[28][28] = 0.00761899041283f;
    edge_joint_probability->data[28][29] = 0.00955398019584f;
    edge_joint_probability->data[28][30] = 0.033368282194f;
    edge_joint_probability->data[28][31] = 0.00505819783703f;

// row 29
    edge_joint_probability->data[29][0] = 0.0265685422054f;
    edge_joint_probability->data[29][1] = 0.0606124799632f;
    edge_joint_probability->data[29][2] = 0.00576484329768f;
    edge_joint_probability->data[29][3] = 0.00419851299023f;
    edge_joint_probability->data[29][4] = 0.0542384791376f;
    edge_joint_probability->data[29][5] = 0.0536476055572f;
    edge_joint_probability->data[29][6] = 0.0213192647348f;
    edge_joint_probability->data[29][7] = 0.0207320049368f;
    edge_joint_probability->data[29][8] = 0.0380480396643f;
    edge_joint_probability->data[29][9] = 0.00410363888476f;
    edge_joint_probability->data[29][10] = 0.00880643680958f;
    edge_joint_probability->data[29][11] = 0.0290751856934f;
    edge_joint_probability->data[29][12] = 0.00378279893915f;
    edge_joint_probability->data[29][13] = 0.0555143295535f;
    edge_joint_probability->data[29][14] = 0.00158344340358f;
    edge_joint_probability->data[29][15] = 0.0520180313678f;
    edge_joint_probability->data[29][16] = 0.0544156175913f;
    edge_joint_probability->data[29][17] = 0.0643936708615f;
    edge_joint_probability->data[29][18] = 0.00314149160739f;
    edge_joint_probability->data[29][19] = 0.0158040890281f;
    edge_joint_probability->data[29][20] = 0.041266098792f;
    edge_joint_probability->data[29][21] = 0.0577303860418f;
    edge_joint_probability->data[29][22] = 0.0296393263194f;
    edge_joint_probability->data[29][23] = 0.00341519529242f;
    edge_joint_probability->data[29][24] = 0.0230138968018f;
    edge_joint_probability->data[29][25] = 0.0155071635815f;
    edge_joint_probability->data[29][26] = 0.022750117436f;
    edge_joint_probability->data[29][27] = 0.0418944354458f;
    edge_joint_probability->data[29][28] = 0.0520954089043f;
    edge_joint_probability->data[29][29] = 0.0537578108378f;
    edge_joint_probability->data[29][30] = 0.0634349678431f;
    edge_joint_probability->data[29][31] = 0.0177266864768f;

// row 30
    edge_joint_probability->data[30][0] = 0.0486256989678f;
    edge_joint_probability->data[30][1] = 0.00537072107678f;
    edge_joint_probability->data[30][2] = 0.0420130359879f;
    edge_joint_probability->data[30][3] = 0.000799536060496f;
    edge_joint_probability->data[30][4] = 0.0423400767926f;
    edge_joint_probability->data[30][5] = 0.0510790312714f;
    edge_joint_probability->data[30][6] = 0.00730739597904f;
    edge_joint_probability->data[30][7] = 0.00824711272221f;
    edge_joint_probability->data[30][8] = 0.0338395395302f;
    edge_joint_probability->data[30][9] = 0.0247703780982f;
    edge_joint_probability->data[30][10] = 0.0460450067798f;
    edge_joint_probability->data[30][11] = 0.0435620183735f;
    edge_joint_probability->data[30][12] = 0.0235823046246f;
    edge_joint_probability->data[30][13] = 0.0225174027249f;
    edge_joint_probability->data[30][14] = 0.0487813366913f;
    edge_joint_probability->data[30][15] = 0.0322342013628f;
    edge_joint_probability->data[30][16] = 0.0577910110761f;
    edge_joint_probability->data[30][17] = 0.0521762534548f;
    edge_joint_probability->data[30][18] = 0.0548305053897f;
    edge_joint_probability->data[30][19] = 0.029997840186f;
    edge_joint_probability->data[30][20] = 0.00777229487217f;
    edge_joint_probability->data[30][21] = 0.00843502811071f;
    edge_joint_probability->data[30][22] = 0.00739308975345f;
    edge_joint_probability->data[30][23] = 0.0363311298261f;
    edge_joint_probability->data[30][24] = 0.0296715515476f;
    edge_joint_probability->data[30][25] = 0.0130839785331f;
    edge_joint_probability->data[30][26] = 0.0346522533914f;
    edge_joint_probability->data[30][27] = 0.0589094002328f;
    edge_joint_probability->data[30][28] = 0.00502599023567f;
    edge_joint_probability->data[30][29] = 0.0277155847932f;
    edge_joint_probability->data[30][30] = 0.052058458151f;
    edge_joint_probability->data[30][31] = 0.0430408334027f;

// row 31
    edge_joint_probability->data[31][0] = 0.0509004883725f;
    edge_joint_probability->data[31][1] = 0.0127595354693f;
    edge_joint_probability->data[31][2] = 0.0549548790253f;
    edge_joint_probability->data[31][3] = 0.0433614720624f;
    edge_joint_probability->data[31][4] = 0.0134477518387f;
    edge_joint_probability->data[31][5] = 0.0337516700787f;
    edge_joint_probability->data[31][6] = 0.00586526335834f;
    edge_joint_probability->data[31][7] = 0.0533269938853f;
    edge_joint_probability->data[31][8] = 0.0182653710467f;
    edge_joint_probability->data[31][9] = 0.0366611873799f;
    edge_joint_probability->data[31][10] = 0.010153725725f;
    edge_joint_probability->data[31][11] = 0.0500836397726f;
    edge_joint_probability->data[31][12] = 0.0529022278906f;
    edge_joint_probability->data[31][13] = 0.0141241536657f;
    edge_joint_probability->data[31][14] = 0.0155658878647f;
    edge_joint_probability->data[31][15] = 0.0416839662232f;
    edge_joint_probability->data[31][16] = 0.0178355178368f;
    edge_joint_probability->data[31][17] = 0.0473187250757f;
    edge_joint_probability->data[31][18] = 0.0492258002904f;
    edge_joint_probability->data[31][19] = 0.0327775444401f;
    edge_joint_probability->data[31][20] = 0.0150525446863f;
    edge_joint_probability->data[31][21] = 0.00477995241857f;
    edge_joint_probability->data[31][22] = 0.0296151042456f;
    edge_joint_probability->data[31][23] = 0.0487915714837f;
    edge_joint_probability->data[31][24] = 0.0500621843664f;
    edge_joint_probability->data[31][25] = 0.0293316657072f;
    edge_joint_probability->data[31][26] = 0.0310342948499f;
    edge_joint_probability->data[31][27] = 0.0131100831587f;
    edge_joint_probability->data[31][28] = 0.0490536318801f;
    edge_joint_probability->data[31][29] = 0.0548995328009f;
    edge_joint_probability->data[31][30] = 0.0174089628276f;
    edge_joint_probability->data[31][31] = 0.00189467027303f;
}