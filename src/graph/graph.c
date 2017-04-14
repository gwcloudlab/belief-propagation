#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "graph.h"

Graph_t
create_graph(unsigned int num_vertices, unsigned int num_edges)
{
	Graph_t g;

	g = (Graph_t)malloc(sizeof(struct graph));
	assert(g);
	g->edges_src_index = (unsigned int *)malloc(sizeof(unsigned int) * num_edges);
	assert(g->edges_src_index);
	g->edges_dest_index = (unsigned int *)malloc(sizeof(unsigned int) * num_edges);
	assert(g->edges_dest_index);
	g->edges_x_dim =(unsigned int *)malloc(sizeof(unsigned int) * num_edges);
	assert(g->edges_x_dim);
	g->edges_y_dim = (unsigned int *)malloc(sizeof(unsigned int) * num_edges);
	assert(g->edges_y_dim);
	g->edges_joint_probabilities = (float *)malloc(sizeof(float) * MAX_STATES * MAX_STATES * num_edges);
	assert(g->edges_joint_probabilities);
	g->edges_messages = (float *)malloc(sizeof(float) * MAX_STATES * num_edges);
	assert(g->edges_messages);
	g->last_edges_messages = (float *)malloc(sizeof(float) * MAX_STATES * num_edges);
	assert(g->last_edges_messages);
	g->node_states = (float *)malloc(sizeof(float) * num_vertices * MAX_STATES);
	assert(g->node_states);
	g->node_num_vars = (unsigned int *)malloc(sizeof(unsigned int) * num_vertices);
	assert(g->node_num_vars);
	g->src_nodes_to_edges = (unsigned int *)malloc(sizeof(unsigned int) * (num_vertices + num_edges));
	assert(g->src_nodes_to_edges);
	g->dest_nodes_to_edges = (unsigned int *)malloc(sizeof(unsigned int) * (num_vertices + num_edges));
	assert(g->dest_nodes_to_edges);
	g->node_names = (char *)malloc(sizeof(char) * CHAR_BUFFER_SIZE * num_vertices);
	assert(g->node_names);
	g->visited = (char *)calloc(sizeof(char), (size_t)num_vertices);
	assert(g->visited);
	g->observed_nodes = (char *)calloc(sizeof(char), (size_t)num_vertices);
	assert(g->observed_nodes);
	g->variable_names = (char *)calloc(sizeof(char), (size_t)num_vertices * CHAR_BUFFER_SIZE * MAX_STATES);
	assert(g->variable_names);
    g->levels_to_nodes = (unsigned int *)malloc(sizeof(unsigned int) * 2 * num_vertices);
    assert(g->levels_to_nodes != NULL);
	
	g->current_edge_messages = &g->edges_messages;
    g->previous_edge_messages = &g->last_edges_messages;
	
    g->hash_table_created = 0;
    g->num_levels = 0;
	g->total_num_vertices = num_vertices;
	g->total_num_edges = num_edges;
	g->current_num_vertices = 0;
	g->current_num_edges = 0;
    g->diameter = -1;
	return g;
}

void initialize_node(Graph_t graph, unsigned int node_index, unsigned int num_variables){
	unsigned int i;

	for(i = 0; i < num_variables; ++i){
		graph->node_states[node_index * MAX_STATES + i] = DEFAULT_STATE;
	}
	graph->node_num_vars[node_index] = num_variables;
}

void init_edge(Graph_t graph, unsigned int edge_index, unsigned int src_index, unsigned int dest_index, unsigned int dim_x,
			   unsigned int dim_y, float * joint_probabilities){
	int i, j;

	assert(src_index >= 0);
	assert(dest_index >= 0);
	assert(edge_index >= 0);

	assert(dim_x >= 0);
	assert(dim_y >= 0);
	assert(dim_x <= MAX_STATES);
	assert(dim_y <= MAX_STATES);

	graph->edges_src_index[edge_index] = src_index;
	graph->edges_dest_index[edge_index] = dest_index;
    graph->edges_x_dim[edge_index] = dim_x;
    graph->edges_y_dim[edge_index] = dim_y;

    for(i = 0; i < dim_x; ++i){
        for(j = 0; j < dim_y; ++j){
            graph->edges_joint_probabilities[MAX_STATES * MAX_STATES * edge_index + MAX_STATES * i + j] = joint_probabilities[MAX_STATES * i + j];
        }
		graph->edges_messages[MAX_STATES * edge_index + i] = 0;
		graph->last_edges_messages[MAX_STATES * edge_index + i] = 0;
    }
}

void node_set_state(Graph_t graph, unsigned int node_index, unsigned int num_variables, float * state){
	unsigned int i;

	for(i = 0; i < num_variables; ++i){
		graph->node_states[node_index * MAX_STATES + i] = state[i];
	}
}

void graph_add_node(Graph_t g, unsigned int num_variables, const char * name) {
	unsigned int node_index;

	node_index = g->current_num_vertices;

	initialize_node(g, node_index, num_variables);
	strncpy(&g->node_names[node_index * CHAR_BUFFER_SIZE], name, CHAR_BUFFER_SIZE);

	g->current_num_vertices += 1;
}

void graph_add_and_set_node_state(Graph_t g, unsigned int num_variables, const char * name, float * state){
	unsigned int node_index;

	node_index = g->current_num_vertices;

	g->observed_nodes[node_index] = 1;
	graph_add_node(g, num_variables, name);
	node_set_state(g, node_index, num_variables, state);
}

void graph_set_node_state(Graph_t g, unsigned int node_index, unsigned int num_states, float * state){

	assert(node_index < g->current_num_vertices);

	assert(num_states <= g->node_num_vars[node_index]);

	g->observed_nodes[node_index] = 1;

	node_set_state(g, node_index, num_states, state);
}

void graph_add_edge(Graph_t graph, unsigned int src_index, unsigned int dest_index, unsigned int dim_x, unsigned int dim_y, float * joint_probabilities) {
	unsigned int edge_index;

	edge_index = graph->current_num_edges;

	assert(graph->node_num_vars[src_index] == dim_x);
	assert(graph->node_num_vars[dest_index] == dim_y);

    init_edge(graph, edge_index, src_index, dest_index, dim_x, dim_y, joint_probabilities);


	graph->current_num_edges += 1;
}

void set_up_src_nodes_to_edges(Graph_t graph){
	unsigned int i, j, edge_index, num_vertices, num_edges;

	assert(graph->current_num_vertices == graph->total_num_vertices);
	assert(graph->current_num_edges <= graph->total_num_edges);

	edge_index = graph->total_num_vertices;

	num_vertices = graph->total_num_vertices;
	num_edges = graph->current_num_edges;

	for(i = 0; i < num_vertices; ++i){
		graph->src_nodes_to_edges[i] = edge_index;
		for(j = 0; j < num_edges; ++j){
			if(graph->edges_src_index[j] == i){
				graph->src_nodes_to_edges[edge_index] = j;
				edge_index += 1;
			}
		}
	}
}

void set_up_dest_nodes_to_edges(Graph_t graph){
	unsigned int i, j, edge_index, num_vertices, num_edges;

	assert(graph->current_num_vertices == graph->total_num_vertices);
	assert(graph->current_num_edges <= graph->total_num_edges);

	edge_index = graph->total_num_vertices;

	num_vertices = graph->total_num_vertices;
	num_edges = graph->current_num_edges;

	for(i = 0; i < num_vertices; ++i){
		graph->dest_nodes_to_edges[i] = edge_index;
		for(j = 0; j < num_edges; ++j){
			if(graph->edges_dest_index[j] == i){
				graph->dest_nodes_to_edges[edge_index] = j;
				edge_index += 1;
			}
		}

	}
}

int graph_vertex_count(Graph_t g) {
	return g->current_num_vertices;
}

int graph_edge_count(Graph_t g) {
	return g->current_num_edges;
}

void graph_destroy(Graph_t g) {
    if(g->hash_table_created != 0){
        hdestroy();
    }
	free(g->edges_src_index);
	free(g->edges_dest_index);
	free(g->edges_x_dim);
	free(g->edges_y_dim);
	free(g->edges_joint_probabilities);

	free(g->edges_messages);
	free(g->last_edges_messages);
	
	free(g->src_nodes_to_edges);
	free(g->dest_nodes_to_edges);
	free(g->node_names);
	free(g->visited);
	free(g->observed_nodes);
	free(g->variable_names);
	free(g->levels_to_nodes);
	free(g->node_num_vars);
	free(g->node_states);
	free(g);
}

void propagate_using_levels_start(Graph_t g){
	unsigned int i, j, k, node_index, edge_index, level_start_index, level_end_index, start_index, end_index, num_vertices;

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
		start_index = g->src_nodes_to_edges[node_index];
		if(node_index + 1 == num_vertices){
			end_index = num_vertices + g->current_num_edges;
		}
		else {
			end_index = g->src_nodes_to_edges[node_index + 1];
		}
		for(i = start_index; i < end_index; ++i){
			g->visited[node_index] = 1;
			edge_index = g->src_nodes_to_edges[i];

			send_message(g->node_states, MAX_STATES * node_index, edge_index, g->edges_joint_probabilities, g->edges_messages, g->edges_x_dim, g->edges_y_dim);

			/*printf("sending message on edge\n");
			print_edge(g, edge_index);
			printf("message: [");
			for(j = 0; j < g->node_num_vars[node_index]; ++j){
				printf("%.6lf\t", g->node_states[MAX_STATES * node_index + j]);
			}
			printf("]\n");



			printf("edge message is:\n[");
			for(j = 0; j < g->edges_x_dim[edge_index]; ++j){
				printf("%.6lf\t", g->edges_messages[MAX_STATES * edge_index + j]);
			}
			printf("]\n");*/
		}
	}
}

void send_message(float * states, unsigned int offset, unsigned int edge_index, float * edge_joint_probabilities, float * edge_messages, unsigned int * edge_num_src,
				  unsigned int * edge_num_dest){
	unsigned int i, j, num_src, num_dest;
	float sum;

	num_src = edge_num_src[edge_index];
	num_dest = edge_num_dest[edge_index];

	sum = 0.0;
	for(i = 0; i < num_src; ++i){
		edge_messages[MAX_STATES * edge_index + i] = 0.0;
		for(j = 0; j < num_dest; ++j){
			edge_messages[MAX_STATES * edge_index + i] += edge_joint_probabilities[MAX_STATES * MAX_STATES * edge_index + MAX_STATES * i + j] * states[offset + j];
		}
		sum += edge_messages[MAX_STATES * edge_index + i];
	}
	if(sum <= 0.0){
		sum = 1.0;
	}
	for (i = 0; i < num_src; ++i) {
		edge_messages[MAX_STATES * edge_index + i] = edge_messages[MAX_STATES * edge_index + i] / sum;
	}
}

#pragma acc routine
static inline void combine_message(float * dest, float * src, unsigned int length, unsigned int offset){
	unsigned int i;

	for(i = 0; i < length; ++i){
		if(src[offset + i] == src[offset + i]) { // ensure no nan's
			dest[i] = dest[i] * src[offset + i];
		}
	}
}

static void propagate_node_using_levels(Graph_t g, unsigned int current_node_index){
	float message_buffer[MAX_STATES];
	unsigned int i, j, num_variables, start_index, end_index, num_vertices, edge_index;
	unsigned int * dest_nodes_to_edges;
	unsigned int * src_nodes_to_edges;

	num_variables = g->node_num_vars[current_node_index];

	// mark as visited
	g->visited[current_node_index] = 1;

	num_vertices = g->current_num_vertices;
	dest_nodes_to_edges = g->dest_nodes_to_edges;
	src_nodes_to_edges = g->src_nodes_to_edges;

	// init buffer
	for(i = 0; i < num_variables; ++i){
		message_buffer[i] = 1.0;
	}

	// get the edges feeding into this node
	start_index = dest_nodes_to_edges[current_node_index];
	if(current_node_index + 1 == num_vertices){
		end_index = num_vertices + g->current_num_edges;
	}
	else{
		end_index = dest_nodes_to_edges[current_node_index + 1];
	}
	for(i = start_index; i < end_index; ++i){
		edge_index = dest_nodes_to_edges[i];

		combine_message(message_buffer, g->edges_messages, num_variables, MAX_STATES * edge_index);
	}

	//send message
	start_index = src_nodes_to_edges[current_node_index];
	if(current_node_index + 1 == num_vertices){
		end_index = num_vertices + g->current_num_edges;
	}
	else {
		end_index = src_nodes_to_edges[current_node_index + 1];
	}

	for(i = start_index; i < end_index; ++i){
		edge_index = src_nodes_to_edges[i];
		//ensure node hasn't been visited yet
		if(g->visited[g->edges_dest_index[edge_index]] == 0){
			/*printf("sending message on edge\n");
			print_edge(g, edge_index);
			printf("message: [");
			for(j = 0; j < num_variables; ++j){
				printf("%.6lf\t", message_buffer[j]);
			}
			printf("]\n");*/
			send_message(message_buffer, 0, edge_index, g->edges_joint_probabilities, g->edges_messages, g->edges_x_dim, g->edges_y_dim);
		}
	}
}

void propagate_using_levels(Graph_t g, unsigned int current_level) {
	unsigned int i, start_index, end_index;

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

static void marginalize_node(Graph_t g, unsigned int node_index){
	unsigned int i, num_variables, start_index, end_index, edge_index;
	char has_incoming;
	float sum;

	unsigned int * dest_nodes_to_edges;

	dest_nodes_to_edges = g->dest_nodes_to_edges;

	num_variables = g->node_num_vars[node_index];

	float new_message[MAX_STATES];
	for(i = 0; i < num_variables; ++i){
		new_message[i] = 1.0;
	}

	has_incoming = 0;

	start_index = dest_nodes_to_edges[node_index];
	if(node_index + 1 == g->current_num_vertices){
		end_index = g->current_num_vertices + g->current_num_edges;
	}
	else {
		end_index = dest_nodes_to_edges[node_index + 1];
	}

	for(i = start_index; i < end_index; ++i){
		edge_index = dest_nodes_to_edges[i];

		combine_message(new_message, g->edges_messages, num_variables, MAX_STATES * edge_index);
		has_incoming = 1;

	}
	if(has_incoming == 1){
		for(i = 0; i < num_variables; ++i){
			g->node_states[MAX_STATES * node_index + i] = new_message[i];
		}
	}
	sum = 0.0;
	for(i = 0; i < num_variables; ++i){
		sum += g->node_states[MAX_STATES * node_index + i];
	}
	if(sum <= 0.0){
		sum = 1.0;
	}

	for(i = 0; i < num_variables; ++i){
		g->node_states[MAX_STATES * node_index + i] = g->node_states[MAX_STATES * node_index + i] / sum;
	}
}

void marginalize(Graph_t g){
	unsigned int i, num_nodes;

	num_nodes = g->current_num_vertices;

	for(i = 0; i < num_nodes; ++i){
		marginalize_node(g, i);
	}
}

void reset_visited(Graph_t g){
	unsigned int i, num_nodes;

	num_nodes = g->current_num_vertices;
	for(i = 0; i < num_nodes; ++i){
		g->visited[i] = 0;
	}
}


void print_node(Graph_t graph, unsigned int node_index){
	unsigned int i, num_vars, variable_name_index;

	num_vars = graph->node_num_vars[node_index];

	printf("Node %s [\n", &graph->node_names[node_index * CHAR_BUFFER_SIZE]);
	for(i = 0; i < num_vars; ++i){
		variable_name_index = node_index * CHAR_BUFFER_SIZE * MAX_STATES + i * CHAR_BUFFER_SIZE;
		printf("%s:\t%.6lf\n", &graph->variable_names[variable_name_index], graph->node_states[MAX_STATES * node_index + i]);
	}
	printf("]\n");
}

void print_edge(Graph_t graph, unsigned int edge_index){
	unsigned int i, j, dim_x, dim_y, src_index, dest_index;


	dim_x = graph->edges_x_dim[edge_index];
	dim_y = graph->edges_y_dim[edge_index];
	src_index = graph->edges_src_index[edge_index];
	dest_index = graph->edges_dest_index[edge_index];

	printf("Edge  %s -> %s [\n", &graph->node_names[src_index * CHAR_BUFFER_SIZE], &graph->node_names[dest_index * CHAR_BUFFER_SIZE]);
	printf("Joint probability matrix: [\n");
	for(i = 0; i < dim_x; ++i){
		printf("[");
		for(j = 0; j < dim_y; ++j){
			printf("\t%.6lf",  graph->edges_joint_probabilities[MAX_STATES * MAX_STATES * edge_index + MAX_STATES * i + j]);
		}
		printf("\t]\n");
	}
	printf("]\nMessage:\n[");
	for(i = 0; i < dim_x; ++i){
		printf("\t%.6lf", graph->edges_messages[MAX_STATES * edge_index + i]);
	}
	printf("\t]\n]\n");
}

void print_nodes(Graph_t g){
	unsigned int i, num_nodes;

	num_nodes = g->current_num_vertices;

	for(i = 0; i < num_nodes; ++i){
		print_node(g, i);
	}
}

void print_edges(Graph_t g){
	unsigned int i, num_edges;

	num_edges = g->current_num_edges;

	for(i = 0; i < num_edges; ++i){
		print_edge(g, i);
	}
}
void print_src_nodes_to_edges(Graph_t g){
	unsigned int i, j, start_index, end_index, num_vertices, edge_index;
	unsigned int * src_node_to_edges;

	printf("src index -> edge index\n");


	src_node_to_edges = g->src_nodes_to_edges;
	num_vertices = g->total_num_vertices;

	for(i = 0; i < num_vertices; ++i){
		printf("Node -----\n");
		print_node(g, i);
		printf("Edges-------\n");
		start_index = src_node_to_edges[i];
		if(i + 1 == num_vertices){
			end_index = num_vertices + g->current_num_edges;
		}
		else{
			end_index = src_node_to_edges[i+1];
		}
		for(j = start_index; j < end_index; ++j){
			edge_index = src_node_to_edges[j];
			print_edge(g, edge_index);
		}
		printf("---------\n");
	}
}
void print_dest_nodes_to_edges(Graph_t g){
	unsigned int i, j, start_index, end_index, num_vertices, edge_index;
	unsigned int * dest_node_to_edges;

	printf("dest index -> edge index\n");

	dest_node_to_edges = g->dest_nodes_to_edges;
	num_vertices = g->total_num_vertices;

	for(i = 0; i < num_vertices; ++i){
		printf("Node -----\n");
		print_node(g, i);
		printf("Edges-------\n");
		start_index = dest_node_to_edges[i];
		if(i + 1 == num_vertices){
			end_index = num_vertices + g->current_num_edges;
		}
		else{
			end_index = dest_node_to_edges[i+1];
		}
		for(j = start_index; j < end_index; ++j){
			edge_index = dest_node_to_edges[j];
			print_edge(g, edge_index);
		}
		printf("---------\n");
	}
}

void init_previous_edge(Graph_t graph){
	unsigned int i, j, num_vertices, start_index, end_index, edge_index;
	unsigned int * src_node_to_edges;
	float * previous_messages;

	num_vertices = graph->current_num_vertices;
	src_node_to_edges = graph->src_nodes_to_edges;
	previous_messages = *graph->previous_edge_messages;

	for(i = 0; i < num_vertices; ++i){
		start_index = src_node_to_edges[i];
		if(i + 1 >= num_vertices){
			end_index = num_vertices + graph->current_num_edges;
		}
		else
		{
			end_index = src_node_to_edges[i + 1];
		}
		for(j = start_index; j < end_index; ++j){
			edge_index = src_node_to_edges[j];

			send_message(graph->node_states, MAX_STATES * i, edge_index, graph->edges_joint_probabilities, previous_messages, graph->edges_x_dim, graph->edges_y_dim);
		}
	}
}

void fill_in_leaf_nodes_in_index(Graph_t graph, unsigned int * start_index, unsigned int * end_index, unsigned int max_count){
	unsigned int i, diff, edge_start_index, edge_end_index;

    graph->levels_to_nodes[0] = *start_index;
    for(i = 0; i < graph->current_num_vertices; ++i){
        edge_start_index = graph->dest_nodes_to_edges[i];
        if(i + 1 == graph->current_num_vertices){
            edge_end_index = graph->current_num_vertices + graph->current_num_edges;
        }
        else{
            edge_end_index = graph->dest_nodes_to_edges[i + 1];
        }
		diff = edge_end_index - edge_start_index;

        if(diff <= max_count){
            graph->levels_to_nodes[*end_index] = i;
            *end_index += 1;
        }
    }
}

void visit_node(Graph_t graph, unsigned int buffer_index, unsigned int * end_index){
	unsigned int node_index, edge_start_index, edge_end_index, edge_index, i, j, dest_node_index;
	char visited;

    node_index = graph->levels_to_nodes[buffer_index];
    if(graph->visited[node_index] == 0){
        graph->visited[node_index] = 1;
        edge_start_index = graph->src_nodes_to_edges[node_index];
        if(node_index == graph->current_num_vertices){
            edge_end_index = graph->current_num_vertices + graph->current_num_edges;
        }
        else{
            edge_end_index = graph->src_nodes_to_edges[node_index + 1];
        }
        for(i = edge_start_index; i < edge_end_index; ++i){
			edge_index = graph->src_nodes_to_edges[i];
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

void init_levels_to_nodes(Graph_t graph){
	unsigned int start_index, end_index, copy_end_index, i;

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

void print_levels_to_nodes(Graph_t graph){
	unsigned int i, j, start_index, end_index;

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

#pragma acc routine
static void initialize_message_buffer(float * message_buffer, float * node_states, unsigned int node_index, unsigned int num_variables){
	unsigned int j;

	//clear buffer
	for(j = 0; j < num_variables; ++j){
		message_buffer[j] = node_states[MAX_STATES * node_index + j];
	}
}

#pragma acc routine
static void read_incoming_messages(float * message_buffer, unsigned int * dest_node_to_edges, float * previous_messages,
                                   unsigned int current_num_edges, unsigned int num_vertices,
								   unsigned int num_variables, unsigned int i){
	unsigned int start_index, end_index, j, edge_index;

	start_index = dest_node_to_edges[i];
	if(i + 1 >= num_vertices){
		end_index = num_vertices + current_num_edges;
	}
	else{
		end_index = dest_node_to_edges[i + 1];
	}

	for(j = start_index; j < end_index; ++j){
		edge_index = dest_node_to_edges[j];

		combine_message(message_buffer, previous_messages, num_variables, MAX_STATES * edge_index);
	}
}

#pragma acc routine
static void send_message_for_edge(float * buffer, unsigned int edge_index,
								  float * joint_probabilities, float * edge_messages, unsigned int * dim_src,
								  unsigned int * dim_dest) {
	unsigned int i, j, num_src, num_dest;
	float sum;

	num_src = dim_src[edge_index];
	num_dest = dim_dest[edge_index];


	sum = 0.0;
	for(i = 0; i < num_src; ++i){
		edge_messages[edge_index * MAX_STATES + i] = 0.0;
		for(j = 0; j < num_dest; ++j){
			edge_messages[edge_index * MAX_STATES + i] += joint_probabilities[MAX_STATES * MAX_STATES * edge_index + MAX_STATES * i + j] * buffer[j];
		}
		sum += edge_messages[edge_index * MAX_STATES + i];
	}
	if(sum <= 0.0){
		sum = 1.0;
	}
	for (i = 0; i < num_src; ++i) {
		edge_messages[edge_index * MAX_STATES + i] = edge_messages[edge_index * MAX_STATES + i] / sum;
	}
}

#pragma acc routine
static void send_message_for_node(unsigned int * src_node_to_edges, float * message_buffer, unsigned int current_num_edges,
								  float * joint_probabilities, float * edge_messages,
								  unsigned int * num_src, unsigned int * num_dest,
								  unsigned int num_vertices, unsigned int i){
	unsigned int start_index, end_index, j, edge_index;

	start_index = src_node_to_edges[i];
	if(i + 1 >= num_vertices){
		end_index = num_vertices + current_num_edges;
	}
	else {
		end_index = src_node_to_edges[i + 1];
	}
    
	for(j = start_index; j < end_index; ++j){
		edge_index = src_node_to_edges[j];
		/*printf("Sending on edge\n");
        print_edge(graph, edge_index);*/
		send_message_for_edge(message_buffer, edge_index, joint_probabilities, edge_messages, num_src, num_dest);
	}
}

static void marginalize_loopy_nodes(Graph_t graph, float * current_messages, unsigned int num_vertices) {
	unsigned int j;

	unsigned int i, num_variables, start_index, end_index, edge_index, current_num_vertices, current_num_edges;
	char has_incoming;
	float sum;
	float * states;
	unsigned int * num_vars;
	float new_message[MAX_STATES];

	unsigned int * dest_nodes_to_edges;

	dest_nodes_to_edges = graph->dest_nodes_to_edges;
	current_num_vertices = graph->current_num_vertices;
	current_num_edges = graph->current_num_edges;
	states = graph->node_states;
	num_vars = graph->node_num_vars;


#pragma omp parallel for default(none) shared(states, num_vars, num_vertices, current_num_vertices, current_num_edges, dest_nodes_to_edges, current_messages) private(i, j, num_variables, start_index, end_index, edge_index, has_incoming, sum, new_message)
	for(j = 0; j < num_vertices; ++j) {
		has_incoming = 0;

		num_variables = num_vars[j];


		for (i = 0; i < num_variables; ++i) {
			new_message[i] = 1.0;
		}

		has_incoming = 0;

		start_index = dest_nodes_to_edges[j];
		if (j + 1 == current_num_vertices) {
			end_index = current_num_vertices + current_num_edges;
		} else {
			end_index = dest_nodes_to_edges[j + 1];
		}

		for (i = start_index; i < end_index; ++i) {
			edge_index = dest_nodes_to_edges[i];

			combine_message(new_message, current_messages, num_variables, MAX_STATES * edge_index);
			has_incoming = 1;

		}
		if (has_incoming == 1) {
			for (i = 0; i < num_variables; ++i) {
				states[MAX_STATES * j + i] = new_message[i];
			}
		}
		sum = 0.0;
		for (i = 0; i < num_variables; ++i) {
			sum += states[MAX_STATES * j + i];
		}
		if (sum <= 0.0) {
			sum = 1.0;
		}

		for (i = 0; i < num_variables; ++i) {
			states[MAX_STATES * j + i] = states[MAX_STATES * j + i] / sum;
		}
	}

/*
#pragma omp parallel for default(none) shared(graph, num_vertices, current) private(i)
	for(i = 0; i < num_vertices; ++i){
		marginalize_node(graph, i, current);
	}*/

}

#pragma acc routine
static void marginalize_node_acc(float * node_states, unsigned int * num_vars, unsigned int node_index,
								 float * edge_messages, unsigned int * dest_nodes_to_edges,
								 unsigned int current_num_vertices, unsigned int current_num_edges){
	unsigned int i, num_variables, start_index, end_index, edge_index;
	char has_incoming;
	float sum;

	num_variables = num_vars[node_index];

	float new_message[MAX_STATES];
	for(i = 0; i < num_variables; ++i){
		new_message[i] = 1.0;
	}

	has_incoming = 0;

	start_index = dest_nodes_to_edges[node_index];
	if(node_index + 1 == current_num_vertices){
		end_index = current_num_vertices + current_num_edges;
	}
	else {
		end_index = dest_nodes_to_edges[node_index + 1];
	}

	for(i = start_index; i < end_index; ++i){
		edge_index = dest_nodes_to_edges[i];

		combine_message(new_message, edge_messages, num_variables, MAX_STATES * edge_index);
		has_incoming = 1;

	}
	if(has_incoming == 1){
		for(i = 0; i < num_variables; ++i){
			node_states[MAX_STATES * node_index + i] = new_message[i];
		}
	}
	sum = 0.0;
	for(i = 0; i < num_variables; ++i){
		sum += node_states[MAX_STATES * node_index + i];
	}
	if(sum <= 0.0){
		sum = 1.0;
	}

	for(i = 0; i < num_variables; ++i){
		node_states[MAX_STATES * node_index + i] = node_states[MAX_STATES * node_index + i] / sum;
	}
}

static void marginalize_nodes_acc(float * node_states, unsigned int * num_vars, float * edge_messages, unsigned int * dest_nodes_to_edges,
								  unsigned int current_num_vertices, unsigned int current_num_edges){
	unsigned int i;

#pragma omp parallel for default(none) shared(node_states, num_vars, edge_messages, dest_nodes_to_edges, current_num_vertices, current_num_edges) private(i)
	for(i = 0; i < current_num_vertices; ++i){
		marginalize_node_acc(node_states, num_vars, i, edge_messages, dest_nodes_to_edges, current_num_vertices, current_num_edges);
	}
}

void loopy_propagate_one_iteration(Graph_t graph){
	unsigned int i, num_variables, num_vertices, num_edges;
	unsigned int * dest_node_to_edges;
	unsigned int * src_node_to_edges;
	unsigned int * num_vars;
	float * node_states;
	float * joint_probabilities;
	float * previous_edge_messages;
	float * current_edge_messages;
	unsigned int * num_src;
	unsigned int * num_dest;
	float ** temp;

	previous_edge_messages = *graph->previous_edge_messages;
	current_edge_messages = *graph->current_edge_messages;
	joint_probabilities = graph->edges_joint_probabilities;
	num_src = graph->edges_x_dim;
	num_dest = graph->edges_y_dim;

	float message_buffer[MAX_STATES];

	num_vertices = graph->current_num_vertices;
	dest_node_to_edges = graph->dest_nodes_to_edges;
	src_node_to_edges = graph->src_nodes_to_edges;
    num_edges = graph->current_num_edges;
	num_vars = graph->node_num_vars;
	node_states = graph->node_states;

#pragma omp parallel for default(none) shared(node_states, num_vars, num_vertices, dest_node_to_edges, src_node_to_edges, num_edges, previous_edge_messages, num_dest, num_src, current_edge_messages, joint_probabilities) private(message_buffer, i, num_variables)
    for(i = 0; i < num_vertices; ++i){
		num_variables = num_vars[i];

		initialize_message_buffer(message_buffer, node_states, i, num_variables);

		//read incoming messages
		read_incoming_messages(message_buffer, dest_node_to_edges, previous_edge_messages, num_edges, num_vertices, num_variables, i);

/*
		printf("Message at node\n");
		print_node(graph, i);
		printf("[\t");
		for(j = 0; j < num_variables; ++j){
			printf("%.6lf\t", message_buffer[j]);
		}
		printf("\t]\n");*/


		//send message
		send_message_for_node(src_node_to_edges, message_buffer, num_edges, joint_probabilities, current_edge_messages, num_src, num_dest, num_vertices, i);

	}

	marginalize_loopy_nodes(graph, current_edge_messages, num_vertices);

	//swap previous and current
	temp = graph->previous_edge_messages;
	graph->previous_edge_messages = graph->current_edge_messages;
	graph->current_edge_messages = temp;
}

unsigned int loopy_propagate_until(Graph_t graph, float convergence, unsigned int max_iterations){
	unsigned int i, j, k, num_edges;
	float delta, diff, previous_delta;
	float * previous_edge_messages;
	float * current_edge_messages;
	unsigned int * edges_x_dim;

	previous_edge_messages = *graph->previous_edge_messages;
	current_edge_messages = *graph->current_edge_messages;
	edges_x_dim = graph->edges_x_dim;

	num_edges = graph->current_num_edges;

	previous_delta = -1.0;
	delta = 0.0;

	for(i = 0; i < max_iterations; ++i){
		//printf("Current iteration: %d\n", i+1);
		loopy_propagate_one_iteration(graph);

		delta = 0.0;

#pragma omp parallel default(none) shared(previous_edge_messages, current_edge_messages, num_edges, edges_x_dim)  private(j, diff, k) reduction(+:delta)
		for(j = 0; j < num_edges; ++j){
			for(k = 0; k < edges_x_dim[j]; ++k){
				diff = previous_edge_messages[j * MAX_STATES + k] - current_edge_messages[j * MAX_STATES + k];
				if(diff != diff){
					diff = 0.0;
				}
				delta += fabs(diff);
			}
		}

		//printf("Current delta: %.6lf\n", delta);
		//printf("Previous delta: %.6lf\n", previous_delta);
		if(delta < convergence || fabs(delta - previous_delta) < convergence){
			break;
		}
		previous_delta = delta;
	}
	if(i == max_iterations){
		printf("No Convergence: previous: %f vs current: %f\n", previous_delta, delta);
	}
	return i;
}

static unsigned int loopy_propagate_iterations_acc(unsigned int num_vertices, unsigned int num_edges,
										   unsigned int *dest_node_to_edges, unsigned int *src_node_to_edges,
										   float * node_states, unsigned int * num_vars,
										   float ** previous_messages, float ** current_messages,
										   float * joint_probabilities, unsigned int * num_src, unsigned int * num_dest,
										   unsigned int max_iterations,
										   float convergence){
	unsigned int i, j, k, num_variables, num_iter;
	float delta, previous_delta, diff;
	float * prev_messages;
	float * curr_messages;
	float * temp;

	prev_messages = *previous_messages;
	curr_messages = *current_messages;

	float message_buffer[MAX_STATES];

	num_iter = 0;

	previous_delta = -1.0f;
	delta = 0.0f;

	for(i = 0; i < max_iterations; i+= BATCH_SIZE){
#pragma acc data present_or_copy(node_states[0:(MAX_STATES * num_vertices)], prev_messages[0:(MAX_STATES * num_edges)], curr_messages[0:(MAX_STATES * num_edges)]) present_or_copyin(dest_node_to_edges[0:(num_vertices + num_edges)], src_node_to_edges[0:(num_vertices + num_edges)], num_vars[0:num_vertices], joint_probabilities[0:(num_edges * MAX_STATES * MAX_STATES)], num_src[0:num_edges], num_dest[0:num_edges])
        {
            //printf("Current iteration: %d\n", i+1);
            for (j = 0; j < BATCH_SIZE; ++j) {
#pragma acc kernels
                for (k = 0; k < num_vertices; ++k) {
                    num_variables = num_vars[k];

                    initialize_message_buffer(message_buffer, node_states, k, num_variables);

                    //read incoming messages
                    read_incoming_messages(message_buffer, dest_node_to_edges, prev_messages, num_edges, num_vertices,
                                           num_variables, k);

/*
		printf("Message at node\n");
		print_node(graph, i);
		printf("[\t");
		for(j = 0; j < num_variables; ++j){
			printf("%.6lf\t", message_buffer[j]);
		}
		printf("\t]\n");*/


                    //send message
                    send_message_for_node(src_node_to_edges, message_buffer, num_edges, joint_probabilities, curr_messages, num_src, num_dest, num_vertices, k);

                }

#pragma acc kernels
                for (k = 0; k < num_vertices; ++k) {
                    marginalize_node_acc(node_states, num_vars, k, curr_messages, dest_node_to_edges, num_vertices, num_edges);
                }

                //swap previous and current
                temp = prev_messages;
                prev_messages = curr_messages;
                curr_messages = temp;

            }


            delta = 0.0f;
#pragma acc kernels create(diff)
            for (j = 0; j < num_edges; ++j) {
                for (k = 0; k < num_src[j]; ++k) {
                    diff = prev_messages[MAX_STATES * j + k] - curr_messages[MAX_STATES * j + k];
                    if (diff != diff) {
                        diff = 0.0f;
                    }
                    delta += fabs(diff);
                }
            }
        }

		num_iter += BATCH_SIZE;

		//printf("Current delta: %.6lf\n", delta);
		//printf("Previous delta: %.6lf\n", previous_delta);
		if(delta < convergence || fabs(delta - previous_delta) < convergence){
			break;
		}
		previous_delta = delta;
	}
	if(i == max_iterations) {
		printf("No Convergence: previous: %f vs current: %f\n", previous_delta, delta);
	}


	return num_iter;
}

unsigned int loopy_progagate_until_acc(Graph_t graph, float convergence, unsigned int max_iterations){
	unsigned int iter;

	/*printf("===BEFORE====\n");
	print_nodes(graph);
	print_edges(graph);
*/
	iter = loopy_propagate_iterations_acc(graph->current_num_vertices, graph->current_num_edges,
	graph->dest_nodes_to_edges, graph->src_nodes_to_edges,
	graph->node_states, graph->node_num_vars,
	graph->previous_edge_messages, graph->current_edge_messages, graph->edges_joint_probabilities,
										  graph->edges_x_dim, graph->edges_y_dim,
										  max_iterations, convergence);

	/*printf("===AFTER====\n");
	print_nodes(graph);
	print_edges(graph);*/

	return iter;
}

void calculate_diameter(Graph_t graph){
    // calculate diameter using floyd-warshall
    int ** dist;
	int ** g;
    unsigned int i, j, k, start_index, end_index;
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
		 start_index = graph->src_nodes_to_edges[i];
		if(i + 1 == graph->current_num_vertices){
			end_index = graph->current_num_vertices + graph->current_num_edges;
		}
		else{
			end_index = graph->src_nodes_to_edges[i+1];
		}
		for(j = start_index; j < end_index; ++j){
			k = graph->src_nodes_to_edges[j];
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
