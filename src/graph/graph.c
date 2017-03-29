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
	g->edges = (Edge_t)malloc(sizeof(struct edge) * num_edges);
	assert(g->edges);
	g->prev_edges = (Edge_t)malloc(sizeof(struct edge) * num_edges);
	assert(g->prev_edges);
	g->nodes = (Node_t)malloc(sizeof(struct node) * num_vertices);
	assert(g->nodes);
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
    g->hash_table_created = 0;
    g->num_levels = 0;
	g->total_num_vertices = num_vertices;
	g->total_num_edges = num_edges;
	g->current_num_vertices = 0;
	g->current_num_edges = 0;
	g->previous = &g->prev_edges;
	g->current = &g->edges;
    g->diameter = -1;
	return g;
}

void graph_add_node(Graph_t g, unsigned int num_variables, const char * name) {
	unsigned int node_index;

	node_index = g->current_num_vertices;

	initialize_node(&g->nodes[node_index], node_index, num_variables);
	strncpy(&g->node_names[node_index * CHAR_BUFFER_SIZE], name, CHAR_BUFFER_SIZE);

	g->current_num_vertices += 1;
}

void graph_add_and_set_node_state(Graph_t g, unsigned int num_variables, const char * name, double * state){
	unsigned int node_index;

	node_index = g->current_num_vertices;

	g->observed_nodes[node_index] = 1;
	graph_add_node(g, num_variables, name);
	node_set_state(&g->nodes[node_index], num_variables, state);
}

void graph_set_node_state(Graph_t g, unsigned int node_index, unsigned int num_states, double * state){
	Node_t node;

	assert(node_index < g->current_num_vertices);

	node = &g->nodes[node_index];

	assert(num_states <= node->num_variables);

	g->observed_nodes[node_index] = 1;

	node_set_state(node, num_states, state);
}

void graph_add_edge(Graph_t graph, unsigned int src_index, unsigned int dest_index, unsigned int dim_x, unsigned int dim_y, double ** joint_probabilities) {
	unsigned int edge_index;

	edge_index = graph->current_num_edges;

	assert(graph->nodes[src_index].num_variables == dim_x);
	assert(graph->nodes[dest_index].num_variables == dim_y);


	init_edge(&graph->edges[edge_index], edge_index, src_index, dest_index, dim_x, dim_y, joint_probabilities);
	init_edge(&graph->prev_edges[edge_index], edge_index, src_index, dest_index, dim_x, dim_y, joint_probabilities);

	graph->current_num_edges += 1;
}

void set_up_src_nodes_to_edges(Graph_t graph){
	unsigned int i, j, edge_index, num_vertices, num_edges;
	Edge_t edge = NULL;

	assert(graph->current_num_vertices == graph->total_num_vertices);
	assert(graph->current_num_edges <= graph->total_num_edges);

	edge_index = graph->total_num_vertices;

	num_vertices = graph->total_num_vertices;
	num_edges = graph->current_num_edges;

	for(i = 0; i < num_vertices; ++i){
		graph->src_nodes_to_edges[i] = edge_index;
		for(j = 0; j < num_edges; ++j){
			edge = &graph->edges[j];

			if(edge->src_index == i){
				graph->src_nodes_to_edges[edge_index] = edge->edge_index;
				edge_index += 1;
			}
		}
	}
}

void set_up_dest_nodes_to_edges(Graph_t graph){
	unsigned int i, j, edge_index, num_vertices, num_edges;
	Edge_t edge = NULL;
	Node_t node = NULL;

	assert(graph->current_num_vertices == graph->total_num_vertices);
	assert(graph->current_num_edges <= graph->total_num_edges);

	edge_index = graph->total_num_vertices;

	num_vertices = graph->total_num_vertices;
	num_edges = graph->current_num_edges;

	for(i = 0; i < num_vertices; ++i){
		graph->dest_nodes_to_edges[i] = edge_index;
		for(j = 0; j < num_edges; ++j){
			edge = &graph->edges[j];

			if(edge->dest_index == i){
				graph->dest_nodes_to_edges[edge_index] = edge->edge_index;
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
	free(g->edges);
	free(g->prev_edges);
	free(g->nodes);
	free(g->src_nodes_to_edges);
	free(g->dest_nodes_to_edges);
	free(g->node_names);
	free(g->visited);
	free(g->observed_nodes);
	free(g->variable_names);
	free(g->levels_to_nodes);
	free(g);
}

void propagate_using_levels_start(Graph_t g){
	unsigned int i, j, k, node_index, edge_index, level_start_index, level_end_index, start_index, end_index, num_vertices;
	Node_t node;
	Edge_t edge;

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
		node = &g->nodes[node_index];
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
			edge = &g->edges[edge_index];
			/*
			printf("sending message on edge\n");
			print_edge(g, edge_index);
			printf("message: [");
			for(j = 0; j < node->num_variables; ++j){
				printf("%.6lf\t", node->states[j]);
			}
			printf("]\n");
			*/
			send_message(edge, node->states);
		}
	}
}

#pragma acc routine
static inline void combine_message(double * dest, Edge_t src_edge, unsigned int length){
	unsigned int i;
	double * src;

	src = src_edge->message;

	for(i = 0; i < length; ++i){
		if(src[i] == src[i]) { // ensure no nan's
			dest[i] = dest[i] * src[i];
		}
	}
}

static void propagate_node_using_levels(Graph_t g, unsigned int current_node_index){
	double message_buffer[MAX_STATES];
	unsigned int i, j, num_variables, start_index, end_index, num_vertices, edge_index;
	Node_t node;
	Edge_t edge;
	unsigned int * dest_nodes_to_edges;
	unsigned int * src_nodes_to_edges;

	node = &g->nodes[current_node_index];
	num_variables = node->num_variables;

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
		edge = &g->edges[edge_index];

		combine_message(message_buffer, edge, num_variables);
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
		edge = &g->edges[edge_index];
		//ensure node hasn't been visited yet
		if(g->visited[edge->dest_index] == 0){
			/*
			printf("sending message on edge\n");
			print_edge(g, edge_index);
			printf("message: [");
			for(j = 0; j < num_variables; ++j){
				printf("%.6lf\t", message_buffer[j]);
			}
			printf("]\n");
			 */
			send_message(edge, message_buffer);
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

static void marginalize_node(Graph_t g, unsigned int node_index, Edge_t edges){
	unsigned int i, num_variables, start_index, end_index, edge_index;
	char has_incoming;
	Edge_t edge;
	Node_t node;
	double sum;

	unsigned int * dest_nodes_to_edges;

	dest_nodes_to_edges = g->dest_nodes_to_edges;

	has_incoming = 0;

	node = &g->nodes[node_index];
	num_variables = node->num_variables;

	double new_message[MAX_STATES];
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
		edge = &edges[edge_index];

		combine_message(new_message, edge, num_variables);
		has_incoming = 1;

	}
	if(has_incoming == 1){
		for(i = 0; i < num_variables; ++i){
			node->states[i] = new_message[i];
		}
	}
	sum = 0.0;
	for(i = 0; i < num_variables; ++i){
		sum += node->states[i];
	}
	if(sum <= 0.0){
		sum = 1.0;
	}

	for(i = 0; i < num_variables; ++i){
		node->states[i] = node->states[i] / sum;
	}
}

void marginalize(Graph_t g){
	unsigned int i, num_nodes;
	Edge_t edges;

	num_nodes = g->current_num_vertices;
	edges = g->edges;

	for(i = 0; i < num_nodes; ++i){
		marginalize_node(g, i, edges);
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
	double * states;
	Node_t n;

	n = &graph->nodes[node_index];
	num_vars = n->num_variables;
	states = n->states;

	printf("Node %s [\n", &graph->node_names[node_index * CHAR_BUFFER_SIZE]);
	for(i = 0; i < num_vars; ++i){
		variable_name_index = node_index * CHAR_BUFFER_SIZE * MAX_STATES + i * CHAR_BUFFER_SIZE;
		printf("%s:\t%.6lf\n", &graph->variable_names[variable_name_index], states[i]);
	}
	printf("]\n");
}

void print_edge(Graph_t graph, unsigned int edge_index){
	unsigned int i, j, dim_x, dim_y;
	Edge_t e;

	e = &graph->edges[edge_index];
	dim_x = e->x_dim;
	dim_y = e->y_dim;

	printf("Edge  %s -> %s [\n", &graph->node_names[e->src_index * CHAR_BUFFER_SIZE], &graph->node_names[e->dest_index * CHAR_BUFFER_SIZE]);
	for(i = 0; i < dim_x; ++i){
		printf("[");
		for(j = 0; j < dim_y; ++j){
			printf("\t%.6lf",  e->joint_probabilities[i][j]);
		}
		printf("\t]\n");
	}
	printf("\n");
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
	Edge_t edge, previous;
	Node_t node;

	num_vertices = graph->current_num_vertices;
	src_node_to_edges = graph->src_nodes_to_edges;
	previous = *graph->previous;

	for(i = 0; i < num_vertices; ++i){
		node = &graph->nodes[i];
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

			edge = &previous[edge_index];

			send_message(edge, node->states);
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
    Edge_t edge;
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
            edge = &graph->edges[edge_index];
            dest_node_index = edge->dest_index;
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
static void initialize_message_buffer(double * message_buffer, Node_t node, unsigned int num_variables){
	unsigned int j;

	//clear buffer
	for(j = 0; j < num_variables; ++j){
		message_buffer[j] = node->states[j];
	}
}

#pragma acc routine
static void read_incoming_messages(double * message_buffer, unsigned int * dest_node_to_edges, Edge_t previous,
                                   unsigned int current_num_edges, unsigned int num_vertices,
								   unsigned int num_variables, unsigned int i){
	unsigned int start_index, end_index, j, edge_index;
	Edge_t edge;

	start_index = dest_node_to_edges[i];
	if(i + 1 >= num_vertices){
		end_index = num_vertices + current_num_edges;
	}
	else{
		end_index = dest_node_to_edges[i + 1];
	}

	for(j = start_index; j < end_index; ++j){
		edge_index = dest_node_to_edges[j];
		edge = &previous[edge_index];

		combine_message(message_buffer, edge, num_variables);
	}
}

#pragma acc routine
static void send_message_for_edge(Edge_t edge, double * message) {
	unsigned int i, j, num_src, num_dest;
	double sum;

	num_src = edge->x_dim;
	num_dest = edge->y_dim;


	sum = 0.0;
	for(i = 0; i < num_src; ++i){
		edge->message[i] = 0.0;
		for(j = 0; j < num_dest; ++j){
			edge->message[i] += edge->joint_probabilities[i][j] * message[j];
		}
		sum += edge->message[i];
	}
	if(sum <= 0.0){
		sum = 1.0;
	}
	for (i = 0; i < num_src; ++i) {
		edge->message[i] = edge->message[i] / sum;
	}
}

#pragma acc routine
static void send_message_for_node(unsigned int * src_node_to_edges, double * message_buffer, unsigned int current_num_edges,
								  Edge_t current, unsigned int num_vertices, unsigned int i){
	unsigned int start_index, end_index, j, edge_index;
	Edge_t edge;

	start_index = src_node_to_edges[i];
	if(i + 1 >= num_vertices){
		end_index = num_vertices + current_num_edges;
	}
	else {
		end_index = src_node_to_edges[i + 1];
	}
    
	for(j = start_index; j < end_index; ++j){
		edge_index = src_node_to_edges[j];
		edge = &current[edge_index];
		/*printf("Sending on edge\n");
        print_edge(graph, edge_index);*/
		send_message_for_edge(edge, message_buffer);
	}
}

static void marginalize_loopy_nodes(Graph_t graph, Edge_t current, unsigned int num_vertices) {
	unsigned int j;

	unsigned int i, num_variables, start_index, end_index, edge_index, current_num_vertices, current_num_edges;
	char has_incoming;
	Edge_t edge, edges;
	Node_t node, nodes;
	double sum;
	double new_message[MAX_STATES];

	unsigned int * dest_nodes_to_edges;

	dest_nodes_to_edges = graph->dest_nodes_to_edges;
	nodes = graph->nodes;
	current_num_vertices = graph->current_num_vertices;
	current_num_edges = graph->current_num_edges;
	edges = graph->edges;

#pragma omp parallel for default(none) shared(num_vertices, current_num_vertices, current_num_edges, dest_nodes_to_edges, nodes, edges) private(i, j, node, num_variables, start_index, end_index, edge_index, has_incoming, edge, sum, new_message)
	for(j = 0; j < num_vertices; ++j) {
		has_incoming = 0;

		node = &nodes[j];
		num_variables = node->num_variables;


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
			edge = &edges[edge_index];

			combine_message(new_message, edge, num_variables);
			has_incoming = 1;

		}
		if (has_incoming == 1) {
			for (i = 0; i < num_variables; ++i) {
				node->states[i] = new_message[i];
			}
		}
		sum = 0.0;
		for (i = 0; i < num_variables; ++i) {
			sum += node->states[i];
		}
		if (sum <= 0.0) {
			sum = 1.0;
		}

		for (i = 0; i < num_variables; ++i) {
			node->states[i] = node->states[i] / sum;
		}
	}

/*
#pragma omp parallel for default(none) shared(graph, num_vertices, current) private(i)
	for(i = 0; i < num_vertices; ++i){
		marginalize_node(graph, i, current);
	}*/

}

#pragma acc routine
static void marginalize_node_acc(Node_t nodes, unsigned int node_index,
								 Edge_t edges, unsigned int * dest_nodes_to_edges,
								 unsigned int current_num_vertices, unsigned int current_num_edges){
	unsigned int i, num_variables, start_index, end_index, edge_index;
	char has_incoming;
	Edge_t edge;
	Node_t node;
	double sum;

	node = &nodes[node_index];
	num_variables = node->num_variables;

	double new_message[MAX_STATES];
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
		edge = &edges[edge_index];

		combine_message(new_message, edge, num_variables);
		has_incoming = 1;

	}
	if(has_incoming == 1){
		for(i = 0; i < num_variables; ++i){
			node->states[i] = new_message[i];
		}
	}
	sum = 0.0;
	for(i = 0; i < num_variables; ++i){
		sum += node->states[i];
	}
	if(sum <= 0.0){
		sum = 1.0;
	}

	for(i = 0; i < num_variables; ++i){
		node->states[i] = node->states[i] / sum;
	}
}

static void marginalize_nodes_acc(Node_t nodes, Edge_t edges, unsigned int * dest_nodes_to_edges,
								  unsigned int current_num_vertices, unsigned int current_num_edges){
	unsigned int i;

#pragma omp parallel for default(none) shared(nodes, edges, dest_nodes_to_edges, current_num_vertices, current_num_edges) private(i)
	for(i = 0; i < current_num_vertices; ++i){
		marginalize_node_acc(nodes, i, edges, dest_nodes_to_edges, current_num_vertices, current_num_edges);
	}
}

void loopy_propagate_one_iteration(Graph_t graph){
	unsigned int i, num_variables, num_vertices, num_edges;
	unsigned int * dest_node_to_edges;
	unsigned int * src_node_to_edges;
	Node_t node, nodes;
	Edge_t previous, current;
	Edge_t * temp;

	previous = *graph->previous;
	current = *graph->current;

	double message_buffer[MAX_STATES];

	num_vertices = graph->current_num_vertices;
	dest_node_to_edges = graph->dest_nodes_to_edges;
	src_node_to_edges = graph->src_nodes_to_edges;
    nodes = graph->nodes;
    num_edges = graph->current_num_edges;

#pragma omp parallel for default(none) shared(nodes, current, previous, num_vertices, dest_node_to_edges, src_node_to_edges, num_edges) private(message_buffer, i, num_variables, node)
    for(i = 0; i < num_vertices; ++i){
		node = &nodes[i];
		num_variables = node->num_variables;

		initialize_message_buffer(message_buffer, node, num_variables);

		//read incoming messages
		read_incoming_messages(message_buffer, dest_node_to_edges, previous, num_edges, num_vertices, num_variables, i);

/*
		printf("Message at node\n");
		print_node(graph, i);
		printf("[\t");
		for(j = 0; j < num_variables; ++j){
			printf("%.6lf\t", message_buffer[j]);
		}
		printf("\t]\n");*/


		//send message
		send_message_for_node(src_node_to_edges, message_buffer, num_edges, current, num_vertices, i);

	}

	marginalize_loopy_nodes(graph, current, num_vertices);

	//swap previous and current
	temp = graph->previous;
	graph->previous = graph->current;
	graph->current = temp;
}

unsigned int loopy_propagate_until(Graph_t graph, double convergence, unsigned int max_iterations){
	unsigned int i, j, k, num_nodes;
	Edge_t previous_edges, previous, current, current_edges;
	double delta, diff, previous_delta;

	previous_edges = *(graph->previous);
	current_edges = *(graph->current);

	num_nodes = graph->current_num_vertices;

	previous_delta = -1.0;
	delta = 0.0;

	for(i = 0; i < max_iterations; ++i){
		//printf("Current iteration: %d\n", i+1);
		loopy_propagate_one_iteration(graph);

		delta = 0.0;

		for(j = 0; j < num_nodes; ++j){
			previous = &previous_edges[j];
			current = &current_edges[j];

			for(k = 0; k < previous->x_dim; ++k){
				diff = previous->message[k] - current->message[k];
				if(diff != diff){
					continue;
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
		printf("No Convergence: previous: %lf vs current: %lf\n", previous_delta, delta);
	}
	return i;
}

static unsigned int loopy_propagate_iterations_acc(unsigned int num_vertices, unsigned int num_edges,
										   unsigned int *dest_node_to_edges, unsigned int *src_node_to_edges,
										   Node_t nodes,
										   Edge_t *previous, Edge_t *current, unsigned int max_iterations,
										   double convergence){
	unsigned int i, j, k, num_variables, num_iter;
	Node_t node;
	Edge_t temp;
	Edge_t previous_edges, current_edges, previous_edge, current_edge;

	double delta, previous_delta, penultimate_delta, diff;

	previous_edges = *previous;
	current_edges = *current;

	double message_buffer[MAX_STATES];

	num_iter = 0;

	previous_delta = -1.0;

	for(i = 0; i < max_iterations; i+= BATCH_SIZE){
#pragma acc data present_or_copy(current_edges[0:num_edges], nodes[0:num_vertices], previous_edges[0:num_edges]) present_or_copyin(num_vertices, num_edges, dest_node_to_edges[0:num_vertices+num_edges], src_node_to_edges[0:num_vertices+num_edges]) copyout(delta)
		{
			//printf("Current iteration: %d\n", i+1);
			for (j = 0; j < BATCH_SIZE; ++j) {
#pragma acc kernels
				for (k = 0; k < num_vertices; ++k) {
					node = &nodes[k];
					num_variables = node->num_variables;

					initialize_message_buffer(message_buffer, node, num_variables);

					//read incoming messages
					read_incoming_messages(message_buffer, dest_node_to_edges, previous_edges, num_edges, num_vertices,
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
					send_message_for_node(src_node_to_edges, message_buffer, num_edges, current_edges, num_vertices, k);

				}

#pragma kernels
				for (k = 0; k < num_vertices; ++k) {
					marginalize_node_acc(nodes, k, current_edges, dest_node_to_edges, num_vertices, num_edges);
				}

				//swap previous and current
				temp = previous_edges;
				previous_edges = current_edges;
				current_edges = temp;

			}


		}

		delta = 0.0;
		for (j = 0; j < num_vertices; ++j) {
			previous_edge = &previous_edges[j];
			current_edge = &current_edges[j];

			for (k = 0; k < previous_edge->x_dim; ++k) {
				diff = previous_edge->message[k] - current_edge->message[k];
				if (diff != diff) {
					diff = 0.0;
				}
				delta += fabs(diff);
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
		printf("No Convergence: previous: %lf vs current: %lf\n", previous_delta, delta);
	}


	return num_iter;
}

unsigned int loopy_progagate_until_acc(Graph_t graph, double convergence, unsigned int max_iterations){
	unsigned int iter;

	/*printf("===BEFORE====\n");
	print_nodes(graph);
	print_edges(graph);
*/
	iter = loopy_propagate_iterations_acc(graph->current_num_vertices, graph->current_num_edges,
	graph->dest_nodes_to_edges, graph->src_nodes_to_edges,
	graph->nodes,
	graph->previous, graph->current, max_iterations, convergence);

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
	Edge_t edge;

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
			edge = &graph->edges[k];
			g[i][edge->dest_index] = 1;
		}
	}

	for(i = 0; i < graph->current_num_vertices; ++i){
		for(j = 0; j < graph->current_num_vertices; ++j){
			dist[i][j] = g[i][j];
		}
	}

	for(k = 0; k < graph->current_num_vertices; ++k){
		#pragma omp parallel for
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
