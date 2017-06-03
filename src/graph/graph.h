/*
 * graph.h
 *
 *  Created on: Feb 5, 2017
 */

/**
 * Based off of http://www.cs.yale.edu/homes/aspnes/pinewiki/C(2f)Graphs.html
 */

#ifndef GRAPH_H_
#define GRAPH_H_

#include "../node/node.h"
#include "../edge/edge.h"

struct graph {
	unsigned int total_num_vertices;
	unsigned int total_num_edges;

	unsigned int current_num_vertices;
	unsigned int current_num_edges;

	Edge_t edges;
	Edge_t prev_edges;

	Edge_t * current;
	Edge_t * previous;

	Node_t nodes;

	unsigned int * src_nodes_to_edges;
	unsigned int * dest_nodes_to_edges;

	unsigned int * levels_to_nodes;
	unsigned int num_levels;

    int diameter;

	char * visited;
	char * node_names;

	char * variable_names;

    char * observed_nodes;

	char graph_name[CHAR_BUFFER_SIZE];
};
typedef struct graph* Graph_t;

/** create a new graph with n vertices labeled 0 to n-1 and no edges */
Graph_t create_graph(unsigned int, unsigned int);

void graph_add_node(Graph_t, unsigned int, const char *);
void graph_add_and_set_node_state(Graph_t, unsigned int, const char *, long double *);
void graph_set_node_state(Graph_t, unsigned int, unsigned int, long double *);

void graph_add_edge(Graph_t, unsigned int, unsigned int, unsigned int, unsigned int, long double **);

void set_up_src_nodes_to_edges(Graph_t);
void set_up_dest_nodes_to_edges(Graph_t);
void init_levels_to_nodes(Graph_t);
void calculate_diameter(Graph_t);
/**
 * Get the counts
 */
int graph_vertex_count(Graph_t);
int graph_edge_count(Graph_t);

/** free space **/
void graph_destroy(Graph_t);


void propagate_using_levels_start(Graph_t);
void propagate_using_levels(Graph_t, unsigned int);

void reset_visited(Graph_t);

void init_previous_edge(Graph_t);
void loopy_propagate_one_iteration(Graph_t);

unsigned int loopy_propagate_until(Graph_t, long double convergence, unsigned int max_iterations);

void loopy_propagate_one_iteration_shared_buffer(Graph_t, long double *);

unsigned int loopy_propagate_until_shared_buffer(Graph_t, long double convergence, unsigned int max_iterations);

void marginalize(Graph_t);

void print_node(Graph_t, unsigned int);
void print_edge(Graph_t, unsigned int);
void print_nodes(Graph_t);
void print_edges(Graph_t);
void print_src_nodes_to_edges(Graph_t);
void print_dest_nodes_to_edges(Graph_t);
void print_levels_to_nodes(Graph_t);

#endif /* GRAPH_H_ */
