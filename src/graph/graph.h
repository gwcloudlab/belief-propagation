/*
 * graph.h
 *
 *  Created on: Feb 5, 2017
 *      Author: mjt5v
 */

/**
 * Based off of http://www.cs.yale.edu/homes/aspnes/pinewiki/C(2f)Graphs.html
 */

#ifndef GRAPH_H_
#define GRAPH_H_

#include "../node/node.h"
#include "../edge/edge.h"

struct graph {
	int total_num_vertices;
	int total_num_edges;

	int current_num_vertices;
	int current_num_edges;

	Edge_t edges;
	Node_t nodes;

	int * src_nodes_to_edges;
	int * dest_nodes_to_edges;

	int * leaf_node_queue;
	int leaf_node_queue_size;

	int * forward_queue;
	int forward_queue_size;

	int * backward_queue;
	int backward_queue_size;

	char * visited;
	char * node_names;
};
typedef struct graph* Graph_t;

/** create a new graph with n vertices labeled 0 to n-1 and no edges */
Graph_t create_graph(int, int);

void graph_add_node(Graph_t, int, const char *);
void graph_add_and_set_node_state(Graph_t, int, const char *, double *);

void graph_add_edge(Graph_t, int, int, int, int, double **);

void set_up_src_nodes_to_edges(Graph_t);
void set_up_dest_nodes_to_edges(Graph_t);
/**
 * Get the counts
 */
int graph_vertex_count(Graph_t);
int graph_edge_count(Graph_t);

/** free space **/
void graph_destroy(Graph_t);

void fill_forward_buffer_with_leaf_nodes(Graph_t, int);
void push_node(int, int *, int *);
int pop_node(int *, int *);
void send_from_leaf_nodes(Graph_t);
void propagate_node(Graph_t, int, int *, int *, int *, int *);
void propagate(Graph_t, int *, int *, int *, int *);
void reset_visited(Graph_t);

void marginalize(Graph_t);

void print_node(Graph_t, int);
void print_edge(Graph_t, int);
void print_nodes(Graph_t);
void print_edges(Graph_t);
void print_src_nodes_to_edges(Graph_t);
void print_dest_nodes_to_edges(Graph_t);

#endif /* GRAPH_H_ */
