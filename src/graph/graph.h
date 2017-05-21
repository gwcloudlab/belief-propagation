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

#ifndef __USE_GNU
#define __USE_GNU
#endif

#include "../constants.h"

#include <search.h>

struct graph {
	unsigned int total_num_vertices;
	unsigned int total_num_edges;

	unsigned int current_num_vertices;
	unsigned int current_num_edges;
	unsigned int max_degree;

	unsigned int * edges_src_index;
	unsigned int * edges_dest_index;
	unsigned int * edges_x_dim;
	unsigned int * edges_y_dim;
	float * edges_joint_probabilities;

	float * edges_messages;
	float * last_edges_messages;

	float ** current_edge_messages;
    float ** previous_edge_messages;


	float * node_states;
	unsigned int * node_num_vars;

	unsigned int * src_nodes_to_edges_node_list;
	unsigned int * src_nodes_to_edges_edge_list;

	unsigned int * dest_nodes_to_edges_node_list;
	unsigned int * dest_nodes_to_edges_edge_list;

	unsigned int * levels_to_nodes;
	unsigned int num_levels;

    int diameter;

	char * visited;
	char * node_names;

	char * variable_names;

    char * observed_nodes;

	char graph_name[CHAR_BUFFER_SIZE];

    char node_hash_table_created;
	struct hsearch_data *node_hash_table;

    struct hsearch_data *src_node_to_edge_table;

    struct hsearch_data *dest_node_to_edge_table;
    char edge_tables_created;

	struct hsearch_data *node_names_to_indices;
};
typedef struct graph* Graph_t;

struct htable_entry {
    unsigned int indices[MAX_DEGREE];
    unsigned int count;
};

/** create a new graph with n vertices labeled 0 to n-1 and no edges */
Graph_t create_graph(unsigned int, unsigned int);

void graph_add_node(Graph_t, unsigned int, const char *);
void graph_add_and_set_node_state(Graph_t, unsigned int, const char *, float *);
void graph_set_node_state(Graph_t, unsigned int, unsigned int, float *);

void graph_add_edge(Graph_t, unsigned int, unsigned int, unsigned int, unsigned int, float *);

void set_up_src_nodes_to_edges(Graph_t);
void set_up_dest_nodes_to_edges(Graph_t);
void init_levels_to_nodes(Graph_t);
void calculate_diameter(Graph_t);

void initialize_node(Graph_t, unsigned int, unsigned int);
void node_set_state(Graph_t, unsigned int, unsigned int, float *);

void init_edge(Graph_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float *);
void send_message(float *, unsigned int, unsigned int, float *, float *, unsigned int *, unsigned int *);

/**
 * Get the counts
 */
int graph_vertex_count(Graph_t);
int graph_edge_count(Graph_t);

/** free space **/
void graph_destroy(Graph_t);

unsigned int find_node_index_by_name(Graph_t, char *);

void propagate_using_levels_start(Graph_t);
void propagate_using_levels(Graph_t, unsigned int);

void reset_visited(Graph_t);

void init_previous_edge(Graph_t);
void loopy_propagate_one_iteration(Graph_t);

unsigned int loopy_propagate_until(Graph_t, float convergence, unsigned int max_iterations);
unsigned int loopy_progagate_until_acc(Graph_t, float convergence, unsigned int max_iterations);

void marginalize(Graph_t);

void print_node(Graph_t, unsigned int);
void print_edge(Graph_t, unsigned int);
void print_nodes(Graph_t);
void print_edges(Graph_t);
void print_src_nodes_to_edges(Graph_t);
void print_dest_nodes_to_edges(Graph_t);
void print_levels_to_nodes(Graph_t);


#endif /* GRAPH_H_ */
