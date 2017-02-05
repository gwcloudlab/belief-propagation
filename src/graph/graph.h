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

	Edge * edges;
	Node * nodes;
};
typedef struct graph *Graph;

/** create a new graph with n vertices labeled 0 to n-1 and no edges */
Graph create_graph(int, int);

void graph_add_node(Graph, Node);

/** add an edge to the graph **/
void graph_add_edge(Graph, Edge);

/**
 * Get the counts
 */
int graph_vertex_count(Graph);
int graph_edge_count(Graph);

/** free space **/
void graph_destroy(Graph);

Node * get_leaf_nodes(Graph, int);




#endif /* GRAPH_H_ */
