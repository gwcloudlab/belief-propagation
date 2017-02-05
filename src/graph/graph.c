#include <stdlib.h>
#include <assert.h>

#include "graph.h"

Graph
create_graph(int num_vertices, int num_edges)
{
	Graph g;

	g = (Graph)malloc(sizeof(struct graph));
	assert(g);
	g->edges = (Edge *)malloc(sizeof(Edge) * num_edges);
	assert(g->edges);
	g->nodes = (Node *)malloc(sizeof(Node) * num_vertices);
	assert(g->nodes);
	g->total_num_vertices = num_vertices;
	g->total_num_edges = num_edges;
	g->current_num_vertices = 0;
	g->current_num_edges = 0;

	return g;
}

void graph_add_node(Graph g, Node n) {
	g->nodes[g->current_num_vertices] = n;
	g->current_num_vertices += 1;
}

void graph_add_edge(Graph g, Edge e) {
	g->edges[g->current_num_edges] = e;
	g->current_num_edges += 1;
}

int graph_vertex_count(Graph g) {
	return g->current_num_vertices;
}

int graph_edge_count(Graph g) {
	return g->current_num_edges;
}

void graph_destroy(Graph g) {
	free(g->nodes);
	free(g->edges);
	free(g);
}


Node * get_leaf_nodes(Graph g, int max_count) {
	Node nodes[g->current_num_vertices];
	int num_leaf_nodes, i, j;
	int num_srcs;
	num_leaf_nodes = 0;

	for(i = 0; i < g->current_num_vertices; ++i){
		Node current_vertex = g->nodes[i];
		num_srcs = 0;
		for(j = 0; j < g->current_num_edges; ++j) {
			Edge edge = g->edges[j];
			if(edge->src == current_vertex){
				num_srcs++;
			}
		}
		if(num_srcs <= max_count) {
			nodes[num_leaf_nodes] = current_vertex;
			num_leaf_nodes++;
		}
	}

	Node * leaf_nodes = (Node *)malloc(sizeof(Node) * num_leaf_nodes);
	assert(leaf_nodes);
	for(i = 0; i < num_leaf_nodes; ++i){
		leaf_nodes[i] = nodes[i];
	}
	return leaf_nodes;
}
