#include <stdlib.h>
#include <assert.h>

#include "../graph/graph.h"

static const int NUM_NODES = 4;
static const int NUM_EDGES = 5;
static const int NUM_VARIABLES = 2;

Node * create_nodes(){
	int i;
	Node * nodes;
	double y2[2];
	y2[0] = 1.0;
	y2[1] = 0.0;

	nodes = (Node *)malloc(sizeof(Node) * NUM_NODES);

	// set up nodes
	nodes[0] = create_node("x1", 2);
	nodes[1] = create_node("x2", 2);
	nodes[2] = create_node("x3", 2);
	nodes[3] = create_node("y1", 2);


	initialize_node(nodes[3], 2, y2);

	return nodes;
}

Edge * create_edges(Node * nodes){
	int i;
	Edge * edges;
	edges = (Edge *)malloc(sizeof(Edge) * NUM_EDGES);

	double ** phi_1_2 = (double **)malloc(sizeof(double*) * NUM_VARIABLES);
	double ** phi_2_3 = (double **)malloc(sizeof(double*) * NUM_VARIABLES);
	double ** phi_2_4 = (double **)malloc(sizeof(double*) * NUM_VARIABLES);
	for(i = 0; i < NUM_VARIABLES; ++i){
		phi_1_2[i] = (double *)malloc(sizeof(double) * NUM_VARIABLES);
		phi_2_3[i] = (double *)malloc(sizeof(double) * NUM_VARIABLES);
		phi_2_4[i] = (double *)malloc(sizeof(double) * NUM_VARIABLES);
	}

	phi_1_2[0][0] = 1.0;
	phi_1_2[0][1] = 0.9;
	phi_1_2[1][0] = 0.9;
	phi_1_2[1][1] = 1.0;

	phi_2_3[0][0] = 0.1;
	phi_2_3[0][1] = 1.0;
	phi_2_3[1][0] = 1.0;
	phi_2_3[1][1] = 0.1;

	phi_2_4[0][0] = 1.0;
	phi_2_4[0][1] = 0.1;
	phi_2_4[1][0] = 0.1;
	phi_2_4[1][1] = 1.0;

	edges[0] = create_edge(nodes[0], nodes[1], phi_1_2);
	edges[1] = create_edge(nodes[1], nodes[0], phi_1_2);
	edges[2] = create_edge(nodes[1], nodes[2], phi_2_3);
	edges[3] = create_edge(nodes[2], nodes[1], phi_2_3);
	edges[4] = create_edge(nodes[3], nodes[1], phi_2_4);

	for(i = 0; i < NUM_VARIABLES; i++){
		free(phi_1_2[i]);
		free(phi_2_3[i]);
		free(phi_2_4[i]);
	}
	free(phi_1_2);
	free(phi_2_3);
	free(phi_2_4);

	return edges;
}

int main() {
	int i;
	Node n;

	Node * nodes = create_nodes();
	Edge * edges = create_edges(nodes);

	Node * queue = (Node *)malloc(sizeof(Node) * NUM_NODES);
	int queue_size = 0;

	Graph g = create_graph(NUM_NODES, NUM_EDGES);
	for(i = 0; i < NUM_NODES; ++i){
		graph_add_node(g, nodes[i]);
	}
	for(i = 0; i < NUM_EDGES; ++i){
		graph_add_edge(g, edges[i]);
	}

	send_from_leaf_nodes(g, queue, &queue_size, 1);

	for(i = 0; i < NUM_NODES; ++i){
		print_node(nodes[i]);
	}

	Node root = propagate(g, queue, &queue_size);
	for(i = 0; i < NUM_NODES; ++i){
		reset_visited(nodes[i]);
	}
	push_node(root, queue, &queue_size);
	propagate(g, queue, &queue_size);
	marginalize(g);
	for(i = 0; i < NUM_NODES; ++i){
		print_node(nodes[i]);
	}

	free(queue);
	graph_destroy(g);
	for(i = 0; i < NUM_EDGES; ++i){
		destroy_edge(edges[i]);
	}
	for(i = 0; i < NUM_NODES; ++i){
		destroy_node(nodes[i]);
	}

	return 0;
}
