#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

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

void push_node(Node n, Node * queue, int * num_elements){
	int i;

	for(i = 0; i < *num_elements; ++i){
		if(queue[i] == n){

			return;
		}
	}

	queue[*num_elements] = n;
	*num_elements += 1;
}

Node pop_node(Node * queue, int * num_elements){
	int i;
	if(num_elements == 0){
		return NULL;
	}
	Node n = queue[0];
	for(i = 1; i < *num_elements; ++i){
		queue[i-1] = queue[i];
	}
	*num_elements -= 1;
	return n;
}


void send_from_leaf_nodes(Graph g, Node * queue, int * num_elements, int max_count) {
	int i, j, current_num_vertices, current_num_edges;
	int num_srcs;
	Node current_vertex;
	Edge edge;

	current_num_vertices = g->current_num_vertices;
	current_num_edges = g->current_num_edges;

	*num_elements = 0;

	for(i = 0; i < current_num_vertices; ++i){
		current_vertex = g->nodes[i];
		num_srcs = 0;
		for(j = 0; j < g->current_num_edges; ++j) {
			edge = g->edges[j];
			if(edge->src == current_vertex){
				num_srcs++;
			}
		}
		if(num_srcs <= max_count) {
			current_vertex->visited = 1;
			//send messages
			for(j = 0; j < current_num_edges; ++j){
				edge = g->edges[j];
				if(edge->src == current_vertex){
					send_message(edge, current_vertex->state);
					push_node(edge->dest, queue, num_elements);
				}
			}
		}
	}
}

void combine_message(double * dest, double * src, int length){
	int i;
	for(i = 0; i < length; ++i){
		dest[i] = dest[i] * src[i];
	}
}

void propagate_node(Graph g, Node dest, Node * queue, int * queue_size){
	int i, j, num_variables, num_edges;
	Edge edge;

	dest->visited = 1;

	num_variables = dest->num_variables;
	num_edges = g->current_num_edges;

	double message[num_variables];

	for(i = 0; i < num_variables; ++i){
		message[i] = 1.0;
	}

	for(j = 0; j < num_edges; ++j){
		edge = g->edges[j];

		if(edge->dest == dest){
			combine_message(message, edge->message, num_variables);
		}
	}

	//send message
	for(j = 0; j < num_edges; ++j){
		edge = g->edges[j];

		if(edge->src == dest && edge->dest->visited == 0){
			printf("Sending message from: %s to %s\n{\t", edge->src->name, edge->dest->name);
			for(i = 0; i < num_variables; ++i){
				printf("%.6f\t", message[i]);
			}
			printf("}\n");
			send_message(edge, message);
		}
	}

	//update queue
	for(j = 0; j < num_edges; ++j){
		edge = g->edges[j];
		if(edge->src == dest && edge->dest->visited == 0){
			push_node(edge->dest, queue, queue_size);
		}
	}
}

Node propagate(Graph g, Node * queue, int * queue_size){
	Node node;
	double * message;

	while(*queue_size != 0){
		node = pop_node(queue, queue_size);
		printf("Visiting node: %s\n", node->name);

		propagate_node(g, node, queue, queue_size);
	}
	printf("All done\n");
	return node;
}

void marginalize_node(Graph g, Node n){
	int i, num_variables, num_edges;
	char has_incoming;
	Edge edge;
	double sum;

	num_variables = n->num_variables;
	num_edges = g->current_num_edges;
	has_incoming = 0;

	double new_message[num_variables];
	for(i = 0; i < num_variables; ++i){
		new_message[i] = 1.0;
	}

	for(i = 0; i < num_edges; ++i){
		edge = g->edges[i];

		if(edge->dest == n){
			combine_message(new_message, edge->message, num_variables);
			has_incoming = 1;
		}
	}
	if(has_incoming == 1){
		for(i = 0; i < num_variables; ++i){
			n->state[i] = new_message[i];
		}
	}
	sum = 0.0;
	for(i = 0; i < num_variables; ++i){
		sum += n->state[i];
	}
	if(sum <= 0.0){
		sum = 1.0;
	}

	for(i = 0; i < num_variables; ++i){
		n->state[i] = n->state[i] / sum;
	}
}

void marginalize(Graph g){
	Node n;
	int i, j, num_nodes;

	num_nodes = g->current_num_vertices;

	for(i = 0; i < num_nodes; ++i){
		n = g->nodes[i];

		marginalize_node(g, n);
	}
}
