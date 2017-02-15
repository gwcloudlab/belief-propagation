#include <stdlib.h>
#include <assert.h>

#include "edge.h"

Edge_t create_edge(int edge_index, int src_index, int dest_index, int dim_x, int dim_y, double ** joint_probabilities) {
	Edge_t e;

	e = (Edge_t)malloc(sizeof(struct edge));
	assert(e);

	init_edge(e, edge_index, src_index, dest_index, dim_x, dim_y, joint_probabilities);

	return e;
}

void init_edge(Edge_t e, int edge_index, int src_index, int dest_index, int dim_x, int dim_y, double ** joint_probabilities){
	int i, j;

	assert(src_index >= 0);
	assert(dest_index >= 0);
	assert(edge_index >= 0);

	assert(dim_x >= 0);
	assert(dim_y >= 0);
	assert(dim_x <= MAX_STATES);
	assert(dim_y <= MAX_STATES);

	e->edge_index = edge_index;
	e->src_index = src_index;
	e->dest_index = dest_index;
	e->x_dim = dim_x;
	e->y_dim = dim_y;

	for(i = 0; i < dim_x; ++i){
		for(j = 0; j < dim_y; ++j){
			e->joint_probabilities[i][j] = joint_probabilities[i][j];
		}
		e->message[i] = 0;
	}
}

void destroy_edge(Edge_t edge) {
	free(edge);
}

void send_message(Edge_t edge, double * message) {
	int i, j, num_src, num_dest;
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
	for(i = 0; i < num_src; ++i){
		edge->message[i] = edge->message[i] / sum;
	}
}
