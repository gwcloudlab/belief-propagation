#include <stdlib.h>
#include <assert.h>

#include "edge.h"

Edge create_edge(Node src, Node dest, double ** joint_probabilities) {
	int i, j;
	Edge e;

	e = (Edge)malloc(sizeof(struct edge));
	assert(e);
	e->joint_probabilities = (double **)malloc(sizeof(double*) * src->num_variables);
	assert(e->joint_probabilities);
	for(i = 0; i < src->num_variables; ++i){
		e->joint_probabilities[i] = (double *)malloc(sizeof(double) * dest->num_variables);
		assert(e->joint_probabilities[i]);
	}
	e->src = src;
	e->dest = dest;

	for(i = 0; i < src->num_variables; ++i){
		for(j = 0; j < dest->num_variables; ++j){
			e->joint_probabilities[i][j] = joint_probabilities[i][j];
		}
	}
	e->message = (double *)malloc(sizeof(double) * src->num_variables);

	return e;
}

void destroy_edge(Edge edge) {
	int i, num_variables;
	num_variables = edge->src->num_variables;
	for(i = 0; i < num_variables; ++i){
		free(edge->joint_probabilities[i]);
	}
	free(edge->joint_probabilities);
	free(edge->message);
	free(edge);
}

void send_message(Edge edge, double * message) {
	int i, j, num_src, num_dest;
	num_src = edge->src->num_variables;
	num_dest = edge->dest->num_variables;
	for(i = 0; i < num_src; ++i){
		edge->message[i] = 0.0;
		for(j = 0; j < num_dest; ++j){
			edge->message[i] += edge->joint_probabilities[i][j] * message[j];
		}
	}
}
