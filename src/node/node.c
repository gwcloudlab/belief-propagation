#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

#include "node.h"

Node_t create_node(unsigned int index, unsigned int num_variables) {
	Node_t n;

	n = (Node_t)malloc(sizeof(struct node));
	initialize_node(n, index, num_variables);

	return n;
}

void initialize_node(Node_t n, unsigned int index, unsigned int num_variables){
	unsigned int i;

	//initialize to default state since initial state not given
	for(i = 0; i < num_variables; ++i){
		n->states[i] = DEFAULT_STATE;
	}

	n->num_variables = num_variables;
	n->index = index;
}

void node_set_state(Node_t n, unsigned int num_variables, double * state){
	unsigned int i;

	for(i = 0; i < num_variables; ++i) {
		n->states[i] = state[i];
	}

}

void destroy_node(Node_t n) {
	free(n);
}

