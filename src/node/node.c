#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "node.h"

Node create_node(const char * name, int num_variables) {
	Node n;
	int i;

	n = (Node)malloc(sizeof(struct node));
	n->state = (double *)malloc(sizeof(double) * num_variables);

	//initialize to default state since initial state not given
	for(i = 0; i < num_variables; ++i){
		n->state[i] = DEFAULT_STATE;
	}

	strncpy(n->name, name, 50);
	n->num_variables = num_variables;

	return n;
}

void initialize_node(Node n, int num_variables, double * state){
	int i;

	assert(num_variables == n->num_variables);

	for(i = 0; i < num_variables; ++i) {
		n->state[i] == state[i];
	}
}

void destroy_node(Node n) {
	free(n->state);
	free(n);
}
