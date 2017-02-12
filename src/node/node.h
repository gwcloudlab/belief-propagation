/*
 * node.h
 *
 *  Created on: Feb 5, 2017
 *      Author: ***REMOVED***
 */

#ifndef NODE_H_
#define NODE_H_

#include "../constants.h"

struct node {
	double states[MAX_STATES];
	int num_variables;
	int index;
};
typedef struct node* Node_t;

Node_t create_node(int index, int num_variables);
void initialize_node(Node_t node, int index, int num_variables);
void node_set_state(Node_t node, int num_variables, double * initial_state);

void destroy_node(Node_t);


#endif /* NODE_H_ */
