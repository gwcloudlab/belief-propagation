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
	unsigned int num_variables;
	unsigned int index;
};
typedef struct node* Node_t;

Node_t create_node(unsigned int index, unsigned int num_variables);
void initialize_node(Node_t node, unsigned int index, unsigned int num_variables);
void node_set_state(Node_t node, unsigned int num_variables, double * initial_state);

void destroy_node(Node_t);


#endif /* NODE_H_ */
