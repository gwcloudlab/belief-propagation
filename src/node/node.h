/*
 * node.h
 *
 *  Created on: Feb 5, 2017
 *      Author: ***REMOVED***
 */

#ifndef NODE_H_
#define NODE_H_

struct node {
	double * state;
	int num_variables;
	char name[50];
};
typedef struct node *Node;

static const double DEFAULT_STATE = 1.0;

Node create_node(const char * name, int num_variables);
void initialize_node(Node node, int num_variables, double * initial_state);

void destroy_node(Node);


#endif /* NODE_H_ */
