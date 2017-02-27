/*
 * edge.h
 *
 *  Created on: Feb 5, 2017
 *      Author: ***REMOVED***
 */

#ifndef EDGE_H_
#define EDGE_H_

#include "../node/node.h"

struct edge {
	unsigned int edge_index;
	unsigned int src_index;
	unsigned int dest_index;
	unsigned int x_dim;
	unsigned int y_dim;
	double joint_probabilities[MAX_STATES][MAX_STATES];
	double message[MAX_STATES];
};
typedef struct edge* Edge_t;

Edge_t create_edge(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, double **);
void init_edge(Edge_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, double **);
void destroy_edge(Edge_t);
void send_message(Edge_t, double *);


#endif /* EDGE_H_ */
