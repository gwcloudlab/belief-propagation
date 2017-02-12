/*
 * edge.h
 *
 *  Created on: Feb 5, 2017
 *      Author: mjt5v
 */

#ifndef EDGE_H_
#define EDGE_H_

#include "../node/node.h"

struct edge {
	int edge_index;
	int src_index;
	int dest_index;
	int x_dim;
	int y_dim;
	double joint_probabilities[MAX_STATES][MAX_STATES];
	double message[MAX_STATES];
};
typedef struct edge* Edge_t;

Edge_t create_edge(int, int, int, int, int, double **);
void init_edge(Edge_t, int, int, int, int, int, double **);
void destroy_edge(Edge_t);
void send_message(Edge_t, double *);


#endif /* EDGE_H_ */
