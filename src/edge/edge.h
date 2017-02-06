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
	Node src;
	Node dest;
	double ** joint_probabilities;
	double * message;
};
typedef struct edge *Edge;

Edge create_edge(Node src, Node dest, double ** joint_probabilities);
void destroy_edge(Edge);
void send_message(Edge, double *);


#endif /* EDGE_H_ */
