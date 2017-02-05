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
	Node src;
	Node dest;
	double ** joint_probabilities;
};
typedef struct edge *Edge;

Edge create_edge(Node src, Node dest, double ** joint_probabilities);
void destroy_edge(Edge);



#endif /* EDGE_H_ */
