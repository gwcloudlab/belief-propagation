/*
 * constants.h
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include <stdint.h>

#define CHAR_BUFFER_SIZE 50

#define AVG_STATES 2
//#define AVG_STATES 3

#define NODE_DIFFERENCE_THRESHOLD 0.001f

#define MAX_STATES 2
//#define MAX_STATES 3

#define DEFAULT_STATE 1.0f

#define WEIGHT_INFINITY SIZE_MAX / 2

#define BATCH_SIZE 10

#define BATCH_SIZE_EDGE 1

#define NUM_ITERATIONS 200

#define NUM_PARTITIONS 32

#define PRECISION 1E-3f

#define PRECISION_ITERATION 1E-3f

#define PARTITION_SIZE 1000

#define NUM_THREAD_PARTITIONS 8

#define BLOCK_SIZE 1024

#define BLOCK_SIZE_NODE_STREAMING 512

#define BLOCK_SIZE_EDGE_STREAMING 1024

#define BLOCK_SIZE_NODE_EDGE_STREAMING 256

#define MIN_BLOCKS_PER_MP 16

#define BLOCK_SIZE_2_D_X 64

#define BLOCK_SIZE_2_D_Y 16

#define BLOCK_SIZE_3_D_X 4

#define BLOCK_SIZE_3_D_Y 16

#define BLOCK_SIZE_3_D_Z 16

#define MAX_NUM_NODES 1048

#define WARP_SIZE 32

#define MAX_DEGREE 4096

#define CHARS_IN_KEY 64

#define READ_SNAP_BUFFER_SIZE 256

#define REGEX_GRAPH_INFO "^# Nodes:[\r\n\t\f\v ]*([0-9]+)[\r\n\t\f\v ]*Edges:[\r\n\t\f\v ]*([0-9]+)[\r\n\t\f\v ]*Beliefs:[\r\n\t\f\v ]*([0-9]+)[\r\n\t\f\v ]*Belief States:[\r\n\t\f\v ]*([0-9]+).*$"
#define REGEX_EDGE_LINE "^([0-9]+)[\r\n\t\f\v ]+([0-9]+)[\r\n\t\f\v ]+(.*)$"
#define REGEX_NODE_LINE "^([0-9]+)[\r\n\t\f\v ]+(.*)$"
#define REGEX_WHITESPACE "\r\n\t\f\v "

#define DAMPENING_FACTOR 0.85f

#endif /* CONSTANTS_H_ */
