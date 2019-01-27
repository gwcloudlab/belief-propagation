
#ifndef PROJECT_SNAP_PARSER_H
#define PROJECT_SNAP_PARSER_H

#include "../graph/graph.h"
#include <stdio.h>

/**
 * Struct to hold the basic data about the graph
 */
struct graph_info {
    /**
     * The number of nodes
     */
    size_t num_nodes;
    /**
     * The number of edges
     */
    size_t num_edges;
    /**
     * The number of beliefs
     */
    size_t num_beliefs;
    /**
     * The number of states per belief
     */
    size_t num_belief_states;
};


Graph_t parse_graph_from_snap_files(const char *, const char *);


void test_sample_snap_dog_files(const char *);
void run_test_belief_propagation_snap_file(const char *, const char *, FILE *);
void run_test_loopy_belief_propagation_snap_file(const char *, const char *, FILE *);
void run_test_loopy_belief_propagation_edge_snap_file(const char *, const char *, FILE *);
void run_test_loopy_belief_propagation_snap_file_acc(const char *, const char *, FILE *);
void run_test_loopy_belief_propagation_edge_snap_file_acc(const char *, const char *, FILE *);

void test_page_rank_sample_file(const char *);
void test_page_rank_sample_edge_file(const char *);

#endif //PROJECT_SNAP_PARSER_H
