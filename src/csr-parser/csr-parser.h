//
// Created by mjt5v on 3/4/18.
//

#ifndef PROJECT_CSR_PARSER_H
#define PROJECT_CSR_PARSER_H

#include "../graph/graph.h"

/**
 * Builds a graph from the provided edges and nodes files
 * @return A constructed graph representing the belief network encoded as the CSR
 */
Graph_t build_graph_from_mtx(const char *, const char *);

#endif //PROJECT_CSR_PARSER_H
