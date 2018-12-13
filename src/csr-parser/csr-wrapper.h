//
// Created by mjt5v on 3/4/18.
//

#ifndef PROJECT_CSR_WRAPPER_H
#define PROJECT_CSR_WRAPPER_H

#include <stdio.h>
#include "../graph/graph.h"

void test_10_20_file(const char *, const char *, const struct joint_probability *, int, int);

void run_test_belief_propagation_mtx_files(const char *, const char *, const struct joint_probability *, int, int, FILE *);
void run_test_loopy_belief_propagation_mtx_files(const char *, const char *, const struct joint_probability *, int, int, FILE *);
void run_test_loopy_belief_propagation_edge_mtx_files(const char *, const char *, const struct joint_probability *, int, int, FILE *);
void run_test_loopy_belief_propagation_mtx_files_acc(const char *, const char *, const struct joint_probability *, int, int, FILE *);
void run_test_loopy_belief_propagation_edge_mtx_files_acc(const char *, const char *, const struct joint_probability *, int, int, FILE *);

#endif //PROJECT_CSR_WRAPPER_H
