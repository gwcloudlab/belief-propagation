//
// Created by mjt5v on 3/4/18.
//

#ifndef PROJECT_CSR_WRAPPER_H
#define PROJECT_CSR_WRAPPER_H

#include <stdio.h>

void test_10_20_file(const char *, const char *);

void run_test_belief_propagation_mtx_files(const char *, const char *, FILE *);
void run_test_loopy_belief_propagation_mtx_files(const char *, const char *, FILE *);
void run_test_loopy_belief_propagation_edge_mtx_files(const char *, const char *, FILE *);
void run_test_loopy_belief_propagation_mtx_files_acc(const char *, const char *, FILE *);
void run_test_loopy_belief_propagation_edge_mtx_files_acc(const char *, const char *, FILE *);

#endif //PROJECT_CSR_WRAPPER_H
