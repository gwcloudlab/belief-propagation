

#ifndef BNF_WRAPPER_H
#define BNF_WRAPPER_H

#include <stdio.h>
#include <assert.h>
#include <time.h>

#include "expression.h"
#include "Parser.h"
#include "Lexer.h"

void test_ast(const char *);
void test_file(const char *);
void test_parse_file(char *);
void test_loopy_belief_propagation(char *);
struct expression * parse_file(const char *);
void run_test_belief_propagation(struct expression *, const char *);
void run_test_loopy_belief_propagation(struct expression *, const char *);
void run_tests_with_file(const char *, unsigned int);

void basic_test_suite(const char *);
void small_test_suite(const char *);
void medium_test_suite(const char *);
void large_test_suite(const char *);
void very_large_test_suite(const char *);


#endif //BNF_WRAPPER_H
