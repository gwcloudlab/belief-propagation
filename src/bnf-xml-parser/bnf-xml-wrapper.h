
#ifndef BNF_XML_WRAPPER_H
#define BNF_XML_WRAPPER_H

#include "xml-expression.h"
#include "../bnf-parser/expression.h"
#include <stdlib.h>
#include <assert.h>
#include "../bnf-parser/Lexer.h"
#include "../bnf-parser/Parser.h"

struct expression * test_parse_xml_file(char *);
void test_dog_files(const char *);
void test_sample_xml_file(const char *);
void run_test_belief_propagation_xml_file(const char *, FILE *);
void run_test_loopy_belief_propagation_xml_file(const char *, FILE *);
void run_test_loopy_belief_propagation_edge_xml_file(const char *, FILE *);
void run_test_loopy_belief_propagation_xml_file_acc(const char *, FILE *);
void run_test_loopy_belief_propagation_edge_xml_file_acc(const char *, FILE *);

void run_test_viterbi_xml_file(const char *);

#endif //BNF_XML_WRAPPER_H
