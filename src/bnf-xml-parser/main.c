#include "xml-expression.h"
#include "../bnf-parser/expression.h"
#include <stdlib.h>
#include <assert.h>
#include "../bnf-parser/Lexer.h"
#include "../bnf-parser/Parser.h"

struct expression * test_parse_file(char * file_name) {
    unsigned int i;
    struct expression *expression;
    yyscan_t scanner;
    YY_BUFFER_STATE state;
    FILE *in;

    assert(yylex_init(&scanner) == 0);

    in = fopen(file_name, "r");

    yyset_in(in, scanner);

    assert(yyparse(&expression, scanner) == 0);
    //yy_delete_buffer(state, scanner);
    yylex_destroy(scanner);

    fclose(in);

    assert(expression != NULL);

    return expression;
}

int main() {
    struct expression * bif_expr;

    bif_expr = test_parse_file("../benchmark_files/dog.bif");
    Graph_t graph_xml = parse_xml_file("../benchmark_files/dog.xml");
    Graph_t graph_bif = build_graph(bif_expr);

    assert(graph_xml->current_num_edges == graph_bif->current_num_edges);
    assert(graph_xml->current_num_vertices == graph_bif->current_num_vertices);

    printf("XML File\n");
    print_nodes(graph_xml);
    print_edges(graph_xml);
    printf("BIF File\n");
    print_nodes(graph_bif);
    print_edges(graph_bif);

    if(bif_expr != NULL){
        delete_expression(bif_expr);
    }
    if(graph_xml != NULL){
        graph_destroy(graph_xml);
    }
    if(graph_bif != NULL){
        graph_destroy(graph_bif);
    }

   /* xml_expr = parse_xml_file("../benchmark_files/xml/bf_10000_20000_1.xml");
    assert(xml_expr);
    graph_xml = build_graph(xml_expr);
    assert(graph_xml);

    if(xml_expr != NULL){
        delete_expression(xml_expr);
    }
    if(graph_xml != NULL){
        graph_destroy(graph_xml);
    }*/

    graph_xml = parse_xml_file("../benchmark_files/xml2/10_20.xml");
    assert(graph_xml);
    printf("Small graph\n");
    print_nodes(graph_xml);
    print_edges(graph_xml);

    graph_destroy(graph_xml);

    return 0;
}