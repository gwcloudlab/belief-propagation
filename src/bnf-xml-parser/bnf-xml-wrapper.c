#include "../bnf-parser/bnf-wrapper.h"
#include "bnf-xml-wrapper.h"

struct expression * test_parse_xml_file(char * file_name) {
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

static char * build_full_file_path(const char *prefix, const char *file, char *buffer, size_t length){
    strncpy(buffer, prefix, length);
    strncat(buffer, file, length);
    return buffer;
}

void test_dog_files(const char * root_dir){
    struct expression * bif_expr;
    char file_name_buffer[128];

    bif_expr = parse_file(build_full_file_path(root_dir, "dog.bif", file_name_buffer, 128));
    Graph_t graph_xml = parse_xml_file(build_full_file_path(root_dir, "dog.xml", file_name_buffer, 128));
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
}

void test_sample_xml_file(const char *root_dir){
    Graph_t graph_xml;
    char file_name_buffer[128];

    graph_xml = parse_xml_file(build_full_file_path(root_dir, "10_20.xml", file_name_buffer, 128));
    assert(graph_xml);
    printf("Small graph\n");
    print_nodes(graph_xml);
    print_edges(graph_xml);

    graph_destroy(graph_xml);
}