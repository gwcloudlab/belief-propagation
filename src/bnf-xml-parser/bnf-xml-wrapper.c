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


void run_test_belief_propagation_xml_file(const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int i;

    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_levels_to_nodes(graph);
    //print_levels_to_nodes(graph);

    propagate_using_levels_start(graph);
    for(i = 1; i < graph->num_levels - 1; ++i){
        propagate_using_levels(graph, i);
    }
    reset_visited(graph);
    for(i = graph->num_levels - 1; i > 0; --i){
        propagate_using_levels(graph, i);
    }

    marginalize(graph);
    end = clock();

    time_elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(out, "%s,regular,%d,%d,%d,2,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_xml_file(const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy,%d,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_edge_xml_file(const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_edge(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy-edge,%d,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    graph_destroy(graph);
}
