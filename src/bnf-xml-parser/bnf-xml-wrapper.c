#include "../bnf-parser/bnf-wrapper.h"
#include "bnf-xml-wrapper.h"

/**
 * Tests parsing an XML file
 * @param file_name The path of the XML file
 * @return A BNF AST
 */
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

/**
 * Helper function for building full file paths from relative ones
 * @param prefix The directory holding the path
 * @param file The file within the prefix directory to parse
 * @param buffer The char buffer for the combined path
 * @param length The size of the buffer
 * @return The buffer
 */
static char * build_full_file_path(const char *prefix, const char *file, char *buffer, size_t length){
    size_t prefix_length;
    prefix_length = strlen(prefix);
    assert(prefix_length < length);
    strncpy(buffer, prefix, length);
    strncat(buffer, file, length - prefix_length);
    return buffer;
}

/**
 * Tests parsing both the BNF BIF and XML Dog network files
 * @param root_dir The directory holding the files
 */
void test_dog_files(const char * root_dir){
    struct expression * bif_expr;
    char file_name_buffer[128];

    // parse the files
    bif_expr = parse_file(build_full_file_path(root_dir, "dog.bif", file_name_buffer, 128));
    Graph_t graph_xml = parse_xml_file(build_full_file_path(root_dir, "dog.xml", file_name_buffer, 128));
    Graph_t graph_bif = build_graph(bif_expr);

    assert(graph_xml->current_num_edges == graph_bif->current_num_edges);
    assert(graph_xml->current_num_vertices == graph_bif->current_num_vertices);

    // check their content
    printf("XML File\n");
    print_nodes(graph_xml);
    print_edges(graph_xml);
    printf("BIF File\n");
    print_nodes(graph_bif);
    print_edges(graph_bif);

    // cleanup
    if(bif_expr != NULL){
        delete_expression(bif_expr);
    }
    graph_destroy(graph_xml);
    graph_destroy(graph_bif);
}

/**
 * Tests parsing a simple XML file
 * @param root_dir The directory holding the file
 */
void test_sample_xml_file(const char *root_dir){
    Graph_t graph_xml;
    char file_name_buffer[128];

    // parse the file and output the graph
    graph_xml = parse_xml_file(build_full_file_path(root_dir, "10_20.xml", file_name_buffer, 128));
    assert(graph_xml);
    printf("Small graph\n");
    print_nodes(graph_xml);
    print_edges(graph_xml);

    // cleanup
    graph_destroy(graph_xml);
}

/**
 * Runs BP on the XML file data
 * @param file_name The path to the file to read
 * @param out The CSV file to output to
 */
void run_test_belief_propagation_xml_file(const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int i;

    // parse the file
    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    // set up the parallel arrays
    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    // start BP
    start = clock();
    init_levels_to_nodes(graph);
    //print_levels_to_nodes(graph);

    // propagate from the leaves to the root
    propagate_using_levels_start(graph);
    for(i = 1; i < graph->num_levels - 1; ++i){
        propagate_using_levels(graph, i);
    }

    // go back the other way
    reset_visited(graph);
    for(i = graph->num_levels - 1; i > 0; --i){
        propagate_using_levels(graph, i);
    }

    marginalize(graph);
    end = clock();

    // output
    time_elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(out, "%s,regular,%d,%d,%d,2,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}

/**
 * Runs node-optimized loopy BP on the XML file
 * @param file_name The path to the XML file
 * @param out The CSV file to output to
 */
void run_test_loopy_belief_propagation_xml_file(const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    // read the data
    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    // set up parallel arrays
    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    // start loopy BP
    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    // output data
    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy,%d,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}

/**
 * Runs edge-optimized loopy BP on the file
 * @param file_name The file to read
 * @param out The CSV file to output to
 */
void run_test_loopy_belief_propagation_edge_xml_file(const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    // read the XML file
    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    // set up parallel arrays
    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    // start loopy BP
    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_edge(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    // output
    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy-edge,%d,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}

/**
 * Runs node-optimized loopy BP for OpenACC
 * @param file_name The path to the file to read
 * @param out The CSV to output to
 */
void run_test_loopy_belief_propagation_xml_file_acc(const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    // read the data
    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    // set up parallel arrays
    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    // start loopy bp
    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_acc(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    // output
    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy,%d,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}

/**
 * Runs edge optimized loopy BP for OpenACC
 * @param file_name The path of the file to read
 * @param out The CSV to output to
 */
void run_test_loopy_belief_propagation_edge_xml_file_acc(const char * file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    // read the data
    graph = parse_xml_file(file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    // set up parallel arrays
    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    // start loopy BP
    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_edge_acc(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    // output
    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy-edge,%d,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}

/**
 * Runs the Viterbi HMM from the Wikipedia example
 * @see https://en.wikipedia.org/wiki/Viterbi_algorithm#Example
 * @param file_name
 * @param out
 */
void run_test_viterbi_xml_file(const char * root_dir){
    Graph_t graph;
    char file_name_buffer[128];

    // read the data
    graph = parse_xml_file(build_full_file_path(root_dir, "viterbi.xml", file_name_buffer, 128));
    assert(graph != NULL);
    print_nodes(graph);
    print_edges(graph);

    set_up_dest_nodes_to_edges(graph);
    set_up_src_nodes_to_edges(graph);
    init_previous_edge(graph);
    viterbi_until(graph, PRECISION, NUM_ITERATIONS);

    print_nodes(graph);
    print_edges(graph);

    assert(graph->node_states[0].data[0] < 0.55f);
    assert(graph->node_states[0].data[0] > 0.54f);

    assert(graph->node_states[0].data[1] > 0.45f);
    assert(graph->node_states[0].data[1] < 0.46f);

    // clean up
    graph_destroy(graph);
}
