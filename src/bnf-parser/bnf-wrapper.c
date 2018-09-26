

#include "bnf-wrapper.h"

/**
 * Forward declaration of parse method; builds AST using expression and scanner
 * @param expr The BNF expression
 * @param scanner The BISON scanner
 * @return 0 if successful; any other int is an error
 */
int yyparse(struct expression ** expr, yyscan_t scanner);

/**
 * Tests parsing a string holding a BNF expression
 * @param expr A string holding the expression to be parsed
 */
void test_ast(const char * expr)
{
    // parse the file
    struct expression * expression;
    yyscan_t scanner;
    YY_BUFFER_STATE state;

    assert(yylex_init(&scanner) == 0);

    assert(scanner != NULL);
    assert(strlen(expr) > 0);

    state = yy_scan_string(expr, scanner);

    assert(yyparse(&expression, scanner) == 0);
    yy_delete_buffer(state, scanner);
    yylex_destroy(scanner);

    // make sure resulting expression is valid
    assert(expression != NULL);

    delete_expression(expression);
}

/**
 * Tests parsing a file in the path specified using the BNF Bison parser
 * @param file_path The path of the file to parse
 */
void test_file(const char * file_path)
{
    // parse file
    struct expression * expression;
    yyscan_t scanner;
    YY_BUFFER_STATE state;
    FILE * in;

    assert(yylex_init(&scanner) == 0);

    in = fopen(file_path, "r");

    assert(in != NULL);

    yyset_in(in, scanner);

    assert(yyparse(&expression, scanner) == 0);
    //yy_delete_buffer(state, scanner);
    yylex_destroy(scanner);

    fclose(in);

    // ensure it's not null
    assert(expression != NULL);

    delete_expression(expression);
}

/**
 * Tests parsing a file and then running non-loopy BP
 * @param file_name The path of the file to parse
 */
void test_parse_file(char * file_name){
    // parse file
    struct expression * expression;
    yyscan_t scanner;
    YY_BUFFER_STATE state;
    FILE * in;
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int i;

    assert(yylex_init(&scanner) == 0);

    in = fopen(file_name, "r");

    assert(in != NULL);

    yyset_in(in, scanner);

    assert(yyparse(&expression, scanner) == 0);
    //yy_delete_buffer(state, scanner);
    yylex_destroy(scanner);

    fclose(in);

    assert(expression != NULL);

    // run BP

    graph = build_graph(expression);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);

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

    // print time data

    time_elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%s,regular,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, time_elapsed);

    //print_nodes(graph);

    assert(graph != NULL);

    delete_expression(expression);

    graph_destroy(graph);
}

/**
 * Tests parsing and running loopy BP
 * @param file_name The path of the file to parse
 */
void test_loopy_belief_propagation(char * file_name){
    // parse file
    struct expression * expression;
    yyscan_t scanner;
    YY_BUFFER_STATE state;
    FILE * in;
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;

    assert(yylex_init(&scanner) == 0);

    in = fopen(file_name, "r");

    assert(in != NULL);

    yyset_in(in, scanner);

    assert(yyparse(&expression, scanner) == 0);
    //yy_delete_buffer(state, scanner);
    yylex_destroy(scanner);

    fclose(in);

    assert(expression != NULL);

    // set up graph data

    graph = build_graph(expression);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);

    start = clock();
    init_previous_edge(graph);

    // run loopy bp

    loopy_propagate_until(graph, 1E-16, 10000);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    printf("%s,loopy,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, time_elapsed);

    delete_expression(expression);

    graph_destroy(graph);
}

/**
 * Tests simply parsing a file
 * @param file_name The path to parse
 * @return A pointer to resulting BNF expression
 */
struct expression * parse_file(const char * file_name){
    struct expression * expression;
    yyscan_t scanner;
    YY_BUFFER_STATE state;
    FILE * in;

    assert(yylex_init(&scanner) == 0);

    in = fopen(file_name, "r");

    assert(in != NULL);

    yyset_in(in, scanner);

    assert(yyparse(&expression, scanner) == 0);
    //yy_delete_buffer(state, scanner);
    yylex_destroy(scanner);

    fclose(in);

    assert(expression != NULL);

    return expression;
}

/**
 * Runs regular BP on the expression
 * @param expression The parsed expression
 * @param file_name The name of the file for display purposes
 */
void run_test_belief_propagation(struct expression * expression, const char * file_name){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int i;

    // set up data

    graph = build_graph(expression);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    calculate_diameter(graph);

    // run BP

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

    // print timing data

    time_elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%s,regular,%d,%d,%d,2,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, time_elapsed);

    //print_nodes(graph);

    graph_destroy(graph);
}

/**
 * Tests running loopy BP using the given expression
 * @param expression The parsed expression
 * @param file_name The name of the file to display
 */
void run_test_loopy_belief_propagation(struct expression * expression, const char * file_name){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    // set up data

    graph = build_graph(expression);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    calculate_diameter(graph);

    // run loopy bp

    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;

    printf("%s,loopy,%d,%d,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    //print_nodes(graph);

    graph_destroy(graph);
}

/**
 * Runs regular BP using the given file for the number of iterations provided
 * @param file_name The path to the file to parse
 * @param num_iterations The number of iterations to run BP for; set > 1 for analytical purposes
 */
void run_tests_with_file(const char * file_name, unsigned int num_iterations){
    unsigned int i;
    struct expression * expr;

    expr = parse_file(file_name);
    for(i = 0; i < num_iterations; ++i){
        run_test_belief_propagation(expr, file_name);
    }

    for(i = 0; i < num_iterations; ++i){
        run_test_loopy_belief_propagation(expr, file_name);
    }

    delete_expression(expr);
}

/**
 * @brief Helper function for handling file paths (useful for test + src dir differences)
 * @details Copies the prefix and then concats file in the provided buffer
 * @param prefix The directory prefix path to use
 * @param file The file to use
 * @param buffer The buffer holding the string to write to
 * @param length The length of the buffer
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
 * Runs a basic suite tests
 * @param root_dir The root dir of the files to read
 */
void basic_test_suite(const char * root_dir){
    //extern int yydebug;
    //yydebug = 1;

	struct expression * expression = NULL;
	const char test[] = "// Bayesian Network in the Interchange Format\n// Produced by BayesianNetworks package in JavaBayes\n// Output created Sun Nov 02 17:49:49 GMT+00:00 1997\n// Bayesian network \nnetwork \"Dog-Problem\" { //5 variables and 5 probability distributions\nproperty \"credal-set constant-density-bounded 1.1\" ;\n}variable  \"light-on\" { //2 values\ntype discrete[2] {  \"true\"  \"false\" };\nproperty \"position = (218, 195)\" ;\n}\nvariable  \"bowel-problem\" { //2 values\ntype discrete[2] {  \"true\"  \"false\" };\nproperty \"position = (335, 99)\" ;\n}";
	test_ast(test);

    char file_name_buffer[128];

    printf("File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)\n");
  	test_parse_file(build_full_file_path(root_dir, "dog.bif", file_name_buffer, 128));
	test_parse_file(build_full_file_path(root_dir, "alarm.bif", file_name_buffer, 128));

	test_parse_file(build_full_file_path(root_dir, "very_large/andes.bif", file_name_buffer, 128));
	test_loopy_belief_propagation(build_full_file_path(root_dir, "very_large/andes.bif", file_name_buffer, 128));

    // diabetes is too big -> adjust
	//test_parse_file(build_full_file_path(root_dir, "Diabetes.bif", file_name_buffer, 128));
	//test_loopy_belief_propagation(build_full_file_path(root_dir, "Diabetes.bif", file_name_buffer, 128));

	test_loopy_belief_propagation(build_full_file_path(root_dir, "dog.bif", file_name_buffer, 128));
	test_loopy_belief_propagation(build_full_file_path(root_dir, "alarm.bif", file_name_buffer, 128));

    test_file(build_full_file_path(root_dir, "dog.bif", file_name_buffer, 128));
    test_file(build_full_file_path(root_dir, "alarm.bif", file_name_buffer, 128));

    expression = parse_file(build_full_file_path(root_dir, "alarm.bif", file_name_buffer, 128));

    assert(expression != NULL);

    delete_expression(expression);
}

/**
 * Runs a test suite of the small BNF networks
 * @param root_dir The path of the dir holding the bif files
 */
void small_test_suite(const char * root_dir){
    char file_name_buffer[128];

    printf("File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)\n");
    run_tests_with_file(build_full_file_path(root_dir, "asia.bif", file_name_buffer, 128), 1);
    run_tests_with_file(build_full_file_path(root_dir, "cancer.bif", file_name_buffer, 128), 1);
    run_tests_with_file(build_full_file_path(root_dir, "earthquake.bif", file_name_buffer, 128), 1);
    run_tests_with_file(build_full_file_path(root_dir, "sachs.bif", file_name_buffer, 128), 1);
    run_tests_with_file(build_full_file_path(root_dir, "survey.bif", file_name_buffer, 128), 1);
}

/**
 * Runs tests against the medium sized BNF bif files
 * @param root_dir The path to the directory holding the files
 */
void medium_test_suite(const char * root_dir){
    char file_name_buffer[128];

    printf("File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)\n");
    run_tests_with_file(build_full_file_path(root_dir, "alarm.bif", file_name_buffer, 128), 1);
    // increase MAX_STATES for commented out tests
    //run_tests_with_file(build_full_file_path(root_dir, "barley.bif", file_name_buffer, 128), 1);
    //run_tests_with_file(build_full_file_path(root_dir, "child.bif", file_name_buffer, 128), 1);
    //run_tests_with_file(build_full_file_path(root_dir, "hailfinder.bif", file_name_buffer, 128), 1);
    run_tests_with_file(build_full_file_path(root_dir, "insurance.bif", file_name_buffer, 128), 1);
    //run_tests_with_file(build_full_file_path(root_dir, "mildew.bif", file_name_buffer, 128), 1);
    run_tests_with_file(build_full_file_path(root_dir, "water.bif", file_name_buffer, 128), 1);
}

/**
 * Runs tests against the large sized BNF bif files
 * @param root_dir The path to the directory holding the files
 */
void large_test_suite(const char * root_dir){
    char file_name_buffer[128];

    printf("File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)\n");
    run_tests_with_file(build_full_file_path(root_dir, "hepar2.bif", file_name_buffer, 128), 1);
    run_tests_with_file(build_full_file_path(root_dir, "win95pts.bif", file_name_buffer, 128), 1);
}

/**
 * Runs tests against the very large BNF bif files
 * @param root_dir The path to the directory holding the files
 */
void very_large_test_suite(const char *root_dir){
    char file_name_buffer[128];

    printf("File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)\n");
    run_tests_with_file(build_full_file_path(root_dir, "andes.bif", file_name_buffer, 128), 1);
    // Increase MAX_STATES for diabetes
    //run_tests_with_file(build_full_file_path(root_dir, "diabetes.bif", file_name_buffer, 128), 1);
    run_tests_with_file(build_full_file_path(root_dir, "link.bif", file_name_buffer, 128), 1);
    //run_tests_with_file(build_full_file_path(root_dir, "munin1.bif", file_name_buffer, 128), 1);
    //run_tests_with_file(build_full_file_path(root_dir, "munin2.bif", file_name_buffer, 128), 1);
    //run_tests_with_file(build_full_file_path(root_dir, "munin3.bif", file_name_buffer, 128), 1);
    //run_tests_with_file(build_full_file_path(root_dir, "munin4.bif", file_name_buffer, 128), 1);
    //run_tests_with_file(build_full_file_path(root_dir, "pathfinder.bif", file_name_buffer, 128), 1);
    // increase max_degree
    //run_tests_with_file(build_full_file_path(root_dir, "pigs.bif", file_name_buffer, 128), 1);

}



