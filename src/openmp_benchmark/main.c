#include <stdio.h>
#include <assert.h>
#include <time.h>

#include "../bnf-parser/expression.h"
#include "../bnf-parser/Parser.h"
#include "../bnf-parser/Lexer.h"
#include "../bnf-xml-parser/xml-expression.h"

int yyparse(struct expression ** expr, yyscan_t scanner);

void test_ast(const char * expr)
{
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

	assert(expression != NULL);

	delete_expression(expression);
}

void test_file(const char * file_path)
{
	struct expression * expression;
	yyscan_t scanner;
	YY_BUFFER_STATE state;
	FILE * in;

	assert(yylex_init(&scanner) == 0);

	in = fopen(file_path, "r");

	yyset_in(in, scanner);

	assert(yyparse(&expression, scanner) == 0);
	//yy_delete_buffer(state, scanner);
	yylex_destroy(scanner);

	fclose(in);

	assert(expression != NULL);

	delete_expression(expression);
}

void test_parse_file(char * file_name){
	unsigned int i;
	struct expression * expression;
	yyscan_t scanner;
	YY_BUFFER_STATE state;
	FILE * in;
	Graph_t graph;
	clock_t start, end;
	double time_elapsed;

	assert(yylex_init(&scanner) == 0);

	in = fopen(file_name, "r");

	yyset_in(in, scanner);

	assert(yyparse(&expression, scanner) == 0);
	//yy_delete_buffer(state, scanner);
	yylex_destroy(scanner);

	fclose(in);

	assert(expression != NULL);

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

	time_elapsed = (double)(end - start) / CLOCKS_PER_SEC;
	printf("%s,regular,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, time_elapsed);

    //print_nodes(graph);

	assert(graph != NULL);

	delete_expression(expression);

	graph_destroy(graph);
}

void test_loopy_belief_propagation(char * file_name){
		struct expression * expression;
		yyscan_t scanner;
		YY_BUFFER_STATE state;
		FILE * in;
		Graph_t graph;
		clock_t start, end;
		double time_elapsed;

		assert(yylex_init(&scanner) == 0);

		in = fopen(file_name, "r");

		yyset_in(in, scanner);

		assert(yyparse(&expression, scanner) == 0);
		//yy_delete_buffer(state, scanner);
		yylex_destroy(scanner);

		fclose(in);

		assert(expression != NULL);

		graph = build_graph(expression);
		assert(graph != NULL);
		//print_nodes(graph);
		//print_edges(graph);

		set_up_src_nodes_to_edges(graph);
		set_up_dest_nodes_to_edges(graph);

		start = clock();
		init_previous_edge(graph);

		loopy_propagate_until(graph, PRECISION, NUM_ITERATIONS);
		end = clock();

		time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
		//print_nodes(graph);
		printf("%s,loopy,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, time_elapsed);

		delete_expression(expression);

		graph_destroy(graph);
}

struct expression * parse_file(const char * file_name){
    struct expression * expression;
    yyscan_t scanner;
    YY_BUFFER_STATE state;
    FILE * in;

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

void run_test_belief_propagation(struct expression * expression, const char * file_name, FILE * out){
	Graph_t graph;
	clock_t start, end;
	double time_elapsed;
	unsigned int i;

	graph = build_graph(expression);
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


void run_test_loopy_belief_propagation(struct expression * expression, const char * file_name, FILE * out){
	Graph_t graph;
	clock_t start, end;
	double time_elapsed;
	unsigned int num_iterations;

	graph = build_graph(expression);
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


void run_tests_with_file(const char * file_name, unsigned int num_iterations, FILE * out){
    unsigned int i;
    struct expression * expr;

    expr = parse_file(file_name);
    for(i = 0; i < num_iterations; ++i){
        run_test_belief_propagation(expr, file_name, out);
    }

    for(i = 0; i < num_iterations; ++i){
        run_test_loopy_belief_propagation(expr, file_name, out);
    }

    delete_expression(expr);
}

void run_tests_with_xml_file(const char * file_name, unsigned int num_iterations, FILE * out){
    unsigned int i;
    /*for(i = 0; i < num_iterations; ++i){
        run_test_belief_propagation(expr, file_name, out);
    }*/

    for(i = 0; i < num_iterations; ++i){
        run_test_loopy_belief_propagation_xml_file(file_name, out);
    }
}


int main(void)
{
/*
	extern int yydebug;
	yydebug = 1;
/*
	struct expression * expression = NULL;
	const char test[] = "// Bayesian Network in the Interchange Format\n// Produced by BayesianNetworks package in JavaBayes\n// Output created Sun Nov 02 17:49:49 GMT+00:00 1997\n// Bayesian network \nnetwork \"Dog-Problem\" { //5 variables and 5 probability distributions\nproperty \"credal-set constant-density-bounded 1.1\" ;\n}variable  \"light-on\" { //2 values\ntype discrete[2] {  \"true\"  \"false\" };\nproperty \"position = (218, 195)\" ;\n}\nvariable  \"bowel-problem\" { //2 values\ntype discrete[2] {  \"true\"  \"false\" };\nproperty \"position = (335, 99)\" ;\n}";
	test_ast(test);

  	test_parse_file("dog.bif");
	test_parse_file("alarm.bif");

	test_parse_file("very_large/andes.bif");
	test_loopy_belief_propagation("very_large/andes.bif");

	test_parse_file("Diabetes.bif");
	test_loopy_belief_propagation("Diabetes.bif");

	test_loopy_belief_propagation("dog.bif");
	test_loopy_belief_propagation("alarm.bif");
*/
	//test_file("dog.bif");
	//test_file("alarm.bif");

	/*expression = read_file("alarm.bif");

	assert(expression != NULL);

	delete_expression(expression);*/

    FILE * out = fopen("openmp_benchmark.csv", "w");
	fprintf(out, "File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)\n");
    fflush(out);
/*
 *
	run_tests_with_file("../benchmark_files/small/asia.bif", 1);
	run_tests_with_file("../benchmark_files/small/cancer.bif", 1);
	run_tests_with_file("../benchmark_files/small/earthquake.bif", 1);
	run_tests_with_file("../benchmark_files/small/sachs.bif", 1);
	run_tests_with_file("../benchmark_files/small/survey.bif", 1);

	run_tests_with_file("../benchmark_files/medium/alarm.bif", 1);
	run_tests_with_file("../benchmark_files/medium/barley.bif", 1);
	//run_tests_with_file("../benchmark_files/medium/child.bif", 1);
	run_tests_with_file("../benchmark_files/medium/hailfinder.bif", 1);
	run_tests_with_file("../benchmark_files/medium/insurance.bif", 1);
	run_tests_with_file("../benchmark_files/medium/mildew.bif", 1);
	run_tests_with_file("../benchmark_files/medium/water.bif", 1);

	run_tests_with_file("../benchmark_files/large/hepar2.bif", 1);
	run_tests_with_file("../benchmark_files/large/win95pts.bif", 1);

    run_tests_with_file("../benchmark_files/very_large/andes.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/diabetes.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/link.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/munin1.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/munin2.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/munin3.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/munin4.bif", 1);
	//run_tests_with_file("../benchmark_files/very_large/munin.bif", 1);
	run_tests_with_file("../benchmark_files/very_large/pathfinder.bif", 1);
    run_tests_with_file("../benchmark_files/very_large/pigs.bif", 1);

    run_tests_with_xml_file("../benchmark_files/xml/bf_1000_2000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_1000_2000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_1000_2000_3.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_2000_4000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_2000_4000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_2000_4000_3.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_5000_10000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_5000_10000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_5000_10000_3.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_10000_20000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_10000_20000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_10000_20000_3.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_12000_24000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_12000_24000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_12000_24000_3.xml", 1);*/

	/*run_tests_with_xml_file("../benchmark_files/xml/bf_15000_30000_1.xml", 1);
	run_tests_with_xml_file("../benchmark_files/xml/bf_15000_30000_2.xml", 1);
	run_tests_with_xml_file("../benchmark_files/xml/bf_15000_30000_3.xml", 1);

	run_tests_with_xml_file("../benchmark_files/xml/bf_20000_40000_1.xml", 1);
	run_tests_with_xml_file("../benchmark_files/xml/bf_20000_40000_2.xml", 1);
	run_tests_with_xml_file("../benchmark_files/xml/bf_20000_40000_3.xml", 1);

	run_tests_with_xml_file("../benchmark_files/xml/bf_25000_50000_1.xml", 1);
	run_tests_with_xml_file("../benchmark_files/xml/bf_25000_50000_2.xml", 1);
	run_tests_with_xml_file("../benchmark_files/xml/bf_25000_50000_3.xml", 1);

	run_tests_with_xml_file("../benchmark_files/xml/bf_30000_60000_1.xml", 1);
	run_tests_with_xml_file("../benchmark_files/xml/bf_30000_60000_2.xml", 1);
	run_tests_with_xml_file("../benchmark_files/xml/bf_30000_60000_3.xml", 1);

    run_tests_with_xml_file("../benchmark_files/xml/bf_40000_80000_1.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_40000_80000_2.xml", 1);
    run_tests_with_xml_file("../benchmark_files/xml/bf_40000_80000_3.xml", 1);

    run_tests_with_xml_file("../benchmark_files/xml/bf_80000_160000_2.xml", 1);*/

    run_tests_with_xml_file("../benchmark_files/xml2/10_20.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/100_200.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/1000_2000.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/10000_20000.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/100000_200000.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/200000_400000.xml", 1, out);
    //run_tests_with_xml_file("../benchmark_files/xml2/300000_600000.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/400000_800000.xml", 1, out);
    //run_tests_with_xml_file("../benchmark_files/xml2/500000_1000000.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/600000_1200000.xml", 1, out);
    //run_tests_with_xml_file("../benchmark_files/xml2/700000_1400000.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/800000_1600000.xml", 1, out);
    //run_tests_with_xml_file("../benchmark_files/xml2/900000_1800000.xml", 1, out);
    run_tests_with_xml_file("../benchmark_files/xml2/1000000_2000000.xml", 1, out);
    //run_tests_with_xml_file("../benchmark_files/xml2/10000000_20000000.xml", 1, out);

    fclose(out);

	return 0;
}
