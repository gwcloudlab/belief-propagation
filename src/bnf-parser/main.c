#include <stdio.h>
#include <assert.h>
#include <time.h>

#include "expression.h"
#include "Parser.h"
#include "Lexer.h"

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
	int i;
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

		loopy_propagate_until(graph, 1E-15, 10000);
		end = clock();

		time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
		//print_nodes(graph);
		printf("%s,loopy,%d,%d,%lf\n", file_name, graph->current_num_vertices, graph->current_num_edges, time_elapsed);

		delete_expression(expression);

		graph_destroy(graph);
}


int main(void)
{
/*
	extern int yydebug;
	yydebug = 1;

	struct expression * expression = NULL;
	const char test[] = "// Bayesian Network in the Interchange Format\n// Produced by BayesianNetworks package in JavaBayes\n// Output created Sun Nov 02 17:49:49 GMT+00:00 1997\n// Bayesian network \nnetwork \"Dog-Problem\" { //5 variables and 5 probability distributions\nproperty \"credal-set constant-density-bounded 1.1\" ;\n}variable  \"light-on\" { //2 values\ntype discrete[2] {  \"true\"  \"false\" };\nproperty \"position = (218, 195)\" ;\n}\nvariable  \"bowel-problem\" { //2 values\ntype discrete[2] {  \"true\"  \"false\" };\nproperty \"position = (335, 99)\" ;\n}";
	test_ast(test);

  	test_parse_file("dog.bif");
	test_parse_file("alarm.bif");

	test_parse_file("very_large/andes.bif");
	test_loopy_belief_propagation("very_large/andes.bif");
*/
	test_parse_file("Diabetes.bif");
	test_loopy_belief_propagation("Diabetes.bif");
/*
	test_loopy_belief_propagation("dog.bif");
	test_loopy_belief_propagation("alarm.bif");
*/
	//test_file("dog.bif");
	//test_file("alarm.bif");

	/*expression = read_file("alarm.bif");

	assert(expression != NULL);

	delete_expression(expression);*/

	return 0;
}
