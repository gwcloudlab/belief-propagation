#include <stdio.h>
#include <assert.h>

#include "expression.h"
#include "Parser.h"
#include "Lexer.h"

int yyparse(struct expression ** expr, yyscan_t scanner);

struct expression * get_ast(const char * expr)
{
	struct expression * expression;
	yyscan_t scanner;
	YY_BUFFER_STATE state;

	assert(yylex_init(&scanner) == 0);

	state = yy_scan_string(expr, scanner);

	assert(yyparse(&expression, scanner) == 0);

	yy_delete_buffer(state, scanner);
	yylex_destroy(scanner);

	return expression;
}

int main(void)
{
	extern int yydebug;
	yydebug = 1;

	struct expression * expression = NULL;
	const char test[] = "// Bayesian Network in the Interchange Format\n// Produced by BayesianNetworks package in JavaBayes\n// Output created Sun Nov 02 17:49:49 GMT+00:00 1997\n// Bayesian network \nnetwork \"Dog-Problem\" { //5 variables and 5 probability distributions\nproperty \"credal-set constant-density-bounded 1.1\" ;\n}variable  \"light-on\" { //2 values\ntype discrete[2] {  \"true\"  \"false\" };\nproperty \"position = (218, 195)\" ;\n}";
	expression = get_ast(test);
}
