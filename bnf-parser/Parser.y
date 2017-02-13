%{
#include "Parser.h"
#include "Lexer.h"
#include "expression.h"

#define YYDEBUG 1

int yyerror(struct expression ** expression, yyscan_t scanner, const char *msg){
	fprintf(stderr, "Error:%s\n", msg);
	return 0;
}

%}

%code requires {

#ifndef YY_TYPEDEF_YY_SCANNER_T
#define YY_TYPEDEF_YY_SCANNER_T
typedef void* yyscan_t;
#endif

}

%output "Parser.c"
%defines "Parser.h"

%define api.pure
%lex-param   { yyscan_t scanner }
%parse-param { struct expression ** expression }
%parse-param { yyscan_t scanner }

%union {
	char word[20];
	int int_value;
	double double_value;
	struct expression * expression;
}

%token TOKEN_NETWORK
%token TOKEN_VARIABLE
%token TOKEN_PROBABILITY
%token TOKEN_PROPERTY
%token TOKEN_VARIABLETYPE
%token TOKEN_DISCRETE
%token TOKEN_DEFAULTVALUE
%token TOKEN_TABLEVALUES
%token TOKEN_L_CURLY_BRACE
%token TOKEN_R_CURLY_BRACE
%token TOKEN_SEMICOLON
%token <double_value> TOKEN_DECIMAL_LITERAL
%token <double_value> TOKEN_FLOATING_POINT_LITERAL
%token <word> TOKEN_WORD
%token <word> TOKEN_PROPERTY_LITERAL

%type <expression> expr
%type <expression> compilation_unit
%type <expression> property
%type <expression> property_list
%type <expression> network_content
%type <expression> network_declaration

%%

input
	: expr { *expression = $1; }
	;
	
expr: compilation_unit { $$ = $1; }
	;

compilation_unit
	: network_declaration variable_or_probability_declaration { $$ = create_expression(COMPILATION_UNIT, $1, $2); }
	;
	
network_declaration
	: TOKEN_NETWORK TOKEN_WORD network_content { $$ = create_expression(NETWORK_DECLARATION, $3, NULL); }
	;
	
network_content
	: TOKEN_L_CURLY_BRACE TOKEN_R_CURLY_BRACE { }
	| TOKEN_L_CURLY_BRACE property_list TOKEN_R_CURLY_BRACE { $$ = create_expression( NETWORK_CONTENT, $2, NULL ); }
	;
	
property_list
	: property { $$ = create_expression(PROPERTY_LIST, $1, NULL); }
	| property_list property { $$ = create_expression(PROPERTY_LIST, $1, $2); }

property
	: TOKEN_PROPERTY { $$ = create_expression(PROPERTY, NULL, NULL); } 
	
variable_or_probability_declaration