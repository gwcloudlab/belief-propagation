%{
#include "Parser.h"
#include "Lexer.h"
#include "expression.h"
#include "../constants.h"

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
	char word[50];
	int int_value;
	double double_value;
	struct expression * expression;
}

%token TOKEN_NETWORK
%token TOKEN_VARIABLE
%token TOKEN_PROBABILITY
%token TOKEN_VARIABLETYPE
%token TOKEN_DISCRETE
%token TOKEN_DEFAULTVALUE
%token TOKEN_TABLEVALUES
%token TOKEN_L_CURLY_BRACE
%token TOKEN_R_CURLY_BRACE
%token TOKEN_L_BRACKET
%token TOKEN_R_BRACKET
%token TOKEN_SEMICOLON
%token <double_value> TOKEN_DECIMAL_LITERAL
%token <double_value> TOKEN_FLOATING_POINT_LITERAL
%token <word> TOKEN_WORD
%token <word> TOKEN_PROPERTY

%type <expression> expr
%type <expression> compilation_unit
%type <expression> property
%type <expression> property_list
%type <expression> network_content
%type <expression> network_declaration
%type <expression> variable_or_probability_declaration
%type <expression> variable_declaration
%type <expression> variable_content
%type <expression> variable_values_list
%type <expression> variable_discrete
%type <expression> property_or_variable_discrete

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
	: TOKEN_PROPERTY {
							struct expression * property_expression =  create_expression(PROPERTY, NULL, NULL);
							strncpy(property_expression->value, $1, CHAR_BUFFER_SIZE);
							$$ =  property_expression;
						} 
	
variable_or_probability_declaration
	: variable_declaration { $$ = create_expression(VARIABLE_DECLARATION, $1, NULL); }
	| variable_or_probability_declaration variable_declaration { $$ = create_expression(VARIABLE_OR_PROBABILITY_DECLARATION, $1, $2); }
	| %empty { }
	
variable_declaration
	: TOKEN_VARIABLE TOKEN_WORD variable_content { $$ = create_expression(VARIABLE_DECLARATION, $3, NULL); }

variable_content
	: TOKEN_L_CURLY_BRACE TOKEN_R_CURLY_BRACE {}
	| TOKEN_L_CURLY_BRACE property_or_variable_discrete TOKEN_R_CURLY_BRACE { $$ = create_expression(VARIABLE_CONTENT, $2, NULL);  }
	
property_or_variable_discrete
	: property { $$ = create_expression(VARIABLE_OR_PROBABILITY, $1, NULL); }
	| variable_discrete { $$ = create_expression(VARIABLE_DISCRETE, $1, NULL); }
	| property_or_variable_discrete property { $$ = create_expression(VARIABLE_OR_PROBABILITY, $1, $2); }
	| property_or_variable_discrete variable_discrete { $$ = create_expression(VARIABLE_OR_PROBABILITY, $1, $2); }
	
variable_discrete
	: TOKEN_VARIABLETYPE TOKEN_DISCRETE TOKEN_L_BRACKET TOKEN_DECIMAL_LITERAL TOKEN_R_BRACKET TOKEN_L_CURLY_BRACE variable_values_list TOKEN_R_CURLY_BRACE {
																																								struct expression * variable_discrete = create_expression(VARIABLE_DISCRETE, $7, NULL);
																																								variable_discrete->double_value = $4;
																																								$$ = variable_discrete;
																																							}
variable_values_list
	: TOKEN_WORD { struct expression * values_list = create_expression(VARIABLE_VALUES_LIST, NULL, NULL);
					strncpy(values_list->value, $1, CHAR_BUFFER_SIZE);
					$$ = values_list;
					}
	| variable_values_list TOKEN_WORD { struct expression * values_list = create_expression(VARIABLE_VALUES_LIST, $1, NULL);
										strncpy(values_list->value, $2, CHAR_BUFFER_SIZE);
										$$ = values_list;
										}																																							