%{
#include "Parser.h"
#include "Lexer.h"
#include "expression.h"
#include "../constants.h"

#define YYDEBUG 1

int yyerror(YYLTYPE * yyltype, struct expression ** expression, yyscan_t scanner, const char *msg){
	fprintf(stderr, "Error on line: %d and column: %d: %s\n", yyltype->last_line, yyltype->last_column, msg);
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
%locations
%lex-param   { yyscan_t scanner }
%parse-param { struct expression ** expression }
%parse-param { yyscan_t scanner }

%union {
	char word[50];
	int int_value;
	float float_value;
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
%token TOKEN_L_PARENS
%token TOKEN_R_PARENS
%token TOKEN_SEMICOLON
%token <int_value> TOKEN_DECIMAL_LITERAL
%token <float_value> TOKEN_FLOATING_POINT_LITERAL
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
%type <expression> probability_names_list
%type <expression> probability_declaration
%type <expression> probability_variables_list
%type <expression> probability_content
%type <expression> probability_content_entries
%type <expression> probability_default_entry
%type <expression> probability_entry
%type <expression> probability_table
%type <expression> floating_point_list
%type <expression> probability_values_list
%type <expression> probability_values

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
	: TOKEN_NETWORK TOKEN_WORD network_content { struct expression * network_expr = create_expression(NETWORK_DECLARATION, $3, NULL);
												strncpy(network_expr->value, $2, CHAR_BUFFER_SIZE);
												$$ =  network_expr;
												}
	;
	
network_content
	: TOKEN_L_CURLY_BRACE TOKEN_R_CURLY_BRACE { $$ = create_expression( NETWORK_CONTENT, NULL, NULL); }
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
	: variable_declaration { $$ = create_expression(VARIABLE_OR_PROBABILITY_DECLARATION, $1, NULL); }
	| probability_declaration { $$ = create_expression(VARIABLE_OR_PROBABILITY_DECLARATION, $1, NULL); }
	| variable_or_probability_declaration variable_declaration { $$ = create_expression(VARIABLE_OR_PROBABILITY_DECLARATION, $1, $2); }
	| variable_or_probability_declaration probability_declaration { $$ = create_expression(VARIABLE_OR_PROBABILITY_DECLARATION, $1, $2); } 
	| %empty { }
	
variable_declaration
	: TOKEN_VARIABLE TOKEN_WORD variable_content { struct expression * expr = create_expression(VARIABLE_DECLARATION, $3, NULL);
													strncpy(expr->value, $2, CHAR_BUFFER_SIZE);
													$$ = expr; 
													}

variable_content
	: TOKEN_L_CURLY_BRACE TOKEN_R_CURLY_BRACE {}
	| TOKEN_L_CURLY_BRACE property_or_variable_discrete TOKEN_R_CURLY_BRACE { $$ = create_expression(VARIABLE_CONTENT, $2, NULL);  }
	
property_or_variable_discrete
	: property { $$ = create_expression(VARIABLE_OR_PROBABILITY, $1, NULL); }
	| variable_discrete { $$ = create_expression(VARIABLE_OR_PROBABILITY, $1, NULL); }
	| property_or_variable_discrete property { $$ = create_expression(VARIABLE_OR_PROBABILITY, $1, $2); }
	| property_or_variable_discrete variable_discrete { $$ = create_expression(VARIABLE_OR_PROBABILITY, $1, $2); }
	
variable_discrete
	: TOKEN_VARIABLETYPE TOKEN_DISCRETE TOKEN_L_BRACKET TOKEN_DECIMAL_LITERAL TOKEN_R_BRACKET TOKEN_L_CURLY_BRACE variable_values_list TOKEN_R_CURLY_BRACE TOKEN_SEMICOLON {
																																								struct expression * variable_discrete = create_expression(VARIABLE_DISCRETE, $7, NULL);
																																								variable_discrete->int_value = $4;
																																								$$ = variable_discrete;
																																							}
variable_values_list
	: TOKEN_WORD { struct expression * values_list = create_expression(VARIABLE_VALUES_LIST, NULL, NULL);
					strncpy(values_list->value, $1, CHAR_BUFFER_SIZE);
					$$ = values_list;
					}
	| TOKEN_DECIMAL_LITERAL {struct expression * values_list = create_expression(VARIABLE_VALUES_LIST, NULL, NULL);
                             					snprintf(values_list->value, CHAR_BUFFER_SIZE, "%d", $1);
                             					$$ = values_list;
	                        }
	| variable_values_list TOKEN_WORD { struct expression * values_list = create_expression(VARIABLE_VALUES_LIST, $1, NULL);
										strncpy(values_list->value, $2, CHAR_BUFFER_SIZE);
										$$ = values_list;
										}
    | variable_values_list TOKEN_DECIMAL_LITERAL {struct expression * values_list = create_expression(VARIABLE_VALUES_LIST, $1, NULL);
                                                                              					snprintf(values_list->value, CHAR_BUFFER_SIZE, "%d", $2);
                                                                              					$$ = values_list;
                                                 	                        }
										
probability_declaration
	: TOKEN_PROBABILITY probability_variables_list probability_content { $$ = create_expression(PROBABILITY_DECLARATION, $2, $3); }
	
probability_variables_list		
	: TOKEN_L_PARENS probability_names_list TOKEN_R_PARENS { $$ = create_expression(PROBABILITY_VARIABLES_LIST, $2, NULL); }
	
probability_names_list
	: TOKEN_WORD { struct expression * names_list = create_expression(PROBABILITY_VARIABLE_NAMES, NULL, NULL);
				   strncpy(names_list->value, $1, CHAR_BUFFER_SIZE);
				   $$ = names_list;
				   }
	| TOKEN_DECIMAL_LITERAL {struct expression * names_list = create_expression(PROBABILITY_VARIABLE_NAMES, NULL, NULL);
                             snprintf(names_list->value, CHAR_BUFFER_SIZE, "%d", $1);
                            $$ = names_list;
	                        }
	| probability_names_list TOKEN_WORD {struct expression * names_list = create_expression(PROBABILITY_VARIABLE_NAMES, $1, NULL);
										   strncpy(names_list->value, $2, CHAR_BUFFER_SIZE);
										   $$ = names_list;
										}
	| probability_names_list TOKEN_DECIMAL_LITERAL {struct expression * names_list = create_expression(PROBABILITY_VARIABLE_NAMES, $1, NULL);
                                                                                snprintf(names_list->value, CHAR_BUFFER_SIZE, "%d", $2);
                                                                               $$ = names_list;
                                                   	                        }
										
probability_content
	: TOKEN_L_CURLY_BRACE TOKEN_R_CURLY_BRACE {}
	| TOKEN_L_CURLY_BRACE probability_content_entries TOKEN_R_CURLY_BRACE { $$ = create_expression(PROBABILITY_CONTENT, $2, NULL); } 
	
probability_content_entries
	: property_list { $$ = create_expression(PROBABILITY_CONTENT_LIST, $1, NULL); }
	| probability_content_entries property_list { $$ = create_expression(PROBABILITY_CONTENT_LIST, $1, $2); }
	| probability_default_entry { $$ = create_expression(PROBABILITY_CONTENT_LIST, $1, NULL); }
	| probability_content_entries probability_default_entry { $$ = create_expression(PROBABILITY_CONTENT_LIST, $1, $2); }
	| probability_entry { $$ = create_expression(PROBABILITY_CONTENT_LIST, $1, NULL); }
	| probability_content_entries probability_entry { $$ = create_expression(PROBABILITY_CONTENT_LIST, $1, $2); }
	| probability_table { $$ = create_expression(PROBABILITY_CONTENT_LIST, $1, NULL); }
	| probability_content_entries probability_table { $$ = create_expression(PROBABILITY_CONTENT_LIST, $1, $2); }

probability_default_entry
	: TOKEN_DEFAULTVALUE floating_point_list TOKEN_SEMICOLON { $$ = create_expression(PROBABILITY_DEFAULT_ENTRY, $2, NULL); }

probability_entry
	: probability_values_list floating_point_list TOKEN_SEMICOLON { $$ = create_expression(PROBABILITY_ENTRY, $1, $2); }

probability_values_list
	: TOKEN_L_PARENS probability_values TOKEN_R_PARENS { $$ = create_expression(PROBABILITY_VALUES_LIST, $2, NULL); }
	
probability_values
	: TOKEN_WORD { struct expression * values_list = create_expression(PROBABILITY_VALUES, NULL, NULL); 
					strncpy(values_list->value, $1, CHAR_BUFFER_SIZE); 
					$$ = values_list; 
					}
    | TOKEN_DECIMAL_LITERAL {
                            struct expression * values_list = create_expression(PROBABILITY_VALUES, NULL, NULL);
                            					snprintf(values_list->value, CHAR_BUFFER_SIZE, "%d", $1);
                            					$$ = values_list;
    					}
	| probability_values TOKEN_WORD { struct expression * values_list = create_expression(PROBABILITY_VALUES, $1, NULL);
										strncpy(values_list->value, $2, CHAR_BUFFER_SIZE);
										$$ = values_list;
										}
	| probability_values TOKEN_DECIMAL_LITERAL {
                                                                           struct expression * values_list = create_expression(PROBABILITY_VALUES, $1, NULL);
                                                                           					snprintf(values_list->value, CHAR_BUFFER_SIZE, "%d", $2);
                                                                           					$$ = values_list;
                                                   					}

probability_table
	: TOKEN_TABLEVALUES floating_point_list TOKEN_SEMICOLON { $$ = create_expression(PROBABILITY_TABLE, $2, NULL); }

floating_point_list
	: TOKEN_FLOATING_POINT_LITERAL { struct expression * fp_list = create_expression(FLOATING_POINT_LIST, NULL, NULL);
									 fp_list->float_value = $1;
									 $$ = fp_list;
									}
	| TOKEN_DECIMAL_LITERAL         {struct expression * fp_list = create_expression(FLOATING_POINT_LIST, NULL, NULL);
                                     									 fp_list->float_value = (float)$1;
                                     									 $$ = fp_list;
	                                }
	| floating_point_list TOKEN_FLOATING_POINT_LITERAL {
	                                                    struct expression * fp_list = create_expression(FLOATING_POINT_LIST, $1, NULL);
														 fp_list->float_value = $2;
														 $$ = fp_list;
														}

    | floating_point_list TOKEN_DECIMAL_LITERAL {
                                                            struct expression * fp_list = create_expression(FLOATING_POINT_LIST, $1, NULL);
                                                             fp_list->float_value = (float)$2;
                                                             $$ = fp_list;
                                                            }