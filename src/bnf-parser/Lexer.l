%{

#include <stdio.h>

#include "expression.h"
#include "Parser.h"

#define YY_USER_ACTION yylloc->first_line = yylineno; yylloc->first_column = yycolumn; yycolumn=yycolumn+yyleng; yylloc->last_column=yycolumn;  yylloc->last_line = yylineno;

%}

%option outfile="Lexer.c" header-file="Lexer.h"
%option warn nodefault
 
%option reentrant noyywrap never-interactive nounistd
%option bison-bridge
%option bison-locations
%option yylineno

NETWORK "network"
VARIABLE "variable"
PROBABILITY "probability"
PROPERTY "property"[^;]*;
VARIABLETYPE "type"
DISCRETE "discrete"
DEFAULTVALUE "default"
TABLEVALUES "table"

L_CURLY_BRACE "{"
R_CURLY_BRACE "}"
L_BRACKET "["
R_BRACKET "]"
L_PARENS "("
R_PARENS ")"
SEMICOLON ";"

WS [ \n\t\r]+
COMMA [,]
PIPE [|]
SINGLE_LINE_COMMENT [/]{2}[^\n\r]*(\n|\r\n)
BLOCK_COMMENT [/][*][^*]*\*([^/][^*]*\*)*[/]


WORD [0-9]*[a-zA-Z_\-]+[0-9a-zA-Z_\-]*

DECIMAL_LITERAL [0-9]+
EXPONENT [eE][+\-]?[0-9]+
FLOATING_POINT_LITERAL ([0-9]+\.[0-9]+({EXPONENT})?)|(\.[0-9]+({EXPONENT})?)|([0-9]+({EXPONENT}))

%%


{WS} { }
{COMMA} { }
{PIPE} { }
{SINGLE_LINE_COMMENT} { }
{BLOCK_COMMENT} { }
{NETWORK} { return TOKEN_NETWORK; }
{VARIABLE} { return TOKEN_VARIABLE; }
{PROBABILITY} {return TOKEN_PROBABILITY; }
{PROPERTY} { strncpy(yylval->word, yytext, CHAR_BUFFER_SIZE); return TOKEN_PROPERTY; }
{VARIABLETYPE} { return TOKEN_VARIABLETYPE; }
{DISCRETE} { return TOKEN_DISCRETE; }
{DEFAULTVALUE} { return TOKEN_DEFAULTVALUE; }
{TABLEVALUES} {return TOKEN_TABLEVALUES; }
{L_CURLY_BRACE} {return TOKEN_L_CURLY_BRACE; }
{R_CURLY_BRACE} {return TOKEN_R_CURLY_BRACE; }
{L_BRACKET} {return TOKEN_L_BRACKET; }
{R_BRACKET} {return TOKEN_R_BRACKET; }
{L_PARENS} { return TOKEN_L_PARENS; }
{R_PARENS} { return TOKEN_R_PARENS; }
{SEMICOLON} { return TOKEN_SEMICOLON; }
{FLOATING_POINT_LITERAL} { sscanf(yytext, "%f", &yylval->float_value); return TOKEN_FLOATING_POINT_LITERAL; }
{DECIMAL_LITERAL} { sscanf(yytext, "%d", &yylval->int_value); return TOKEN_DECIMAL_LITERAL; }
{WORD} { strncpy(yylval->word, yytext, CHAR_BUFFER_SIZE); return TOKEN_WORD; }
. { }

%%

static int yyerror(const char* msg) {
	fprintf(stderr, "Error: %s\n", msg);
	return 0;
}