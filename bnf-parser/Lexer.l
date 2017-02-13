%{

#include <stdio.h>

#include "expression.h"
#include "Parser.h"

%}

%option outfile="Lexer.c" header-file="Lexer.h"
%option warn nodefault
 
%option reentrant noyywrap never-interactive nounistd
%option bison-bridge


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
SEMICOLON ";"

WS [ \n\t\r]+
COMMA [,]
PIPE [|]
SINGLE_LINE_COMMENT [/]{2}[^\n\r]*(\n|\r\n)
BLOCK_COMMENT [/][*][^*]*\*([^/][^*]*\*)*[/]


WORD [a-zA-Z_-][a-zA-Z0-9_-]*

DECIMAL_LITERAL [1-9][0-9]*
EXPONENT [eE][+-]?[0-9]+
FLOATING_POINT_LITERAL ([0-9]+\.(EXPONENT)?)|(\.[0-9]+(EXPONENT)?)|([0-9]+(EXPONENT)?)

%%


{WS} { }
{COMMA} { }
{PIPE} { }
{SINGLE_LINE_COMMENT} { }
{BLOCK_COMMENT} { }
{NETWORK} { return TOKEN_NETWORK; }
{VARIABLE} { return TOKEN_VARIABLE; }
{PROBABILITY} {return TOKEN_PROBABILITY; }
{PROPERTY} { return TOKEN_PROPERTY; }
{VARIABLETYPE} { return TOKEN_VARIABLETYPE; }
{DISCRETE} { return TOKEN_DISCRETE; }
{DEFAULTVALUE} { return TOKEN_DEFAULTVALUE; }
{TABLEVALUES} {return TOKEN_TABLEVALUES; }
{L_CURLY_BRACE} {return TOKEN_L_CURLY_BRACE; }
{R_CURLY_BRACE} {return TOKEN_R_CURLY_BRACE; }
{DECIMAL_LITERAL} { sscanf(yytext, "%d", &yylval->int_value); return TOKEN_DECIMAL_LITERAL; }
{FLOATING_POINT_LITERAL} { sscanf(yytext, "%lf", &yylval->double_value); return TOKEN_FLOATING_POINT_LITERAL; }
{WORD} { strncpy(yylval->word, yytext, 20); return TOKEN_WORD; }
. { }

%%

static int yyerror(const char* msg) {
	fprintf(stderr, "Error:%s\n", msg);
	return 0;
}