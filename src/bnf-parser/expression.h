/*
 * expression.h
 *
 *  Created on: Feb 6, 2017
 */

#ifndef EXPRESSION_H_
#define EXPRESSION_H_
/*
 from http://www.cs.washington.edu/dm/vfml/appendixes/bif.htm
 https://en.wikipedia.org/wiki/GNU_bison
*/

#include "../constants.h"
#include "../graph/graph.h"

typedef enum expressionType
{
	COMPILATION_UNIT,
	NETWORK_DECLARATION,
	NETWORK_CONTENT,
	PROPERTY_LIST,
	PROPERTY,
	BLANK,
	VARIABLE_OR_PROBABILITY_DECLARATION,
	VARIABLE_OR_PROBABILITY,
	VARIABLE_DECLARATION,
	VARIABLE_CONTENT,
	VARIABLE_DISCRETE,
	VARIABLE_VALUES_LIST,
	PROBABILITY_DECLARATION,
	PROBABILITY_VARIABLES_LIST,
	PROBABILITY_VARIABLE_NAMES,
	PROBABILITY_CONTENT,
	PROBABILITY_CONTENT_LIST,
	PROBABILITY_DEFAULT_ENTRY,
	PROBABILITY_ENTRY,
	PROBABILITY_VALUES_LIST,
	PROBABILITY_VALUES,
	PROBABILITY_TABLE,
	FLOATING_POINT_LIST
} eType;

struct expression {
	eType type;

	char value[CHAR_BUFFER_SIZE];
	float float_value;
	int int_value;
	struct expression *left;
	struct expression *right;
};

struct expression * create_expression(eType, struct expression * left, struct expression * right);
void delete_expression(struct expression *);
void print_expression(struct expression *);

Graph_t build_graph(struct expression *);

#endif /* EXPRESSION_H_ */
