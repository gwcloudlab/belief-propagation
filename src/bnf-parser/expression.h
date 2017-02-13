/*
 * expression.h
 *
 *  Created on: Feb 6, 2017
 *      Author: mjt5v
 */

#ifndef EXPRESSION_H_
#define EXPRESSION_H_
/*
 from http://www.cs.washington.edu/dm/vfml/appendixes/bif.htm
 https://en.wikipedia.org/wiki/GNU_bison
*/

#include "../constants.h"

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
	VARIABLE_VALUES_LIST
} eType;

struct expression {
	eType type;

	char value[CHAR_BUFFER_SIZE];
	double double_value;
	struct expression *left;
	struct expression *right;
};

struct expression * create_expression(eType, struct expression * left, struct expression * right);
struct expression * create_word_expression(eType, char *);
void delete_expression(struct expression *);


#endif /* EXPRESSION_H_ */
