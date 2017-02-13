#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

#include "expression.h"

static struct expression * allocate_expression()
{
	struct expression * expr = (struct expression *)malloc(sizeof(struct expression));
	assert(expr != NULL);

	expr->type = BLANK;
	expr->double_value = 0.0;
	expr->int_value = 0;
	expr->left = NULL;
	expr->right = NULL;

	return expr;
}

struct expression * create_expression(eType type, struct expression * left, struct expression * right)
{
	struct expression * expr = allocate_expression();

	expr->type = type;
	expr->left = left;
	expr->right = right;

	return expr;
}

void delete_expression(struct expression * expr){
	if(expr == NULL){
		return;
	}

	assert(expr != NULL);

	//print_expression(expr);

	delete_expression(expr->left);
	delete_expression(expr->right);

	free(expr);
}

void print_expression(struct expression * expr){
	printf("Expression: {\n");
	printf("Type: ");
	switch(expr->type){
		case COMPILATION_UNIT: printf("Compilation Unit"); break;
		case NETWORK_DECLARATION: printf("Network Declaration"); break;
		case NETWORK_CONTENT: printf("Network Content"); break;
		case PROPERTY_LIST: printf("Property List"); break;
		case PROPERTY: printf("Property"); break;
		case BLANK: printf("Blank"); break;
		case VARIABLE_OR_PROBABILITY_DECLARATION: printf("Variable or Probability Declaration"); break;
		case VARIABLE_OR_PROBABILITY: printf("Variable or Probability"); break;
		case VARIABLE_DECLARATION: printf("Variable Declaration"); break;
		case VARIABLE_CONTENT: printf("Variable Content"); break;
		case VARIABLE_DISCRETE: printf("Variable Discrete"); break;
		case VARIABLE_VALUES_LIST: printf("Variable Values List"); break;
		case PROBABILITY_DECLARATION: printf("Probability Declaration"); break;
		case PROBABILITY_VARIABLES_LIST: printf("Probability Variables List"); break;
		case PROBABILITY_VARIABLE_NAMES: printf("Probability Variable Names"); break;
		case PROBABILITY_CONTENT: printf("Probability Content"); break;
		case PROBABILITY_CONTENT_LIST: printf("Probability Content List"); break;
		case PROBABILITY_DEFAULT_ENTRY: printf("Probability Default Entry"); break;
		case PROBABILITY_ENTRY: printf("Probability Entry"); break;
		case PROBABILITY_VALUES_LIST: printf("Probability Values List"); break;
		case PROBABILITY_VALUES: printf("Probability Values"); break;
		case PROBABILITY_TABLE: printf("Probability Table"); break;
		case FLOATING_POINT_LIST_FLOAT: printf("Floating Point List (Float)"); break;
		case FLOATING_POINT_LIST_INT: printf("Floating Point List (Int)"); break;
	}
	printf("\n");

	switch(expr->type){
		case NETWORK_CONTENT:
		case PROPERTY:
		case VARIABLE_VALUES_LIST:
		case PROBABILITY_VARIABLE_NAMES:
		case PROBABILITY_VALUES:
			printf("Content: %s\n", expr->value);
			break;
	}

	switch(expr->type){
		case FLOATING_POINT_LIST_FLOAT:
			printf("Double value: %lf\n", expr->double_value);
			break;
	}

	switch(expr->type){
	case VARIABLE_DISCRETE:
	case FLOATING_POINT_LIST_INT:
		printf("Int value: %d\n", expr->int_value);
		break;
	}

	printf("}\n");
}
