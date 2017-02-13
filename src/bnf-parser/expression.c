#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "expression.h"

static struct expression * allocate_expression()
{
	struct expression * expr = (struct expression *)malloc(sizeof(struct expression));
	assert(expr != NULL);

	expr->type = BLANK;

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

struct expression * create_word_expression(eType type, char * word){
	struct expression * expr = allocate_expression();

	expr->type = type;
	strncpy(expr->value, word, 20);

	return expr;
}

void delete_expression(struct expression * expr){
	if(expr == NULL){
		return;
	}

	delete_expression(expr->left);
	delete_expression(expr->right);

	free(expr);
}
