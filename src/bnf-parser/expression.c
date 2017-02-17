#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

#include "expression.h"
#include "../graph/graph.h"

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

	print_expression(expr);

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
		case FLOATING_POINT_LIST: printf("Floating Point List"); break;
	}
	printf("\n");

	switch(expr->type){
		case NETWORK_DECLARATION:
		case PROPERTY:
		case VARIABLE_DECLARATION:
		case VARIABLE_VALUES_LIST:
		case PROBABILITY_VARIABLE_NAMES:
		case PROBABILITY_VALUES:
			printf("Content: %s\n", expr->value);
			break;
	}

	switch(expr->type){
		case FLOATING_POINT_LIST:
			printf("Double value: %lf\n", expr->double_value);
			break;
	}

	switch(expr->type){
	case VARIABLE_DISCRETE:
		printf("Int value: %d\n", expr->int_value);
		break;
	}

	printf("}\n");
}

static int count_nodes(struct expression * expr){
	int count;

	count = 0;

	if(expr == NULL){
		return count;
	}

	if(expr->type == VARIABLE_CONTENT){
		count = 1;
	}

	count += count_nodes(expr->left);
	count += count_nodes(expr->right);

	return count;
}

static int count_edges(struct expression * expr){
	int count;

	if(expr == NULL){
		return count;
	}

	if(expr->type == PROBABILITY_VARIABLE_NAMES) {
		count = 1;
	}
	else if(expr->type == PROBABILITY_VARIABLES_LIST) {
		count = -1;
	}
	else {
		count = 0;
	}

	count += count_edges(expr->left);
	count += count_edges(expr->right);

	return count;
}

static void add_variable_discrete(struct expression * expr, Graph_t graph, int * state_index){
	char * node_name;
	int num_vertices, char_index;

	if(expr == NULL){
		return;
	}

	num_vertices = graph->current_num_vertices;

	char_index = num_vertices * MAX_STATES * CHAR_BUFFER_SIZE + *state_index * CHAR_BUFFER_SIZE;
	node_name = &graph->variable_names[char_index];
	strncpy(node_name, expr->value, CHAR_BUFFER_SIZE);

	//printf("Adding value: %s\n", expr->value);

	*state_index += 1;

	add_variable_discrete(expr->right, graph, state_index);

}

static void add_property_or_variable_discrete(struct expression * expr, Graph_t graph, int * state_index)
{
	int num_states;

	if(expr == NULL){
		return;
	}

	if(expr->type == VARIABLE_OR_PROBABILITY){
		add_property_or_variable_discrete(expr->left, graph, state_index);
		add_property_or_variable_discrete(expr->right, graph, state_index);
	}
	if(expr->type == VARIABLE_DISCRETE){
		num_states = expr->int_value;
		assert(num_states <= MAX_STATES);
		add_variable_discrete(expr->left, graph, state_index);
		assert(*state_index == num_states);
	}
}

static void add_variable_content_to_graph(struct expression * expr, Graph_t graph, int * state_index){
	if(expr == NULL){
		return;
	}

	add_property_or_variable_discrete(expr->left, graph, state_index);
}


static void add_node_to_graph(struct expression * expr, Graph_t graph){
	char variable_name[CHAR_BUFFER_SIZE];
	int state_index;

	state_index = 0;

	strncpy(variable_name, expr->value, CHAR_BUFFER_SIZE);

	add_variable_content_to_graph(expr->left, graph, &state_index);

	graph_add_node(graph, state_index, variable_name);
}

static void add_nodes_to_graph(struct expression * expr, Graph_t graph){
	if(expr == NULL){
		return;
	}

	// add name if possible
	if(expr->type == NETWORK_DECLARATION){
		strncpy(graph->graph_name, expr->value, CHAR_BUFFER_SIZE);
		return;
	}

	// add nodes
	if(expr->type == VARIABLE_DECLARATION){
		add_node_to_graph(expr, graph);
		return;
	}
	if(expr->type == PROBABILITY_DECLARATION){
		return;
	}
	add_nodes_to_graph(expr->left, graph);
	add_nodes_to_graph(expr->right, graph);
}

static void add_edges_to_graph(struct expression * expr, Graph_t graph){
	if(expr == NULL){
		return;
	}


}

Graph_t build_graph(struct expression * root){
	Graph_t graph;

	int num_nodes = count_nodes(root);
	int num_edges = count_edges(root);

	graph = create_graph(num_nodes, num_edges);
	add_nodes_to_graph(root, graph);

	return graph;
}
