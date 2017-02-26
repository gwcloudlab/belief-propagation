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

static void reverse_buffer(char *buffer, int num_nodes){
	int i;
	char temp[CHAR_BUFFER_SIZE];

	for(i = 0; i < num_nodes/2; ++i){
		strncpy(temp, &buffer[i * CHAR_BUFFER_SIZE], CHAR_BUFFER_SIZE);
		strncpy(&buffer[i * CHAR_BUFFER_SIZE], &buffer[(num_nodes - i - 1) * CHAR_BUFFER_SIZE], CHAR_BUFFER_SIZE);
		strncpy(&buffer[(num_nodes - i - 1) * CHAR_BUFFER_SIZE], temp, CHAR_BUFFER_SIZE);
	}
}

static void reverse_probability_table(double * probability_table, int num_probabilities){
	int i;
	double temp;

	for(i = 0; i < num_probabilities/2; ++i){
		temp = probability_table[i];
		probability_table[i] = probability_table[num_probabilities - i - 1];
		probability_table[num_probabilities - i - 1] = temp;
	}
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

	add_variable_discrete(expr->left, graph, state_index);
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

static void count_number_of_node_names(struct expression *expr, int * count){
	if(expr == NULL){
		return;
	}
	if(expr->type == PROBABILITY_VARIABLE_NAMES){
		*count += 1;
	}

	count_number_of_node_names(expr->left, count);
	count_number_of_node_names(expr->right, count);
}

static void fill_in_node_names(struct expression *expr, char *buffer, int *curr_index){
	if(expr == NULL){
		return;
	}

	if(expr->type == PROBABILITY_VARIABLE_NAMES){
		strncpy(&buffer[*curr_index * CHAR_BUFFER_SIZE], expr->value, CHAR_BUFFER_SIZE);
		*curr_index += 1;
	}

	fill_in_node_names(expr->left, buffer, curr_index);
	fill_in_node_names(expr->right, buffer, curr_index);
}

static void fill_in_probability_table_value(struct expression *expr, double *probability_buffer,
											int num_probabilities, int *current_index){
	if(expr == NULL){
		return;
	}

	assert(*current_index < num_probabilities);

	if(expr->type == FLOATING_POINT_LIST){
        if(probability_buffer[*current_index] < 0) {
            probability_buffer[*current_index] = expr->double_value;
        }
		*current_index += 1;
	}

	fill_in_probability_table_value(expr->left, probability_buffer, num_probabilities, current_index);
	fill_in_probability_table_value(expr->right, probability_buffer, num_probabilities, current_index);
}

static void fill_in_probability_table_table(struct expression * expr, double * probability_buffer, int num_elements, int * curr_index){
	if(expr == NULL){
		return;
	}
	assert(*curr_index < num_elements);
	if(expr->type == FLOATING_POINT_LIST){
		probability_buffer[*curr_index] = expr->double_value;
		*curr_index += 1;
	}

	fill_in_probability_table_table(expr->left, probability_buffer, num_elements, curr_index);
	fill_in_probability_table_table(expr->right, probability_buffer, num_elements, curr_index);
}

static void fill_in_probability_table_entry(struct expression * expr, double * probability_buffer, int num_probabilities,
											int pos, int jump, int num_states, int * current_index){
    int index;
	if(expr == NULL){
        return;
    }

    assert(*current_index < num_states);

    if(expr->type == FLOATING_POINT_LIST){
		index = (num_states - *current_index - 1) * jump + pos;
		assert(index < num_probabilities);
		probability_buffer[index] = expr->double_value;
        *current_index += 1;
    }

    fill_in_probability_table_entry(expr->left, probability_buffer, num_probabilities, pos, jump, num_states, current_index);
    fill_in_probability_table_entry(expr->right, probability_buffer, num_probabilities, pos, jump, num_states, current_index);
}

static void fill_in_probability_buffer_default(struct expression * expr, double * probability_buffer, int num_probabilities){
	int current_index;
	if(expr == NULL){
		return;
	}

	if(expr->type == PROBABILITY_DEFAULT_ENTRY){
        current_index = 0;
        while(current_index < num_probabilities) {
            fill_in_probability_table_value(expr, probability_buffer, num_probabilities, &current_index);
        }
	}

	fill_in_probability_buffer_default(expr->left, probability_buffer, num_probabilities);
	fill_in_probability_buffer_default(expr->right, probability_buffer, num_probabilities);
}

static void fill_in_probability_buffer_table(struct expression * expr, double * probability_buffer, int num_entries){
	int current_index;
	if(expr == NULL){
		return;
	}

	current_index = 0;

	if(expr->type == PROBABILITY_TABLE){
		fill_in_probability_table_table(expr, probability_buffer, num_entries, &current_index);
	}

	fill_in_probability_buffer_table(expr->left, probability_buffer, num_entries);
	fill_in_probability_buffer_table(expr->right, probability_buffer, num_entries);
}

static void count_num_state_names(struct expression * expr, int * count){
	if(expr == NULL){
		return;
	}

	if(expr->type == PROBABILITY_VALUES_LIST){
		count_num_state_names(expr->left, count);
		return;
	}

	if(expr->type == PROBABILITY_VALUES) {
		*count += 1;
	}
	count_num_state_names(expr->left, count);
	count_num_state_names(expr->right, count);
}

static void fill_in_state_names(struct expression * expr, int count, int * current_index, char * buffer){
	if(expr == NULL){
		return;
	}

	if(expr->type == PROBABILITY_VALUES_LIST){
		fill_in_state_names(expr->left, count, current_index, buffer);
		fill_in_state_names(expr->right, count, current_index, buffer);
		return;
	}

	if(expr->type == PROBABILITY_VALUES){
		assert(*current_index < count);
		strncpy(&(buffer[*current_index * CHAR_BUFFER_SIZE]), expr->value, CHAR_BUFFER_SIZE);
		*current_index += 1;
	}

	fill_in_state_names(expr->left, count, current_index, buffer);
	fill_in_state_names(expr->right, count, current_index, buffer);
}

static Node_t find_node_by_name(char * name, Graph_t graph){
	Node_t node;
	char * node_name;
	int i;

	node = NULL;

	for(i = 0; i < graph->current_num_vertices; ++i){
		node_name = &(graph->node_names[i * CHAR_BUFFER_SIZE]);
		if(strcmp(name, node_name) == 0){
			node = &graph->nodes[i];
			break;
		}
	}

	assert(node != NULL);

	return node;
}

static int calculate_entry_offset(char * state_names, int num_states, char * variable_names, int num_variables,
								  Graph_t graph){
    int i, j, k, pos, step, index, var_name_index;
    char * state;
    char * node_state_name;
    Node_t current_node;

	int * indices = (int *)malloc(sizeof(int) * num_states);

	for(i = 0; i < num_states; ++i){
		state = &(state_names[i * CHAR_BUFFER_SIZE]);
		current_node = find_node_by_name(&(variable_names[(i+1)*CHAR_BUFFER_SIZE]), graph);
		for(j = 0; j < current_node->num_variables; ++j){
			var_name_index =  current_node->index * MAX_STATES * CHAR_BUFFER_SIZE + j * CHAR_BUFFER_SIZE;
			node_state_name = &graph->variable_names[var_name_index];
			if(strcmp(state, node_state_name) == 0){
				indices[i] = j;
				break;
			}
		}
	}
/*
	printf("Indices for Sink: %s\n", variable_names);
	for(i = 0; i < num_states; ++i) {
		printf("State '%s': %d\n", &(state_names[i * CHAR_BUFFER_SIZE]), indices[i]);
	}
*/
    pos = 0;
    step = 1;
    for(k = num_states; k > 0; --k){
		pos += indices[k - 1] * step;
		current_node = find_node_by_name(&(variable_names[k * CHAR_BUFFER_SIZE]), graph);
		step *= current_node->num_variables;
	}

	free(indices);

//	printf("Posiition: %d\n", pos);

    return pos;
}

static void fill_in_probability_buffer_entry(struct expression * expr, double * probability_buffer,
											 int num_probabilities,
											 char * variables, int num_variables,
											 int first_num_states,
                                             Graph_t graph){
	int count, index, pos, jump;
	char * state_names;

	if(expr == NULL){
		return;
	}

	if(expr->type == PROBABILITY_ENTRY){
		jump = num_probabilities / first_num_states;

		count = 0;
		count_num_state_names(expr, &count);

		state_names = (char *)malloc(sizeof(char) * count * CHAR_BUFFER_SIZE);

		index = 0;
		fill_in_state_names(expr, count, &index, state_names);
        reverse_buffer(state_names, count);

        pos = calculate_entry_offset(state_names, count, variables, num_variables, graph);

        index = 0;

        fill_in_probability_table_entry(expr, probability_buffer, num_probabilities, pos, jump, first_num_states, &index);

		free(state_names);
	}

	fill_in_probability_buffer_entry(expr->left, probability_buffer, num_probabilities, variables, num_variables, first_num_states, graph);
	fill_in_probability_buffer_entry(expr->right, probability_buffer, num_probabilities, variables, num_variables, first_num_states, graph);
}

static int calculate_num_probabilities(char *node_name_buffer, int num_nodes, Graph_t graph){
	int i, j, num_probabilities, found_index;
	char * curr_name;
	char * curr_node_name;
	Node_t curr_node;

	num_probabilities = 1;

	for(i = 0; i < num_nodes; ++i){
		curr_name = &(node_name_buffer[i * CHAR_BUFFER_SIZE]);
		curr_node_name = NULL;
		found_index = -1;
		for(j = 0; j < graph->current_num_vertices; ++j){
			curr_node_name = &(graph->node_names[j * CHAR_BUFFER_SIZE]);
			if(strcmp(curr_name, curr_node_name) == 0){
				found_index = j;
				break;
			}
		}
		assert(found_index < graph->current_num_vertices);
		curr_node = &(graph->nodes[found_index]);

		num_probabilities *= curr_node->num_variables;
	}

	return num_probabilities;
}

static void update_node_in_graph(struct expression * expr, Graph_t graph){
	char * buffer;
	char * node_names;
	double * probability_buffer;
	int index, num_node_names, num_probabilities;
    int node_index;

	if(expr == NULL){
		return;
	}

	num_node_names = 0;
	count_number_of_node_names(expr, &num_node_names);
	if(num_node_names != 1){
		return;
	}

	index = 0;
	buffer = (char *)malloc(sizeof(char) * CHAR_BUFFER_SIZE * num_node_names);
	assert(buffer);

	fill_in_node_names(expr, buffer, &index);

	num_probabilities = calculate_num_probabilities(buffer, num_node_names, graph);

	probability_buffer = (double *)malloc(sizeof(double) * num_probabilities);
	assert(probability_buffer);
    for(index = 0; index < num_probabilities; ++index){
        probability_buffer[index] = -1.0;
    }

	fill_in_probability_buffer_table(expr, probability_buffer, num_probabilities);
    fill_in_probability_buffer_default(expr, probability_buffer, num_probabilities);

    for(index = 0; index < num_probabilities; ++index){
        if(probability_buffer[index] < 0.0){
            probability_buffer[index] = 0.0;
        }
    }

	reverse_probability_table(probability_buffer, num_probabilities);

    node_index = -1;
	node_names = graph->node_names;
    for(index = 0; index < graph->current_num_vertices; ++index){
        if(strcmp(buffer, &node_names[index * CHAR_BUFFER_SIZE]) == 0){
            node_index = index;
            break;
        }
    }

    assert(node_index >= 0);

    graph_set_node_state(graph, node_index, num_probabilities, probability_buffer);

	free(buffer);
	free(probability_buffer);
}

static void insert_edges_into_graph(char * variable_buffer, int num_node_names, double * probability_buffer, int num_probabilities, Graph_t graph){
	Node_t dest;
	Node_t src;
	int i, j, k, offset, slice, index, delta, next, diff;
	double ** sub_graph;
	double ** transpose;

	assert(num_node_names > 1);

	dest = find_node_by_name(variable_buffer, graph);
	slice = num_probabilities / dest->num_variables;

    /*
	printf("Values for sinK: %s\n", variable_buffer);
	for(i = 0; i < num_probabilities; ++i){
		printf("%.6lf\t", probability_buffer[i]);
		if(i % dest->num_variables == dest->num_variables - 1){
			printf("\n");
		}
	}
	printf("\n");
*/
	offset = 1;
	for(i = num_node_names - 1; i > 0; --i){
		src = find_node_by_name(&(variable_buffer[i * CHAR_BUFFER_SIZE]), graph);
        //printf("LOOKING AT src: %s\n", &(variable_buffer[i * CHAR_BUFFER_SIZE]));

        delta = src->num_variables;

		sub_graph = (double **)calloc(sizeof(double*), (size_t)src->num_variables);
		transpose = (double **)calloc(sizeof(double*), (size_t)dest->num_variables);
		for(j = 0; j < src->num_variables; ++j){
			sub_graph[j] = (double *)calloc(sizeof(double), (size_t)dest->num_variables);
		}
		for(j = 0; j < dest->num_variables; ++j){
			transpose[j] = (double *)calloc(sizeof(double), (size_t)src->num_variables);
		}

		for(k = 0; k < dest->num_variables; ++k){
			for(j = 0; j < src->num_variables; ++j){
				diff = 0;
				index = j * offset + diff;
                while(index <= slice) {
					index = j * offset + diff;
					next = (j + 1) * offset + diff;
                    //printf("Current Index: %d; Next: %d; Delta: %d; Diff: %d\n", index, next, delta, diff);
                    while (index < next) {
                        sub_graph[j][k] += probability_buffer[index + k * slice];
                        index++;
                    }
					index += delta * offset;
                    diff += delta * offset;
                }
			}
		}

		for(j = 0; j < src->num_variables; ++j){
			for(k = 0; k < dest->num_variables; ++k){
				transpose[k][j] = sub_graph[j][k];
			}
		}

		graph_add_edge(graph, src->index, dest->index, src->num_variables, dest->num_variables, sub_graph);
		if(graph->observed_nodes[src->index] != 1 ){
			graph_add_edge(graph, dest->index, src->index, dest->num_variables, src->num_variables, transpose);
		}

		for(j = 0; j < src->num_variables; ++j){
			free(sub_graph[j]);
		}
		free(sub_graph);

		for(j = 0; j < dest->num_variables; ++j){
			free(transpose[j]);
		}
		free(transpose);

		offset *= src->num_variables;
	}
}

static void add_edge_to_graph(struct expression * expr, Graph_t graph){
    char * buffer;
    double * probability_buffer;
    int index, num_node_names, num_probabilities, first_num_states;

    if(expr == NULL){
        return;
    }

    num_node_names = 0;
	count_number_of_node_names(expr, &num_node_names);
    if(num_node_names <= 1){
        return;
    }

    index = 0;
    buffer = (char *)malloc(sizeof(char) * CHAR_BUFFER_SIZE * num_node_names);
    assert(buffer);

    fill_in_node_names(expr, buffer, &index);
	reverse_buffer(buffer, num_node_names);

    num_probabilities = calculate_num_probabilities(buffer, num_node_names, graph);
    first_num_states = calculate_num_probabilities(buffer, 1, graph);

    probability_buffer = (double *)malloc(sizeof(double) * num_probabilities);
    assert(probability_buffer);
    for(index = 0; index < num_probabilities; ++index){
        probability_buffer[index] = -1.0;
    }

    fill_in_probability_buffer_table(expr, probability_buffer, num_probabilities);
    fill_in_probability_buffer_default(expr, probability_buffer, num_probabilities);

	reverse_probability_table(probability_buffer, num_probabilities);

    fill_in_probability_buffer_entry(expr, probability_buffer, num_probabilities, buffer, num_node_names, first_num_states, graph);

    for(index = 0; index < num_probabilities; ++index){
        if(probability_buffer[index] < 0.0){
            probability_buffer[index] = 0.0;
        }
    }



	insert_edges_into_graph(buffer, num_node_names, probability_buffer, num_probabilities, graph);

    free(buffer);
    free(probability_buffer);
}

static void update_nodes_in_graph(struct expression * expr, Graph_t graph){
	if(expr == NULL){
		return;
	}

	if(expr->type == NETWORK_DECLARATION){
		return;
	}

	if(expr->type == VARIABLE_DECLARATION){
		return;
	}

	if(expr->type == PROBABILITY_DECLARATION){
		update_node_in_graph(expr, graph);
	}
    update_nodes_in_graph(expr->left, graph);
    update_nodes_in_graph(expr->right, graph);
}

static void add_edges_to_graph(struct expression * expr, Graph_t graph){
	if(expr == NULL){
		return;
	}

    if(expr->type == NETWORK_DECLARATION){
        return;
    }

    if(expr->type == VARIABLE_DECLARATION){
        return;
    }

    if(expr->type == PROBABILITY_DECLARATION){
        add_edge_to_graph(expr, graph);
    }
    add_edges_to_graph(expr->left, graph);
    add_edges_to_graph(expr->right, graph);
}

static void reverse_node_names(Graph_t graph){
	int i, j, index_1, index_2;
	char temp[CHAR_BUFFER_SIZE];
	Node_t curr_node;

	for(i = 0; i < graph->current_num_vertices; ++i){
		curr_node = &graph->nodes[i];
		for(j = 0; j < curr_node->num_variables/2; ++j){
			index_1 = i * MAX_STATES * CHAR_BUFFER_SIZE + j * CHAR_BUFFER_SIZE;
			index_2 = i * MAX_STATES * CHAR_BUFFER_SIZE + (curr_node->num_variables / 2 - j) * CHAR_BUFFER_SIZE;
			strncpy(temp, &(graph->variable_names[index_1]), CHAR_BUFFER_SIZE);
			strncpy(&(graph->variable_names[index_1]), &(graph->variable_names[index_2]), CHAR_BUFFER_SIZE);
			strncpy(&(graph->variable_names[index_2]), temp, CHAR_BUFFER_SIZE);
		}
	}
}

Graph_t build_graph(struct expression * root){
	Graph_t graph;

	int num_nodes = count_nodes(root);
	//int num_edges = count_edges(root);

	graph = create_graph(num_nodes, num_nodes * num_nodes);
	add_nodes_to_graph(root, graph);
	reverse_node_names(graph);

    update_nodes_in_graph(root, graph);
    add_edges_to_graph(root, graph);


	return graph;
}
