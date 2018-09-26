#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include "../graph/graph.h"

// from http://6.869.csail.mit.edu/fa13/lectures/slideNotesCh7rev.pdf
// Suite of tests using the graph in the PDF

static const int NUM_NODES = 4;
static const int NUM_EDGES = 5;
static const int NUM_VARIABLES = 2;

void assert_value(float diff){
    assert(diff > -0.0001);
    assert(diff < 0.0001);
}

void add_nodes(Graph_t graph){
    unsigned int node_index;
	struct belief y2;

	y2.size = NUM_VARIABLES;
	y2.data[0] = 1.0;
	y2.data[1] = 0.0;

	graph_add_node(graph, NUM_VARIABLES, "x1");
	graph_add_node(graph, NUM_VARIABLES, "x2");
	graph_add_node(graph, NUM_VARIABLES, "x3");
	graph_add_and_set_node_state(graph, NUM_VARIABLES, "y2", &y2);

    // validate nodes inserted correctly
    assert(graph->current_num_vertices == 4);

    for(node_index = 0; node_index < 3; ++node_index) {
        assert(graph->node_states[node_index].size == 2);
        assert_value(graph->node_states[node_index].data[0] - 1.0f);
        assert_value(graph->node_states[node_index].data[1] - 1.0f);
    }
    node_index = 3;
    assert(graph->node_states[node_index].size == 2);
    assert_value(graph->node_states[node_index].data[0] - 1.0f);
    assert_value(graph->node_states[node_index].data[1] - 0.0f);
}

void add_edges(Graph_t graph){
	struct joint_probability phi_1_2, phi_2_3, phi_2_4;

	phi_1_2.dim_x = 2;
	phi_1_2.dim_y = 2;
	phi_1_2.data[0][0] = 1.0;
	phi_1_2.data[0][1] = 0.9;
	phi_1_2.data[1][0] = 0.9;
	phi_1_2.data[1][1] = 1.0;

	phi_2_3.dim_x = 2;
	phi_2_3.dim_y = 2;
	phi_2_3.data[0][0] = 0.1;
	phi_2_3.data[0][1] = 1.0;
	phi_2_3.data[1][0] = 1.0;
	phi_2_3.data[1][1] = 0.1;

	phi_2_4.dim_x = 2;
	phi_2_4.dim_y = 2;
	phi_2_4.data[0][0] = 1.0;
	phi_2_4.data[0][1] = 0.1;
	phi_2_4.data[1][0] = 0.1;
	phi_2_4.data[1][1] = 1.0;

	graph_add_edge(graph, 0, 1, 2, 2, &phi_1_2);
	graph_add_edge(graph, 1, 0, 2, 2, &phi_1_2);
	graph_add_edge(graph, 1, 2, 2, 2, &phi_2_3);
	graph_add_edge(graph, 2, 1, 2, 2, &phi_2_3);
	graph_add_edge(graph, 3, 1, 2, 2, &phi_2_4);


}

void validate_nodes(Graph_t graph){
	unsigned int node_index;
	float value;

	node_index = 0;

	//x1
	value = graph->node_states[node_index].data[0] - 0.521531100478469f;
	assert_value(value);
	value = graph->node_states[node_index].data[1] - 0.47846889952153115f;
	assert_value(value);

	node_index++;

	//x2
	value = graph->node_states[node_index].data[0] - 0.9090909090909091f;
	assert_value(value);
	value = graph->node_states[node_index].data[1] - 0.09090909090909091f;
	assert_value(value);

	node_index++;

	//x3
	value = graph->node_states[node_index].data[0] - 0.1652892561983471f;
	assert_value(value);
	value = graph->node_states[node_index].data[1] - 0.8347107438016529f;
	assert_value(value);

	node_index++;

	//y2
	value = graph->node_states[node_index].data[0] - 1.0f;
	assert_value(value);
	value = graph->node_states[node_index].data[1] - 0.0f;
	assert_value(value);
}

void forward_backward_belief_propagation() {
	Graph_t graph;
	unsigned int i;

	graph = create_graph(NUM_NODES, NUM_EDGES);

	add_nodes(graph);
	add_edges(graph);

	set_up_src_nodes_to_edges(graph);
	set_up_dest_nodes_to_edges(graph);

	init_levels_to_nodes(graph);

	print_levels_to_nodes(graph);

	print_nodes(graph);
	print_edges(graph);
	print_src_nodes_to_edges(graph);
	print_dest_nodes_to_edges(graph);

	propagate_using_levels_start(graph);
	for(i = 1; i < graph->num_levels - 1; ++i){
		propagate_using_levels(graph, i);
	}
	reset_visited(graph);
    //printf("Resetting...\n");
   	//print_nodes(graph);
	for(i = graph->num_levels - 1; i > 0; --i){
		propagate_using_levels(graph, i);
	}
    print_edges(graph);
 	marginalize(graph);

	print_nodes(graph);

	validate_nodes(graph);

	graph_destroy(graph);

}

void loopy_belief_propagation() {
	Graph_t graph;

	graph = create_graph(NUM_NODES, NUM_EDGES);

	add_nodes(graph);
	add_edges(graph);

	set_up_src_nodes_to_edges(graph);
	set_up_dest_nodes_to_edges(graph);

	print_nodes(graph);
	print_edges(graph);

	init_previous_edge(graph);

    loopy_propagate_until(graph, 1E-9, 10000);


    print_nodes(graph);


	graph_destroy(graph);
}

int main() {
	forward_backward_belief_propagation();

	loopy_belief_propagation();

	return 0;
}
