/*
 * graph.h
 *
 *  Created on: Feb 5, 2017
 */

/**
 * Based off of http://www.cs.yale.edu/homes/aspnes/pinewiki/C(2f)Graphs.html
 */

#ifndef GRAPH_H_
#define GRAPH_H_

#ifndef __USE_GNU
#define __USE_GNU
#endif

#include "../constants.h"

#include <search.h>

/**
 * Struct holding the priori probabilities and the size of the probabilities
 */
struct belief {
	/**
	 * The priori probabilities
	 */
    float data[MAX_STATES];
	/**
	 * The size of the probabilities
	 */
    unsigned int size;
	/**
	 * The previous sum
	 */
	float previous;
	/**
	 * The current sum
	 */
	float current;
};

/**
 * Struct holding the joint probabilities on the edge
 */
struct joint_probability {
	/**
	 * The joint probability table
	 */
    float data[MAX_STATES][MAX_STATES];
	/**
	 * The first dimension of the table
	 */
    unsigned int dim_x;
	/**
	 * The second dimension of the table
	 */
    unsigned int dim_y;
};

/**
 * Struct holding the graph data
 */
struct graph {
	/**
	 * The number of nodes in the graph allocated
	 */
	unsigned int total_num_vertices;
	/**
	 * The number of edges in the graph allocated
	 */
	unsigned int total_num_edges;

	/**
	 * The number of nodes currently added to the graph
	 */
	unsigned int current_num_vertices;
	/**
	 * The number of edges currently added to the graph
	 */
	unsigned int current_num_edges;
	/**
	 * The maximum degree of any node in the graph
	 */
	unsigned int max_degree;

	/**
	 * Array of edges to the index of their source nodes
	 */
	unsigned int * edges_src_index;
	/**
	 * Array of edges to the index of the destination nodes
	 */
	unsigned int * edges_dest_index;
	/**
	 * Array of edges by their first dimension of the joint probability
	 */

	/**
	 * Array of joint probabilities indexed by edge
	 */
	struct joint_probability * edges_joint_probabilities;

	/**
	 * The array of current beliefs
	 */
	struct belief * edges_messages;

	/**
	 * Array of belief states indexed by node
	 */
	struct belief * node_states;

	/**
	 * Array of indices in src_nodes_to_edges_edge_list indexed by their source node
	 */
	unsigned int * src_nodes_to_edges_node_list;
	/**
	 * Array of edges indexed by their source node
	 */
	unsigned int * src_nodes_to_edges_edge_list;

	/**
	 * Array of indices in dest_nodes_to_edges_edge_list indexed by their destination node
	 */
	unsigned int * dest_nodes_to_edges_node_list;
	/**
	 * Array of edges index by their destination node
	 */
	unsigned int * dest_nodes_to_edges_edge_list;

	/**
	 * Levels in the tree to the nodes there
	 */
	unsigned int * levels_to_nodes;
	/**
	 * The size of the level array
	 */
	unsigned int num_levels;

	/**
	 * Array of nodes left in the work queue
	 */
	unsigned int *work_queue_nodes;

	/**
	 * Array of edges left in the work queue
	 */
	unsigned int *work_queue_edges;

	/**
	 * Array for scratch space
	 */
	unsigned int *work_queue_scratch;

	/**
	 * Number of items in work queue
	 */
	unsigned int num_work_items_nodes;
	unsigned int num_work_items_edges;


	/**
	 * The diameter of the graph
	 */
    int diameter;

	/**
	 * Bit vector if a node has been visited
	 */
	char * visited;

	/**
	 * The array of node names
	 */
	char * node_names;

	/**
	 * The array of belief names within the nodes
	 */
	char * variable_names;

	/**
	 * Bit vector of nodes if they are observed nodes
	 */
    char * observed_nodes;

	/**
	 * The name of the network
	 */
	char graph_name[CHAR_BUFFER_SIZE];

	/**
	 * Flag if the node name to node index hash has been created
	 */
    char node_hash_table_created;
	/**
	 * Hash table of node name to node index
	 */
	struct hsearch_data *node_hash_table;

	/**
	 * Hash table of src node index to edge
	 */
    struct hsearch_data *src_node_to_edge_table;
	/**
	 * Hash table of dest node index to edge
	 */
    struct hsearch_data *dest_node_to_edge_table;
	/**
	 * Flag if hash tables have been created
	 */
    char edge_tables_created;
};
typedef struct graph* Graph_t;

/**
 * Entry within the hash table to store the indices and count
 */
struct htable_entry {
	/**
	 * The array of indices
	 */
    unsigned int indices[MAX_DEGREE];
	/**
	 * The acutal size of the array
	 */
    unsigned int count;
};

Graph_t create_graph(unsigned int, unsigned int);

void graph_add_node(Graph_t, unsigned int, const char *);
void graph_add_and_set_node_state(Graph_t, unsigned int, const char *, struct belief *);
void graph_set_node_state(Graph_t, unsigned int, unsigned int, struct belief *);

void graph_add_edge(Graph_t, unsigned int, unsigned int, unsigned int, unsigned int, struct joint_probability *);

void set_up_src_nodes_to_edges(Graph_t);
void set_up_dest_nodes_to_edges(Graph_t);
void init_levels_to_nodes(Graph_t);
void calculate_diameter(Graph_t);
void prep_as_page_rank(Graph_t);

void initialize_node(Graph_t, unsigned int, unsigned int);
void node_set_state(Graph_t, unsigned int, unsigned int, struct belief *);

void init_edge(Graph_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, struct joint_probability *);
void send_message(struct belief *, unsigned int, struct joint_probability *, struct belief *);

void fill_in_node_hash_table(Graph_t);
unsigned int find_node_by_name(char *, Graph_t);

void graph_destroy(Graph_t);

void propagate_using_levels_start(Graph_t);
void propagate_using_levels(Graph_t, unsigned int);

void reset_visited(Graph_t);

void init_previous_edge(Graph_t);
void loopy_propagate_one_iteration(Graph_t);
void loopy_propagate_edge_one_iteration(Graph_t);
void page_rank_one_iteration(Graph_t);
void page_rank_edge_one_iteration(Graph_t);
void viterbi_one_iteration(Graph_t);
void viterbi_edge_one_iteration(Graph_t);

unsigned int loopy_propagate_until(Graph_t, float, unsigned int);
unsigned int loopy_propagate_until_edge(Graph_t, float, unsigned int);
unsigned int loopy_propagate_until_acc(Graph_t, float, unsigned int);
unsigned int loopy_propagate_until_edge_acc(Graph_t, float, unsigned int);

unsigned int page_rank_until(Graph_t, float, unsigned int);
unsigned int page_rank_until_edge(Graph_t, float, unsigned int);
unsigned int page_rank_until_acc(Graph_t, float, unsigned int);
unsigned int page_rank_until_edge_acc(Graph_t, float, unsigned int);

unsigned int viterbi_until(Graph_t, float, unsigned int);
unsigned int viterbi_until_edge(Graph_t, float, unsigned int);
unsigned int viterbi_until_acc(Graph_t, float, unsigned int);
unsigned int viterbi_until_edge_acc(Graph_t, float, unsigned int);

void marginalize(Graph_t);

void print_node(Graph_t, unsigned int);
void print_edge(Graph_t, unsigned int);
void print_nodes(Graph_t);
void print_edges(Graph_t);
void print_src_nodes_to_edges(Graph_t);
void print_dest_nodes_to_edges(Graph_t);
void print_levels_to_nodes(Graph_t);

void init_work_queue_nodes(Graph_t);
void init_work_queue_edges(Graph_t);

void update_work_queue_nodes(Graph_t, float);
void update_work_queue_edges(Graph_t, float);


#endif /* GRAPH_H_ */
