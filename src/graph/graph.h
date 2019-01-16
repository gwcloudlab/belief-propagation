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
#include <sys/queue.h>

/**
 * Struct holding the priori probabilities and the size of the probabilities
 */
struct belief {
	/**
	 * The priori probabilities
	 */
    float data[MAX_STATES];
};

/**
 * Struct holding the joint probabilities on the edge
 */

struct joint_probability {
	/**
	 * The joint probability table
	 */
    float data[MAX_STATES][MAX_STATES];
};

/**
 * Struct holding the graph data
 */
struct graph {
	/**
	 * The number of nodes in the graph allocated
	 */
	unsigned long total_num_vertices;
	/**
	 * The number of edges in the graph allocated
	 */
	unsigned long total_num_edges;

	/**
	 * The number of nodes currently added to the graph
	 */
	unsigned long current_num_vertices;
	/**
	 * The number of edges currently added to the graph
	 */
	unsigned long current_num_edges;
	/**
	 * The maximum degree of any node in the graph
	 */
	int max_degree;

	/**
	 * Array of edges to the index of their source nodes
	 */
	int * edges_src_index;
	/**
	 * Array of edges to the index of the destination nodes
	 */
	int * edges_dest_index;
	/**
	 * Array of edges by their first dimension of the joint probability
	 */

	/**
	 * Array of joint probabilities indexed by edge
	 */
	struct joint_probability edge_joint_probability;
	int edge_joint_probability_dim_x;
	int edge_joint_probability_dim_y;

	/**
	 * The array of current beliefs
	 */
	struct belief * edges_messages;
	float *edges_messages_current;
	float *edges_messages_previous;
	int *edges_messages_size;

	/**
	 * Array of belief states indexed by node
	 */
	struct belief * node_states;
	float *node_states_current;
	float *node_states_previous;
	int *node_states_size;

	/**
	 * Array of indices in src_nodes_to_edges_edge_list indexed by their source node
	 */
	int * src_nodes_to_edges_node_list;
	/**
	 * Array of edges indexed by their source node
	 */
	int * src_nodes_to_edges_edge_list;

	/**
	 * Array of indices in dest_nodes_to_edges_edge_list indexed by their destination node
	 */
	int * dest_nodes_to_edges_node_list;
	/**
	 * Array of edges index by their destination node
	 */
	int * dest_nodes_to_edges_edge_list;

	/**
	 * Levels in the tree to the nodes there
	 */
	int * levels_to_nodes;
	/**
	 * The size of the level array
	 */
	int num_levels;

	/**
	 * Array of nodes left in the work queue
	 */
	int *work_queue_nodes;

	/**
	 * Array of edges left in the work queue
	 */
	int *work_queue_edges;

	/**
	 * Array for scratch space
	 */
	int *work_queue_scratch;

	/**
	 * Number of items in work queue
	 */
	int num_work_items_nodes;
	int num_work_items_edges;


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
    unsigned char node_hash_table_created;
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

struct htable_index {
	int index;
	TAILQ_ENTRY(htable_index) next_index;
};


/**
 * Entry within the hash table to store the indices and count
 */
struct htable_entry {
	/**
	 * The array of indices
	 */
    TAILQ_HEAD(, htable_index) indices;
    int count;
};

Graph_t create_graph(unsigned long, unsigned long, const struct joint_probability *, int, int);

void graph_add_node(Graph_t, int, const char *);
void graph_add_and_set_node_state(Graph_t, int, const char *, struct belief *);
void graph_set_node_state(Graph_t, int, int, struct belief *);

void graph_add_edge(Graph_t, int, int, int, int);

void set_up_src_nodes_to_edges(Graph_t);
void set_up_dest_nodes_to_edges(Graph_t);
void init_levels_to_nodes(Graph_t);
void calculate_diameter(Graph_t);
void prep_as_page_rank(Graph_t);

void initialize_node(Graph_t, int, int);
void node_set_state(Graph_t, int, int, struct belief *);

void init_edge(Graph_t, int, int, int, int);
void send_message(const struct belief * __restrict__, int, const struct joint_probability * __restrict__, int, int , struct belief *, float *, float *);

void fill_in_node_hash_table(Graph_t);
long find_node_by_name(char *, Graph_t);

void add_index(struct htable_entry *, int index);
void delete_indices(struct htable_entry *);

void graph_destroy_htables(Graph_t);
void graph_destroy(Graph_t);

void propagate_using_levels_start(Graph_t);
void propagate_using_levels(Graph_t, int);

void reset_visited(Graph_t);

void init_previous_edge(Graph_t);
void loopy_propagate_one_iteration(Graph_t);
void loopy_propagate_edge_one_iteration(Graph_t);
void page_rank_one_iteration(Graph_t);
void page_rank_edge_one_iteration(Graph_t);
void viterbi_one_iteration(Graph_t);
void viterbi_edge_one_iteration(Graph_t);

int loopy_propagate_until(Graph_t, float, int);
int loopy_propagate_until_edge(Graph_t, float, int);
int loopy_propagate_until_acc(Graph_t, float, int);
int loopy_propagate_until_edge_acc(Graph_t, float, int);

int page_rank_until(Graph_t, float, int);
int page_rank_until_edge(Graph_t, float, int);
int page_rank_until_acc(Graph_t, float, int);
int page_rank_until_edge_acc(Graph_t, float, int);

int viterbi_until(Graph_t, float, int);
int viterbi_until_edge(Graph_t, float, int);
int viterbi_until_acc(Graph_t, float, int);
int viterbi_until_edge_acc(Graph_t, float, int);

void marginalize(Graph_t);

void print_node(Graph_t, int);
void print_edge(Graph_t, int);
void print_nodes(Graph_t);
void print_edges(Graph_t);
void print_src_nodes_to_edges(Graph_t);
void print_dest_nodes_to_edges(Graph_t);
void print_levels_to_nodes(Graph_t);

void init_work_queue_nodes(Graph_t);
void init_work_queue_subset_edges(Graph_t, int, int);
void init_work_queue_edges(Graph_t);

void update_work_queue_nodes(Graph_t, float);
void update_work_queue_edges(Graph_t, float);

float difference(struct belief *, int, struct belief *, int);

void set_joint_probability_yahoo_web(struct joint_probability *, int *, int *);
void set_joint_probability_twitter(struct joint_probability *, int *, int *);
void set_joint_probability_vc(struct joint_probability *, int *, int *);
void set_joint_probability_32(struct joint_probability *, int *, int *);

#endif /* GRAPH_H_ */
