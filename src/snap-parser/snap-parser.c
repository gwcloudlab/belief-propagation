#include <stdio.h>
#include <stdlib.h>
#include <regex.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "snap-parser.h"

/**
 * Global buffer used for hash lookups
 */
static char match_buffer[READ_SNAP_BUFFER_SIZE];

/**
 * Parses the SNAP edge file for the graph summary
 * @param edge_file The SNAP edge file
 * @param info The graph info object to fill in
 */
static void find_graph_info(const char * edge_file, struct graph_info * info){
    FILE * in;
    char buffer[READ_SNAP_BUFFER_SIZE];
    char * end_ptr;
    regex_t regex;
    regmatch_t groups[5];
    int reti, i;
    unsigned long parsed_long;

    //create the regex
    reti = regcomp(&regex, REGEX_GRAPH_INFO, REG_EXTENDED);
    if(reti){
        perror("Could not compile regex to find graph info\n");
        exit(-1);
    }

    in = fopen(edge_file, "r");
    while(fgets(buffer, READ_SNAP_BUFFER_SIZE, in)){
        reti = regexec(&regex, buffer, 5, groups, 0);
        if(reti == 0) {
            for(i = 1; i < 5; ++i){
                if(groups[i].rm_so == (size_t) - 1){
                    break; // no more groups
                }
                memset(match_buffer, 0, READ_SNAP_BUFFER_SIZE);
                strncpy(match_buffer, buffer + groups[i].rm_so, (size_t)(groups[i].rm_eo - groups[i].rm_so));
                parsed_long = strtoul(match_buffer, &end_ptr, 10);
                if(i == 1){
                    info->num_nodes = (unsigned int)parsed_long;
                }
                else if(i == 2){
                    info->num_edges = (unsigned int)parsed_long;
                }
                else if(i == 3){
                    info->num_beliefs = (unsigned int)parsed_long;
                }
                else {
                    info->num_belief_states = (unsigned int)parsed_long;
                }
            }
            break;
        }
    }
    fclose(in);
}

/**
 * Fills in the nodes in the graph from edge file
 * @param graph The graph to fill in
 * @param info The graph info file
 * @param edge_file The SNAP edge file
 * @param node_hash A hash of nodes to their indices
 */
static void create_nodes(Graph_t graph, struct graph_info * info, const char * edge_file, struct hsearch_data *node_hash){
    ENTRY item, *result;
    FILE *in;
    char buffer[READ_SNAP_BUFFER_SIZE];
    char * end_ptr;
    regex_t regex;
    regmatch_t *groups;
    int reti, i;
    size_t num_groups;
    unsigned int node_index;

    node_index = 0;

    num_groups = 4;
    groups = (regmatch_t *)malloc(sizeof(regmatch_t) * num_groups);

    //create the regex
    reti = regcomp(&regex, REGEX_EDGE_LINE, REG_EXTENDED);
    if(reti){
        perror("Could not compile regex to find edges\n");
        exit(-1);
    }

    in = fopen(edge_file, "r");
    assert(in);

    while(fgets(buffer, READ_SNAP_BUFFER_SIZE, in)){
        reti = regexec(&regex, buffer, num_groups, groups, 0);
        if(reti == 0){
            for(i = 1; i < 3; ++i){
                if(groups[i].rm_so == (size_t) - 1){
                    break; // no more groups
                }
                memset(match_buffer, 0, READ_SNAP_BUFFER_SIZE);
                strncpy(match_buffer, buffer + groups[i].rm_so, (size_t)(groups[i].rm_eo - groups[i].rm_so));
                item.key = match_buffer;
                item.data = NULL;
                hsearch_r(item, FIND, &result, node_hash);
                if(result == NULL){
                    item.data = (void *)node_index;
                    assert(hsearch_r(item, ENTER, &result, node_hash) != 0);
                    graph_add_node(graph, info->num_belief_states, match_buffer);
                    node_index++;
                }
            }
        }
    }

    free(groups);

    fclose(in);
}

/**
 * Fills in the probability for the given float
 * @param dest The float address to write to
 * @param token The string to write
 */
static void fill_in_probability(float *dest, char *token){
    char *end_ptr;

    assert(token);
    *dest = strtof(token, &end_ptr);
    assert(0.0f <= *dest);
    assert(1.0f >= *dest);
}

/**
 * Fills in the observed nodes in the graph
 * @param graph The graph to fill in
 * @param info The graph summary info
 * @param observed_node_file The file containing the observed node beliefs
 * @param node_hash The node hash to look up nodes
 * @param observed_node_hash The observed node hash to fill in
 */
static void add_observed_nodes(Graph_t graph, struct graph_info *info, const char * observed_node_file,
                               struct hsearch_data *node_hash, struct hsearch_data *observed_node_hash){
    FILE *in;
    ENTRY data, *result, *observed_result;
    char buffer[READ_SNAP_BUFFER_SIZE];
    char * token;
    regex_t regex;
    regmatch_t *groups;
    int reti, i, j;
    size_t num_groups;
    struct belief belief;
    unsigned int node_index;

    node_index = 0;

    belief.size = info->num_belief_states;

    num_groups = 3;
    groups = (regmatch_t *)malloc(sizeof(regmatch_t) * num_groups);

    //create the regex
    reti = regcomp(&regex, REGEX_NODE_LINE, REG_EXTENDED);
    if(reti){
        perror("Could not compile regex to find nodes\n");
        exit(-1);
    }

    in = fopen(observed_node_file, "r");
    assert(in);

    while(fgets(buffer, READ_SNAP_BUFFER_SIZE, in)){
        reti = regexec(&regex, buffer, num_groups, groups, 0);
        if(reti == 0){
            for(i = 1; i < num_groups; ++i){
                if(groups[i].rm_so == (size_t) - 1){
                    break; // no more groups
                }
                memset(match_buffer, 0, READ_SNAP_BUFFER_SIZE);
                strncpy(match_buffer, buffer + groups[i].rm_so, (size_t)(groups[i].rm_eo - groups[i].rm_so));
                //node name
                if(i == 1){
                    data.key = match_buffer;
                    data.data = NULL;
                    assert(hsearch_r(data, FIND, &result, node_hash) != 0);
                    node_index = (unsigned int)result->data;
                    // insert into observed node hash
                    assert(hsearch_r(data, ENTER, &observed_result, observed_node_hash) != 0);
                }
                else{
                    token = strtok(match_buffer, REGEX_WHITESPACE);
                    for(j = 0; j < belief.size; ++j){
                        fill_in_probability(&(belief.data[j]), token);
                        token = strtok(NULL, REGEX_WHITESPACE);
                    }
                }
            }
            graph_set_node_state(graph, node_index, info->num_belief_states, &belief);
        }
    }

    free(groups);

    fclose(in);
}

/**
 * Adds edges to the graph
 * @param graph The graph to fill in
 * @param info The graph summary
 * @param edge_file The SNAP edge file
 * @param node_hash The hash containing node ids to their indices
 * @param observed_node_hash The hash containing the node ids of the observed nodes
 */
static void add_edge(Graph_t graph, struct graph_info *info, const char * edge_file,
                     struct hsearch_data *node_hash, struct hsearch_data *observed_node_hash) {
    ENTRY item, *result, *observed_result;
    FILE *in;
    char buffer[READ_SNAP_BUFFER_SIZE];
    char *token;
    regex_t regex;
    regmatch_t *groups;
    int reti, i, j, k;
    size_t num_groups;
    unsigned int src_index, dest_index;
    struct joint_probability joint_probability, inverted_joint_probability;
    char src_to_dest_valid, dest_to_src_valid;

    src_index = 0;
    dest_index = 0;

    joint_probability.dim_x = info->num_belief_states;
    joint_probability.dim_y = info->num_belief_states;

    inverted_joint_probability.dim_y = info->num_belief_states;
    inverted_joint_probability.dim_x = info->num_belief_states;

    num_groups = 4;
    groups = (regmatch_t *) malloc(sizeof(regmatch_t) * num_groups);

    //create the regex
    reti = regcomp(&regex, REGEX_EDGE_LINE, REG_EXTENDED);
    if (reti) {
        perror("Could not compile regex to find edges\n");
        exit(-1);
    }

    in = fopen(edge_file, "r");
    assert(in);

    while (fgets(buffer, READ_SNAP_BUFFER_SIZE, in)) {
        dest_to_src_valid = 0;
        src_to_dest_valid = 0;
        reti = regexec(&regex, buffer, num_groups, groups, 0);
        if (reti == 0) {
            for(i = 1; i < num_groups; ++i) {
                if (groups[i].rm_so == (size_t) -1) {
                    break; // no more groups
                }
                memset(match_buffer, 0, READ_SNAP_BUFFER_SIZE);
                strncpy(match_buffer, buffer + groups[i].rm_so, (size_t)(groups[i].rm_eo - groups[i].rm_so));
                if (i < 3) {
                    item.key = match_buffer;
                    item.data = NULL;
                    assert(hsearch_r(item, FIND, &result, node_hash) != 0);
                    hsearch_r(item, FIND, &observed_result, observed_node_hash);
                    if (i == 1) {
                        src_index = (unsigned int) result->data;
                        if(observed_result == NULL){
                            dest_to_src_valid = 1;
                        }
                    } else {
                        dest_index = (unsigned int) result->data;
                        if(observed_result == NULL){
                            src_to_dest_valid = 1;
                        }
                    }
                } else {
                    token = strtok(match_buffer, REGEX_WHITESPACE);
                    for(j = 0; j < joint_probability.dim_x; ++j){
                        for(k = 0; k < joint_probability.dim_y; ++k){
                            fill_in_probability(&(joint_probability.data[j][k]), token);
                            fill_in_probability(&(inverted_joint_probability.data[k][j]), token);
                            token = strtok(NULL, REGEX_WHITESPACE);
                        }
                    }
                }
            }
            if(src_to_dest_valid == 1) {
                graph_add_edge(graph, src_index, dest_index, info->num_belief_states, info->num_belief_states,
                               &joint_probability);
            }
            if(dest_to_src_valid == 1){
                graph_add_edge(graph, dest_index, src_index, info->num_belief_states, info->num_belief_states,
                               &inverted_joint_probability);
            }
        }
    }

    free(groups);

    fclose(in);
}

/**
 * Generates a graph from the enhanced SNAP files
 * @param edge_file The edge file to read
 * @param observed_node_file The observed node file to read
 * @return A graph
 */
Graph_t parse_graph_from_snap_files(const char * edge_file, const char * observed_node_file) {
    struct graph_info info;
    Graph_t graph;
    struct hsearch_data *node_hash, *observed_node_hash;

    node_hash = calloc(1, sizeof(struct hsearch_data));
    observed_node_hash = calloc(1, sizeof(struct hsearch_data));

    find_graph_info(edge_file, &info);

    assert(hcreate_r(info.num_nodes, node_hash) != 0);
    assert(hcreate_r(info.num_nodes, observed_node_hash) != 0);
/*    assert(info.num_nodes == 5);
    assert(info.num_edges == 5);
    assert(info.num_beliefs == 2);
    assert(info.num_belief_states == 2);*/
    graph = create_graph(info.num_nodes, info.num_edges);

    create_nodes(graph, &info, edge_file, node_hash);
    add_observed_nodes(graph, &info, observed_node_file, node_hash, observed_node_hash);

    add_edge(graph, &info, edge_file, node_hash, observed_node_hash);

    hdestroy_r(node_hash);
    hdestroy_r(observed_node_hash);
    free(node_hash);
    free(observed_node_hash);

    return graph;
}

/**
 * Helper function for building full file paths from relative ones
 * @param prefix The directory holding the path
 * @param file The file within the prefix directory to parse
 * @param buffer The char buffer for the combined path
 * @param length The size of the buffer
 * @return The buffer
 */
static char * build_full_file_path(const char *prefix, const char *file, char *buffer, size_t length){
    size_t prefix_length;
    prefix_length = strlen(prefix);
    assert(prefix_length < length);
    strncpy(buffer, prefix, length);
    strncat(buffer, file, length - prefix_length);
    return buffer;
}

/**
 * Tests parsing a simple SNAP file
 * @param root_dir The directory holding the file
 */
void test_sample_snap_dog_files(const char * root_dir){
    char edge_path[128], node_path[128];
    Graph_t graph;

    build_full_file_path(root_dir, "dog-edges.txt", edge_path, 128);
    build_full_file_path(root_dir, "dog-nodes.txt", node_path, 128);

    graph = parse_graph_from_snap_files(edge_path, node_path);

    // check their content
    printf("SNAP File\n");
    print_nodes(graph);
    print_edges(graph);

    graph_destroy(graph);
}

/**
 * Tests parsing a simple page rank file
 * @param root_dir The directory holding the file
 */
void test_page_rank_sample_file(const char * root_dir){
    char edge_path[128], node_path[128];
    char * node_name;
    char highest_node[CHAR_BUFFER_SIZE];
    unsigned int node_index;
    float highest_belief, current_belief;
    Graph_t graph;

    build_full_file_path(root_dir, "small_page_rank.txt", edge_path, 128);
    build_full_file_path(root_dir, "small_page_rank_node.txt", node_path, 128);

    graph = parse_graph_from_snap_files(edge_path, node_path);
    // set up parallel arrays
    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    // set up as page rank
    prep_as_page_rank(graph);
    init_previous_edge(graph);

    page_rank_until(graph, PRECISION, NUM_ITERATIONS);

    print_nodes(graph);

    highest_belief = graph->node_states[0].data[0];
    // node 8709207 should be the highest
    for(node_index = 0; node_index < graph->current_num_vertices; ++node_index){
        node_name = &graph->node_names[node_index];
        current_belief = graph->node_states[node_index].data[0];
        if(current_belief > highest_belief){
            highest_belief = current_belief;
            strncpy(highest_node, node_name, CHAR_BUFFER_SIZE);
        }
    }

    assert(highest_belief < 19.6);
    assert(highest_belief > 19.5);
    assert(strcmp("8709207", highest_node));

    graph_destroy(graph);
}

/**
 * Tests parsing a simple page rank file
 * @param root_dir The directory holding the file
 */
void test_page_rank_sample_edge_file(const char * root_dir){
    char edge_path[128], node_path[128];
    char * node_name;
    char highest_node[CHAR_BUFFER_SIZE];
    unsigned int node_index;
    float highest_belief, current_belief;
    Graph_t graph;

    build_full_file_path(root_dir, "small_page_rank.txt", edge_path, 128);
    build_full_file_path(root_dir, "small_page_rank_node.txt", node_path, 128);

    graph = parse_graph_from_snap_files(edge_path, node_path);
    // set up parallel arrays
    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    // set up as page rank
    prep_as_page_rank(graph);
    init_previous_edge(graph);

    page_rank_until_edge(graph, PRECISION, NUM_ITERATIONS);

    print_nodes(graph);

    highest_belief = graph->node_states[0].data[0];
    // node 8709207 should be the highest
    for(node_index = 0; node_index < graph->current_num_vertices; ++node_index){
        node_name = &graph->node_names[node_index];
        current_belief = graph->node_states[node_index].data[0];
        if(current_belief > highest_belief){
            highest_belief = current_belief;
            strncpy(highest_node, node_name, CHAR_BUFFER_SIZE);
        }
    }

    assert(highest_belief < 19.6);
    assert(highest_belief > 19.5);
    assert(strcmp("8709207", highest_node));

    graph_destroy(graph);
}

/**
 * Runs BP on the SNAP files
 * @param edge_file The path to the edge file to read
 * @param node_file The path to the node file to read
 * @param out The CSV file to output to
 */
void run_test_belief_propagation_snap_file(const char * edge_file, const char * node_file, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int i;

    // parse the file
    graph = parse_graph_from_snap_files(edge_file, node_file);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    // set up the parallel arrays
    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    // start BP
    start = clock();
    init_levels_to_nodes(graph);
    //print_levels_to_nodes(graph);

    // propagate from the leaves to the root
    propagate_using_levels_start(graph);
    for(i = 1; i < graph->num_levels - 1; ++i){
        propagate_using_levels(graph, i);
    }

    // go back the other way
    reset_visited(graph);
    for(i = graph->num_levels - 1; i > 0; --i){
        propagate_using_levels(graph, i);
    }

    marginalize(graph);
    end = clock();

    // output
    time_elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(out, "%s-%s,regular,%d,%d,%d,2,%lf\n", edge_file, node_file, graph->current_num_vertices, graph->current_num_edges, graph->diameter, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}

/**
 * Runs node-optimized loopy BP on the XML file
 * @param edge_file_name The file to read for the SNAP edges
 * @param node_file_name The file to read for the SNAP observed nodes
 * @param out The CSV file to output to
 */
void run_test_loopy_belief_propagation_snap_file(const char * edge_file_name, const char * node_file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    // read the data
    graph = parse_graph_from_snap_files(edge_file_name, node_file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    // set up parallel arrays
    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    // start loopy BP
    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    // output data
    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy,%d,%d,%d,%d,%lf\n", edge_file_name, node_file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}


/**
 * Runs edge-optimized loopy BP on the file
 * @param edge_file_name The file to read for the SNAP edges
 * @param node_file_name The file to read for the SNAP observed nodes
 * @param out The CSV file to output to
 */
void run_test_loopy_belief_propagation_edge_snap_file(const char * edge_file_name, const char * node_file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    // read the XML file
    graph = parse_graph_from_snap_files(edge_file_name, node_file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    // set up parallel arrays
    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    // start loopy BP
    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_edge(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    // output
    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy-edge,%d,%d,%d,%d,%lf\n", edge_file_name, node_file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}

/**
 * Runs node-optimized loopy BP for OpenACC
 * @param edge_file_name The file to read for the SNAP edges
 * @param node_file_name The file to read for the SNAP observed nodes
 * @param out The CSV file to output to
 */
void run_test_loopy_belief_propagation_snap_file_acc(const char * edge_file_name, const char * node_file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    // read the data
    graph = parse_graph_from_snap_files(edge_file_name, node_file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    // set up parallel arrays
    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    // start loopy bp
    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_acc(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    // output
    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy,%d,%d,%d,%d,%lf\n", edge_file_name, node_file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}

/**
 * Runs edge optimized loopy BP for OpenACC
 * @param edge_file_name The file to read for the SNAP edges
 * @param node_file_name The file to read for the SNAP observed nodes
 * @param out The CSV file to output to
 */
void run_test_loopy_belief_propagation_edge_snap_file_acc(const char * edge_file_name, const char * node_file_name, FILE * out){
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    unsigned int num_iterations;

    // read the data
    graph = parse_graph_from_snap_files(edge_file_name, node_file_name);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    // set up parallel arrays
    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    // start loopy BP
    start = clock();
    init_previous_edge(graph);

    num_iterations = loopy_propagate_until_edge_acc(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    // output
    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s-%s,loopy-edge,%d,%d,%d,%d,%lf\n", edge_file_name, node_file_name, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}
