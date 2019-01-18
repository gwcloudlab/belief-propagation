#include <stdio.h>
#include <stdlib.h>
#include <regex.h>
#include <assert.h>
#include "csr-parser.h"

#define READ_CHAR_BUFFER_SIZE 102400


static size_t parse_number_of_nodes(const char *nodes_mtx, regex_t *regex_comment) {
    FILE *fp;
    char buff[READ_CHAR_BUFFER_SIZE];
    int reti;
    size_t num_nodes_1, num_nodes_2;
    char *p_end;

    num_nodes_1 = 0;

    fp = fopen(nodes_mtx, "r");
    if(fp == NULL) {
        fprintf(stderr, "Unable to open file: '%s'", nodes_mtx);
        exit(EXIT_FAILURE);
    }

    while ( fgets(buff, READ_CHAR_BUFFER_SIZE, fp) != NULL ) {
        reti = regexec(regex_comment, buff, 0, NULL, 0);
        if(reti == REG_NOMATCH) {
            num_nodes_1 = strtoul(buff, &p_end, 10);
            num_nodes_2 = strtoul(p_end, &p_end, 10);
            assert(num_nodes_1 == num_nodes_2);
            assert(num_nodes_1 >= 0);
            break;
        }
    }

    fclose(fp);
    return num_nodes_1;
}

static size_t parse_number_of_node_states(const char *nodes_mtx, regex_t *regex_comment) {
    FILE *fp;
    char buff[READ_CHAR_BUFFER_SIZE];
    int reti;
    size_t node_1, node_2, num_beliefs;
    char *p_end, *prev;
    char no_skip = 0;

    num_beliefs = 0;

    fp = fopen(nodes_mtx, "r");
    if(fp == NULL) {
        fprintf(stderr, "Unable to open file: '%s'", nodes_mtx);
        exit(EXIT_FAILURE);
    }

    while ( fgets(buff, READ_CHAR_BUFFER_SIZE, fp) != NULL ) {
        reti = regexec(regex_comment, buff, 0, NULL, 0);
        if(reti == REG_NOMATCH && no_skip == 0) {
            no_skip = 1;
        }
        else if(reti == REG_NOMATCH && no_skip > 0) {
            node_1 = strtoul(buff, &p_end, 10);
            node_2 = strtoul(p_end, &p_end, 10);
            assert(node_1 == node_2);
            prev = p_end;
            strtof(p_end, &p_end);
            while(p_end != prev) {
                num_beliefs += 1;
                prev = p_end;
                strtof(p_end, &p_end);
            }
            break;
        }
    }

    fclose(fp);
    assert(num_beliefs <= MAX_STATES);
    return (size_t)num_beliefs;
}

static size_t parse_number_of_edges(const char *edges_mtx, regex_t *regex_comment) {
    FILE *fp;
    char buff[READ_CHAR_BUFFER_SIZE];
    int reti;
    size_t num_cols, num_rows, num_non_zeroes;
    char *p_end;

    fp = fopen(edges_mtx, "r");
    if(fp == NULL) {
        fprintf(stderr, "Unable to open file: '%s'", edges_mtx);
        exit(EXIT_FAILURE);
    }

    num_non_zeroes = 0;

    while ( fgets(buff, READ_CHAR_BUFFER_SIZE, fp) != NULL ) {
        reti = regexec(regex_comment, buff, 0, NULL, 0);
        if(reti == REG_NOMATCH) {
            num_cols = strtoul(buff, &p_end, 10);
            num_rows = strtoul(p_end, &p_end, 10);
            num_non_zeroes = strtoul(p_end, &p_end, 10);
            assert(num_rows >= 0);
            assert(num_cols >= 0);
            assert(num_non_zeroes >= 0);
            break;
        }
    }

    fclose(fp);
    return num_non_zeroes;
}

static void add_nodes(Graph_t graph, const char *nodes_mtx, regex_t *comment_regex, size_t num_states) {
    FILE *fp;
    char buff[READ_CHAR_BUFFER_SIZE];
    char name[READ_CHAR_BUFFER_SIZE];
    char *p_end, *prev;
    int reti;
    char found_header;
    size_t node_id_1, node_id_2;
    float prob;
    struct belief curr_belief;
    size_t curr_belief_index;

    found_header = 0;

    // set up belief

    fp = fopen(nodes_mtx, "r");
    if(fp == NULL) {
        fprintf(stderr, "Unable to open file: '%s'", nodes_mtx);
        exit(EXIT_FAILURE);
    }

    while ( fgets(buff, READ_CHAR_BUFFER_SIZE, fp) != NULL ) {
        curr_belief_index = 0;

        reti = regexec(comment_regex, buff, 0, NULL, 0);
        if(reti == REG_NOMATCH) {
            // skip over header line
            if(found_header == 0) {
                found_header = 1;
            }
            else {
                node_id_1 = strtoul(buff, &p_end, 10);
                node_id_2 = strtoul(p_end, &p_end, 10);

                assert(node_id_1 == node_id_2);
                assert(node_id_1 >= 1);

                sprintf(name, "Node %ld", node_id_1);
                prev = p_end;
                prob = strtof(p_end, &p_end);
                while(p_end != prev) {
                    assert(prob >= 0.0f);
                    assert(prob <= 1.0f);
                    curr_belief.data[curr_belief_index] = prob;
                    curr_belief_index++;
                    prev = p_end;
                    prob = strtof(p_end, &p_end);
                }
                assert(num_states == curr_belief_index);

                // check if observed node
                if(curr_belief.data[0] < DEFAULT_STATE) {
                    graph_add_and_set_node_state(graph, num_states, name, &curr_belief);
                }
                else {
                    graph_add_node(graph, num_states, name);
                }
            }
        }
    }

    fclose(fp);
}

static void add_edges(Graph_t graph, const char *edges_mtx, regex_t *comment_regex, int num_states) {
    FILE *fp;
    char buff[READ_CHAR_BUFFER_SIZE];
    int reti;
    char found_header;
    char *p_end, *prev;
    size_t src_id, dest_id;
    size_t src_index, dest_index, x, y;
    float prob;

    found_header = 0;
    fp = fopen(edges_mtx, "r");
    if(fp == NULL) {
        fprintf(stderr, "Unable to open file: '%s'", edges_mtx);
        exit(EXIT_FAILURE);
    }

    while ( fgets(buff, READ_CHAR_BUFFER_SIZE, fp) != NULL ) {
        x = 0;
        y = 0;

        reti = regexec(comment_regex, buff, 0, NULL, 0);
        if(reti == REG_NOMATCH) {
            if(found_header == 0) {
                found_header = 1;
            }
            else {
                src_id = strtoul(buff, &p_end, 10);
                dest_id = strtoul(p_end, &p_end, 10);

                assert(src_id > 0);
                assert(dest_id > 0);

                src_index = (size_t)(src_id - 1);
                dest_index = (size_t)(dest_id - 1);

                graph_add_edge(graph, src_index, dest_index, num_states, num_states);
            }
        }
    }

    fclose(fp);
}

Graph_t build_graph_from_mtx(const char *edges_mtx, const char *nodes_mtx, const struct joint_probability * edge_joint_probability, int dim_x, int dim_y) {
    regex_t regex_comment;
    int reti;
    size_t num_nodes, num_edges;
    size_t num_node_states, num_joint_probabilities;
    Graph_t graph;

    // compile comment regex
    reti = regcomp(&regex_comment, "^[[:space:]]*%", 0);
    if (reti) {
        perror("Could not compile regex\n");
        exit(1);
    }

    num_nodes = parse_number_of_nodes(nodes_mtx, &regex_comment);
    num_node_states = parse_number_of_node_states(nodes_mtx, &regex_comment);
    assert(num_node_states > 0);
    assert(num_node_states <= MAX_STATES);
    num_edges = parse_number_of_edges(edges_mtx, &regex_comment);

    graph = create_graph(num_nodes, num_edges, edge_joint_probability, dim_x, dim_y);
    add_nodes(graph, nodes_mtx, &regex_comment, num_node_states);
    add_edges(graph, edges_mtx, &regex_comment, num_node_states);

    regfree(&regex_comment);

    return graph;
}

