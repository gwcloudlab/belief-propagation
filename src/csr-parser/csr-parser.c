#include <stdio.h>
#include <stdlib.h>
#include <regex.h>
#include <assert.h>
#include "csr-parser.h"


static unsigned int parse_number_of_nodes(const char *nodes_mtx, regex_t *regex_comment) {
    FILE *fp;
    char buff[255];
    int reti;
    long num_nodes_1, num_nodes_2;
    char *p_end;

    num_nodes_1 = 0;

    fp = fopen(nodes_mtx, "r");
    if(fp == NULL) {
        fprintf(stderr, "Unable to open file: '%s'", nodes_mtx);
        exit(EXIT_FAILURE);
    }

    while ( fgets(buff, 255, fp) != NULL ) {
        reti = regexec(regex_comment, buff, 0, NULL, 0);
        if(reti == REG_NOMATCH) {
            num_nodes_1 = strtol(buff, &p_end, 10);
            num_nodes_2 = strtol(p_end, &p_end, 10);
            assert(num_nodes_1 == num_nodes_2);
            assert(num_nodes_1 >= 0);
            break;
        }
    }

    fclose(fp);
    return (unsigned int)num_nodes_1;
}

static unsigned int parse_number_of_node_states(const char *nodes_mtx, regex_t *regex_comment) {
    FILE *fp;
    char buff[255];
    int reti;
    long node_1, node_2, num_beliefs;
    char *p_end, *prev;
    char no_skip = 0;

    num_beliefs = 0;

    fp = fopen(nodes_mtx, "r");
    if(fp == NULL) {
        fprintf(stderr, "Unable to open file: '%s'", nodes_mtx);
        exit(EXIT_FAILURE);
    }

    while ( fgets(buff, 255, fp) != NULL ) {
        reti = regexec(regex_comment, buff, 0, NULL, 0);
        if(reti == REG_NOMATCH && no_skip == 0) {
            no_skip = 1;
        }
        else if(reti == REG_NOMATCH && no_skip > 0) {
            node_1 = strtol(buff, &p_end, 10);
            node_2 = strtol(p_end, &p_end, 10);
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
    return (unsigned int)num_beliefs;
}

static unsigned int parse_number_of_edges(const char *edges_mtx, regex_t *regex_comment) {
    FILE *fp;
    char buff[255];
    int reti;
    long num_cols, num_rows, num_non_zeroes;
    char *p_end;

    fp = fopen(edges_mtx, "r");
    if(fp == NULL) {
        fprintf(stderr, "Unable to open file: '%s'", edges_mtx);
        exit(EXIT_FAILURE);
    }

    num_non_zeroes = 0;

    while ( fgets(buff, 255, fp) != NULL ) {
        reti = regexec(regex_comment, buff, 0, NULL, 0);
        if(reti == REG_NOMATCH) {
            num_cols = strtol(buff, &p_end, 10);
            num_rows = strtol(p_end, &p_end, 10);
            num_non_zeroes = strtol(p_end, &p_end, 10);
            assert(num_rows >= 0);
            assert(num_cols >= 0);
            assert(num_non_zeroes >= 0);
            break;
        }
    }

    fclose(fp);
    return (unsigned int)num_non_zeroes;
}

static unsigned int parse_number_of_joint_probabilities(const char *edges_mtx, regex_t *regex_comment) {
    FILE *fp;
    char buff[255];
    int reti;
    long src_id, dest_id, num_joint_probs;
    char *p_end, *prev;
    char no_skip = 0;

    num_joint_probs = 0;

    fp = fopen(edges_mtx, "r");
    if(fp == NULL) {
        fprintf(stderr, "Unable to open file: '%s'", edges_mtx);
        exit(EXIT_FAILURE);
    }

    while ( fgets(buff, 255, fp) != NULL ) {
        reti = regexec(regex_comment, buff, 0, NULL, 0);
        if(reti == REG_NOMATCH && no_skip == 0) {
            no_skip = 1;
        }
        else if(reti == REG_NOMATCH && no_skip != 0) {
            src_id = strtol(buff, &p_end, 10);
            dest_id = strtol(p_end, &p_end, 10);
            prev = p_end;
            strtof(p_end, &p_end);
            while(prev != p_end) {
                num_joint_probs += 1;
                prev = p_end;
                strtof(p_end, &p_end);
            }
            break;
        }
    }

    fclose(fp);
    return (unsigned int)num_joint_probs;
}

static void add_nodes(Graph_t graph, const char *nodes_mtx, regex_t *comment_regex, unsigned int num_states) {
    FILE *fp;
    char buff[255];
    char name[255];
    char *p_end, *prev;
    int reti;
    char found_header;
    long node_id_1, node_id_2;
    float prob;
    struct belief curr_belief;
    unsigned int curr_belief_index;

    found_header = 0;

    // set up belief
    curr_belief.size = num_states;

    fp = fopen(nodes_mtx, "r");
    if(fp == NULL) {
        fprintf(stderr, "Unable to open file: '%s'", nodes_mtx);
        exit(EXIT_FAILURE);
    }

    while ( fgets(buff, 255, fp) != NULL ) {
        curr_belief_index = 0;

        reti = regexec(comment_regex, buff, 0, NULL, 0);
        if(reti == REG_NOMATCH) {
            // skip over header line
            if(found_header == 0) {
                found_header = 1;
            }
            else {
                node_id_1 = strtol(buff, &p_end, 10);
                node_id_2 = strtol(p_end, &p_end, 10);

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
                    graph_add_and_set_node_state(graph, 1, name, &curr_belief);
                }
                else {
                    graph_add_node(graph, 1, name);
                }
            }
        }
    }

    fclose(fp);
}

static void add_edges(Graph_t graph, const char *edges_mtx, regex_t *comment_regex, unsigned int num_states,
        unsigned int num_probabilities) {
    FILE *fp;
    char buff[255];
    int reti;
    char found_header;
    char *p_end, *prev;
    long src_id, dest_id;
    unsigned int src_index, dest_index, x, y;
    float prob;
    struct joint_probability joint_probability;

    joint_probability.dim_x = num_states;
    joint_probability.dim_y = num_states;

    found_header = 0;
    fp = fopen(edges_mtx, "r");
    if(fp == NULL) {
        fprintf(stderr, "Unable to open file: '%s'", edges_mtx);
        exit(EXIT_FAILURE);
    }

    while ( fgets(buff, 255, fp) != NULL ) {
        x = 0;
        y = 0;

        reti = regexec(comment_regex, buff, 0, NULL, 0);
        if(reti == REG_NOMATCH) {
            if(found_header == 0) {
                found_header = 1;
            }
            else {
                src_id = strtol(buff, &p_end, 10);
                dest_id = strtol(p_end, &p_end, 10);

                assert(src_id > 0);
                assert(dest_id > 0);

                src_index = (unsigned int)(src_id - 1);
                dest_index = (unsigned int)(dest_id - 1);

                prev = p_end;
                prob = strtof(p_end, &p_end);
                while(p_end != prev) {
                    if(y >= num_states) {
                        x++;
                        y = 0;
                    }
                    assert(prob >= 0.0f);
                    assert(prob <= 1.0f);
                    joint_probability.data[x][y] = prob;
                    y++;
                    prev = p_end;
                    prob = strtof(p_end, &p_end);
                }
                assert(y == num_states);
                assert(x+1 == num_states);

                graph_add_edge(graph, src_index, dest_index, 1, 1, &joint_probability);
            }
        }
    }

    fclose(fp);
}

Graph_t build_graph_from_mtx(const char *edges_mtx, const char *nodes_mtx) {
    regex_t regex_comment;
    int reti;
    unsigned int num_nodes, num_edges, num_node_states, num_joint_probabilities;
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
    num_joint_probabilities = parse_number_of_joint_probabilities(edges_mtx, &regex_comment);
    assert(num_joint_probabilities > 0);
    assert(num_joint_probabilities <= MAX_STATES * MAX_STATES);

    // not necessarily true but true for generated stuff
    assert(num_joint_probabilities == num_node_states * num_node_states);

    graph = create_graph(num_nodes, num_edges);
    add_nodes(graph, nodes_mtx, &regex_comment, num_node_states);
    add_edges(graph, edges_mtx, &regex_comment, num_node_states, num_joint_probabilities);

    regfree(&regex_comment);

    return graph;
}

