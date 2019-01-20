//
// Created by mjt5v on 3/4/18.
//

#include "csr-wrapper.h"
#include "csr-parser.h"
#include "../../../../../../usr/include/lzma.h"
#include <assert.h>
#include <stdio.h>
#include <time.h>

void test_10_20_file(const char *edges_mtx, const char *nodes_mtx, const struct joint_probability * edge_joint_probability, size_t dim_x, size_t dim_y) {
    Graph_t graph;

    graph = build_graph_from_mtx(edges_mtx, nodes_mtx, edge_joint_probability, dim_x, dim_y);

    assert(graph->current_num_edges == 12);
    assert(graph->current_num_vertices == 10);

    printf("=====Nodes=====\n");
    print_nodes(graph);

    printf("=====Edges=====\n");
    print_edges(graph);

    graph_destroy(graph);
}

void run_test_belief_propagation_mtx_files(const char *edges_mtx, const char *nodes_mtx, const struct joint_probability * edge_joint_probability, size_t dim_x, size_t dim_y, FILE *out) {
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    size_t i;

    // parse files
    graph = build_graph_from_mtx(edges_mtx, nodes_mtx, edge_joint_probability, dim_x, dim_y);
    assert(graph != NULL);

    // set up parallel arrays
    set_up_src_nodes_to_edges(graph);
    set_up_dest_nodes_to_edges(graph);

    // start BP
    start = clock();
    init_levels_to_nodes(graph);

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
    fprintf(out, "%s,regular,%ld,%ld,%d,2,%lf\n", edges_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_mtx_files(const char * edges_mtx, const char * nodes_mtx, const struct joint_probability * edge_joint_probability, size_t dim_x, size_t dim_y, FILE *out) {
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    int num_iterations;

    // read data
    graph = build_graph_from_mtx(edges_mtx, nodes_mtx, edge_joint_probability, dim_x, dim_y);

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
    fprintf(out, "%s,loopy,%ld,%ld,%d,%d,%lf\n", edges_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}


void run_test_loopy_belief_propagation_edge_mtx_files(const char * edges_mtx, const char * nodes_mtx, const struct joint_probability * edge_joint_probability, size_t dim_x, size_t dim_y, FILE *out) {
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    int num_iterations;

    graph = build_graph_from_mtx(edges_mtx, nodes_mtx, edge_joint_probability, dim_x, dim_y);

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
    fprintf(out, "%s,loopy-edge,%ld,%ld,%d,%d,%lf\n", edges_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_mtx_files_acc(const char *edges_mtx, const char *nodes_mtx, const struct joint_probability * edge_joint_probability, size_t dim_x, size_t dim_y, FILE *out) {
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    int num_iterations;

    graph = build_graph_from_mtx(edges_mtx, nodes_mtx, edge_joint_probability, dim_x, dim_y);

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
    fprintf(out, "%s,loopy,%ld,%ld,%d,%d,%lf\n", edges_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_edge_mtx_files_acc(const char *edges_mtx, const char *nodes_mtx, const struct joint_probability * edge_joint_probability, size_t dim_x, size_t dim_y, FILE *out) {
    Graph_t graph;
    clock_t start, end;
    double time_elapsed;
    int num_iterations;

    graph = build_graph_from_mtx(edges_mtx, nodes_mtx, edge_joint_probability, dim_x, dim_y);
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
    fprintf(out, "%s,loopy-edge,%ld,%ld,%d,%d,%lf\n", edges_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, num_iterations, time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}