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
    clock_t begin, start, end;
    double time_elapsed, total_time_elapsed;
    size_t i;

    begin = clock();

    // parse files
    graph = build_graph_from_mtx(edges_mtx, nodes_mtx, edge_joint_probability, dim_x, dim_y);
    assert(graph != NULL);

    // set up parallel arrays
    set_up_src_nodes_to_edges_no_hsearch(graph);
    set_up_dest_nodes_to_edges_no_hsearch(graph);

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
    total_time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
    fprintf(out, "%s,loopy-edge,%ld,%ld,%d,%d,%lf,%d,%lf,%d,%lf,%lf,%lf\n", edges_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, graph->max_in_degree, graph->avg_in_degree, graph->max_out_degree, graph->avg_out_degree, 2, time_elapsed, time_elapsed/2, total_time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_mtx_files(const char * edges_mtx, const char * nodes_mtx, const struct joint_probability * edge_joint_probability, size_t dim_x, size_t dim_y, FILE *out) {
    Graph_t graph;
    clock_t begin, start, end;
    double time_elapsed, total_time_elapsed;
    int num_iterations;

    begin = clock();

    // read data
    graph = build_graph_from_mtx(edges_mtx, nodes_mtx, edge_joint_probability, dim_x, dim_y);

    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    // set up parallel arrays
    set_up_src_nodes_to_edges_no_hsearch(graph);
    set_up_dest_nodes_to_edges_no_hsearch(graph);
    //print_src_nodes_to_edges(graph);
    //print_dest_nodes_to_edges(graph);
    //calculate_diameter(graph);

    // start loopy BP
    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    // output data
    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    total_time_elapsed = (double)(end - begin)/CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy,%ld,%ld,%d,%d,%lf,%d,%lf,%d,%lf,%lf,%lf\n", edges_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, graph->max_in_degree, graph->avg_in_degree, graph->max_out_degree, graph->avg_out_degree, num_iterations, time_elapsed, time_elapsed/num_iterations, total_time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}


void run_test_loopy_belief_propagation_edge_mtx_files(const char * edges_mtx, const char * nodes_mtx, const struct joint_probability * edge_joint_probability, size_t dim_x, size_t dim_y, FILE *out) {
    Graph_t graph;
    clock_t begin, start, end;
    double time_elapsed, total_time_elapsed;
    int num_iterations;

    begin = clock();

    graph = build_graph_from_mtx(edges_mtx, nodes_mtx, edge_joint_probability, dim_x, dim_y);

    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    // set up parallel arrays
    set_up_src_nodes_to_edges_no_hsearch(graph);
    set_up_dest_nodes_to_edges_no_hsearch(graph);
    //calculate_diameter(graph);

    // start loopy BP
    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_edge(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    // output
    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    total_time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy-edge,%ld,%ld,%d,%d,%lf,%d,%lf,%d,%lf,%lf,%lf\n", edges_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, graph->max_in_degree, graph->avg_in_degree, graph->max_out_degree, graph->avg_out_degree, num_iterations, time_elapsed, time_elapsed/num_iterations, total_time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_mtx_files_acc(const char *edges_mtx, const char *nodes_mtx, const struct joint_probability * edge_joint_probability, size_t dim_x, size_t dim_y, FILE *out) {
    Graph_t graph;
    clock_t begin, start, end;
    double time_elapsed, total_time_elapsed;
    int num_iterations;

    begin = clock();

    graph = build_graph_from_mtx(edges_mtx, nodes_mtx, edge_joint_probability, dim_x, dim_y);

    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    // set up parallel arrays
    set_up_src_nodes_to_edges_no_hsearch(graph);
    set_up_dest_nodes_to_edges_no_hsearch(graph);
    //calculate_diameter(graph);

    // start loopy bp
    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_acc(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    // output
    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    total_time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy,%ld,%ld,%d,%d,%lf,%d,%lf,%d,%lf,%lf,%lf\n", edges_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, graph->max_in_degree, graph->avg_in_degree, graph->max_out_degree, graph->avg_out_degree, num_iterations, time_elapsed, time_elapsed/num_iterations, total_time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}

void run_test_loopy_belief_propagation_edge_mtx_files_acc(const char *edges_mtx, const char *nodes_mtx, const struct joint_probability * edge_joint_probability, size_t dim_x, size_t dim_y, FILE *out) {
    Graph_t graph;
    clock_t begin, start, end;
    double time_elapsed, total_time_elapsed;
    int num_iterations;

    begin = clock();

    graph = build_graph_from_mtx(edges_mtx, nodes_mtx, edge_joint_probability, dim_x, dim_y);
    assert(graph != NULL);
    //print_nodes(graph);
    //print_edges(graph);

    // set up parallel arrays
    set_up_src_nodes_to_edges_no_hsearch(graph);
    set_up_dest_nodes_to_edges_no_hsearch(graph);
    //calculate_diameter(graph);

    // start loopy BP
    init_previous_edge(graph);

    start = clock();
    num_iterations = loopy_propagate_until_edge_acc(graph, PRECISION, NUM_ITERATIONS);
    end = clock();

    // output
    time_elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    total_time_elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
    //print_nodes(graph);
    fprintf(out, "%s,loopy-edge,%ld,%ld,%d,%d,%lf,%d,%lf,%d,%lf,%lf,%lf\n", edges_mtx, graph->current_num_vertices, graph->current_num_edges, graph->diameter, graph->max_in_degree, graph->avg_in_degree, graph->max_out_degree, graph->avg_out_degree, num_iterations, time_elapsed, time_elapsed/num_iterations, total_time_elapsed);
    fflush(out);

    // cleanup
    graph_destroy(graph);
}