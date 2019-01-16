#include "belief-propagation.hpp"

int main(void) {
    FILE * out = fopen("cuda_benchmark_loopy_edge.csv", "w");
    fprintf(out, "File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)\n");
    fflush(out);

    struct joint_probability edge_joint_probability;
    int dim_x, dim_y;
    set_joint_probability_twitter(&edge_joint_probability, &dim_x, &dim_y);

    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/10_nodes_40_edges_3_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/10_nodes_40_edges_3_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/100_nodes_400_edges_3_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/100_nodes_400_edges_3_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/1000_nodes_4000_edges_3_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/1000_nodes_4000_edges_3_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/10000_nodes_40000_edges_3_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/10000_nodes_40000_edges_3_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/100000_nodes_400000_edges_3_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/100000_nodes_400000_edges_3_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/200000_nodes_800000_edges_3_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/200000_nodes_800000_edges_3_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/400000_nodes_1600000_edges_3_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/400000_nodes_1600000_edges_3_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/600000_nodes_2400000_edges_3_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/600000_nodes_2400000_edges_3_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/800000_nodes_3200000_edges_3_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/800000_nodes_3200000_edges_3_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/1000000_nodes_4000000_edges_3_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/1000000_nodes_4000000_edges_3_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);

    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-delicious_3.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-delicious_3.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-follows-mun_3.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-follows-mun_3.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-google-plus_3.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-google-plus_3.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/web-Stanford_3.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/web-Stanford_3.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/web-it-2004_3_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/web-it-2004_3_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/web-wiki-ch-internal.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/web-wiki-ch-internal.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    //run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-2010_3.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-2010_3.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);


    fclose(out);
    return 0;
}