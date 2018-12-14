#include "../bnf-xml-parser/bnf-xml-wrapper.h"
#include "../csr-parser/csr-wrapper.h"

int main(void) {
    FILE * out = fopen("openmp_benchmark_loopy_node.csv", "w");
    fprintf(out, "File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)\n");
    fflush(out);

    struct joint_probability edge_joint_probability;
    int dim_x, dim_y;
    set_joint_probability_yahoo_web(&edge_joint_probability, &dim_x, &dim_y);

    run_test_loopy_belief_propagation_mtx_files("/home/mjt5v/Desktop/belief-network-const-joint-probability/10_nodes_40_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/10_nodes_40_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files("/home/mjt5v/Desktop/belief-network-const-joint-probability/100_nodes_400_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/100_nodes_400_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files("/home/mjt5v/Desktop/belief-network-const-joint-probability/1000_nodes_4000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/1000_nodes_4000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files("/home/mjt5v/Desktop/belief-network-const-joint-probability/10000_nodes_40000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/10000_nodes_40000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files("/home/mjt5v/Desktop/belief-network-const-joint-probability/100000_nodes_400000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/100000_nodes_400000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files("/home/mjt5v/Desktop/belief-network-const-joint-probability/200000_nodes_800000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/200000_nodes_800000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files("/home/mjt5v/Desktop/belief-network-const-joint-probability/400000_nodes_1600000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/400000_nodes_1600000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);


    fclose(out);
    return 0;
}