#include "../bnf-xml-parser/bnf-xml-wrapper.h"
#include "../csr-parser/csr-wrapper.h"

int main(void) {
    FILE * out = fopen("openmp_benchmark_loopy_node_no_work_queue.csv", "w");
    fprintf(out, "File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Max In-degree,Avg In-degree,Max Out-degree,Avg Out-degree,Number of Iterations,BP Run Time(s),BP Run Time Per Iteration (s),Total Run Time(s)\n");
    fflush(out);

    struct joint_probability edge_joint_probability;
    size_t dim_x, dim_y;
    set_joint_probability_yahoo_web(&edge_joint_probability, &dim_x, &dim_y);


    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/10_nodes_40_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/10_nodes_40_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/100_nodes_400_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/100_nodes_400_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/1000_nodes_4000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/1000_nodes_4000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/10000_nodes_40000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/10000_nodes_40000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/100000_nodes_400000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/100000_nodes_400000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/200000_nodes_800000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/200000_nodes_800000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/400000_nodes_1600000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/400000_nodes_1600000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/600000_nodes_2400000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/600000_nodes_2400000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/800000_nodes_3200000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/800000_nodes_3200000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/1000000_nodes_4000000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/1000000_nodes_4000000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/2000000_nodes_8000000_edges_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/2000000_nodes_8000000_edges_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-delicious_2.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-delicious_2.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-follows-mun_2.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-follows-mun_2.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-google-plus_2.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-google-plus_2.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/web-Stanford_2.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/web-Stanford_2.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/web-it-2004_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/web-it-2004_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/web-wiki-ch-internal.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/web-wiki-ch-internal.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);

    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/kron_g500-logn18_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/kron_g500-logn18_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/tech-p2p_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/tech-p2p_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/kron_g500-logn17_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/kron_g500-logn17_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-orkut_2_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-orkut_2_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    //run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-2010_2.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-2010_2.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_no_work_queue("/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-2010_2_red.edges.mtx", "/home/mjt5v/Desktop/belief-network-const-joint-probability/soc-twitter-2010_2.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);

    fclose(out);
    return 0;
}