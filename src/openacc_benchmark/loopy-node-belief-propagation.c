#include "../bnf-xml-parser/bnf-xml-wrapper.h"
#include "../csr-parser/csr-wrapper.h"

int main(void) {
    FILE * out = fopen("openacc_benchmark_loopy_node.csv", "w");
    fprintf(out, "File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Max In-degree,Avg In-degree,Max Out-degree,Avg Out-degree,Number of Iterations,BP Run Time(s),BP Run Time Per Iteration (s),Total Run Time(s)\n");
    fflush(out);

    struct joint_probability edge_joint_probability;
    size_t dim_x, dim_y;
    set_joint_probability_32(&edge_joint_probability, &dim_x, &dim_y);
/*
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/10_nodes_40_edges_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/10_nodes_40_edges_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/100_nodes_400_edges_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/100_nodes_400_edges_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/1000_nodes_4000_edges_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/1000_nodes_4000_edges_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/10000_nodes_40000_edges_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/10000_nodes_40000_edges_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/100000_nodes_400000_edges_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/100000_nodes_400000_edges_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/200000_nodes_800000_edges_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/200000_nodes_800000_edges_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/400000_nodes_1600000_edges_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/400000_nodes_1600000_edges_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/600000_nodes_2400000_edges_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/600000_nodes_2400000_edges_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/800000_nodes_3200000_edges_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/800000_nodes_3200000_edges_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/1000000_nodes_4000000_edges_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/1000000_nodes_4000000_edges_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/2000000_nodes_8000000_edges_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/2000000_nodes_8000000_edges_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
*/
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/hollywood-2009_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/hollywood-2009_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/soc-pokec-relationships_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/soc-pokec-relationships_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/kron_g500-logn16_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/kron_g500-logn16_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/kron_g500-logn19_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/kron_g500-logn19_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/kron_g500-logn20_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/kron_g500-logn20_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/kron_g500-logn21_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/kron_g500-logn21_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/wiki-Talk_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/wiki-Talk_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/soc-LiveJournal1_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/soc-LiveJournal1_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/loc-gowalla_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/loc-gowalla_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/com-youtube_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/com-youtube_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/wikipedia_link_en_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/wikipedia_link_en_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/friendster_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/friendster_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);

    /*
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/soc-delicious_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/soc-delicious_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/soc-twitter-follows-mun_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/soc-twitter-follows-mun_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/soc-google-plus_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/soc-google-plus_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/web-Stanford_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/web-Stanford_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/web-it-2004_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/web-it-2004_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/web-wiki-ch-internal_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/web-wiki-ch-internal_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);

    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/kron_g500-logn18_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/kron_g500-logn18_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/tech-p2p_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/tech-p2p_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/kron_g500-logn17_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/kron_g500-logn17_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/soc-orkut_32_beliefs.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/soc-orkut_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
    run_test_loopy_belief_propagation_mtx_files_acc("/home/ubuntu/belief-network-const-joint-probability/soc-twitter-2010_32_beliefs_red.edges.mtx", "/home/ubuntu/belief-network-const-joint-probability/soc-twitter-2010_32_beliefs.nodes.mtx", &edge_joint_probability, dim_x, dim_y, out);
*/

    fclose(out);
    return 0;
}