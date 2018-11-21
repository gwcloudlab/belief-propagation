#include "belief-propagation-kernels.hpp"

int main(void) {
    FILE * out = fopen("cuda_kernels_benchmark_loopy_node.csv", "w");
    fprintf(out, "File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)\n");
    fflush(out);

    /*
    run_test_loopy_belief_propagation_xml_file_kernels("../benchmark_files/xml2/10_20.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("../benchmark_files/xml2/100_200.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("../benchmark_files/xml2/1000_2000.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("../benchmark_files/xml2/10000_20000.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("../benchmark_files/xml2/100000_200000.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("../benchmark_files/xml2/200000_400000.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("../benchmark_files/xml2/400000_600000.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("../benchmark_files/xml2/600000_1200000.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("../benchmark_files/xml2/800000_1600000.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("../benchmark_files/xml2/1000000_2000000.xml", out);
*/
    /*
    run_test_loopy_belief_propagation_xml_file_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/10_20.bif.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/100_200.bif.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/1000_2000.bif.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/10000_20000.bif.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/100000_200000.bif.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/200000_400000.bif.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/400000_800000.bif.xml", out);
    run_test_loopy_belief_propagation_xml_file_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/800000_1600000.bif.xml", out);*/
/*

    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/10_20.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/10_20.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/100_200.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/100_200.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/1000_2000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/1000_2000.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/10000_20000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/10000_20000.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/100000_200000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/100000_200000.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/200000_400000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/200000_400000.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/400000_800000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/400000_800000.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/gunrock_benchmark_files/800000_1600000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/800000_1600000.bif.nodes.mtx", out);
*/
/*
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/10_nodes_20_edges_10_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/10_nodes_20_edges_10_beliefs.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/100_nodes_200_edges_10_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/100_nodes_200_edges_10_beliefs.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/1000_nodes_2000_edges_10_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/1000_nodes_2000_edges_10_beliefs.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/10000_nodes_20000_edges_10_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/10000_nodes_20000_edges_10_beliefs.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/100000_nodes_200000_edges_10_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/100000_nodes_200000_edges_10_beliefs.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/200000_nodes_400000_edges_10_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/200000_nodes_400000_edges_10_beliefs.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/400000_nodes_800000_edges_10_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/400000_nodes_800000_edges_10_beliefs.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/600000_nodes_1200000_edges_10_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/600000_nodes_1200000_edges_10_beliefs.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/800000_nodes_1600000_edges_10_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/800000_nodes_1600000_edges_10_beliefs.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/1000000_nodes_2000000_edges_10_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/1000000_nodes_2000000_edges_10_beliefs.nodes.mtx", out);
*/

    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/10_20_32_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/10_20_32_beliefs.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/100_200_32_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/100_200_32_beliefs.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/1000_2000_32_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/1000_2000_32_beliefs.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/10000_20000_32_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/10000_20000_32_beliefs.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/100000_200000_32_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/100000_200000_32_beliefs.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_kernels("/home/mjt5v/Desktop/belief_network/200000_400000_32_beliefs.edges.mtx", "/home/mjt5v/Desktop/belief_network/200000_400000_32_beliefs.nodes.mtx", out);


    fclose(out);

    return 0;
}