#include "belief-propagation.hpp"

int main(void) {
    FILE * out = fopen("cuda_benchmark_loopy_edge.csv", "w");
    fprintf(out, "File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)\n");
    fflush(out);

    /*
    run_test_loopy_belief_propagation_xml_file_edge_cuda("../benchmark_files/xml2/10_20.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("../benchmark_files/xml2/100_200.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("../benchmark_files/xml2/1000_2000.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("../benchmark_files/xml2/10000_20000.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("../benchmark_files/xml2/100000_200000.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("../benchmark_files/xml2/200000_400000.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("../benchmark_files/xml2/400000_600000.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("../benchmark_files/xml2/600000_1200000.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("../benchmark_files/xml2/800000_1600000.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("../benchmark_files/xml2/1000000_2000000.xml", out);
*/
    /*
    run_test_loopy_belief_propagation_xml_file_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/10_20.bif.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/100_200.bif.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/1000_2000.bif.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/10000_20000.bif.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/100000_200000.bif.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/200000_400000.bif.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/400000_800000.bif.xml", out);
    run_test_loopy_belief_propagation_xml_file_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/800000_1600000.bif.xml", out);*/

    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/10_20.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/10_20.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/100_200.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/100_200.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/1000_2000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/1000_2000.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/10000_20000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/10000_20000.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/100000_200000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/100000_200000.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/200000_400000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/200000_400000.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/400000_800000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/400000_800000.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/800000_1600000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/800000_1600000.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/1000000_2000000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/1000000_2000000.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/2000000_4000000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/2000000_4000000.bif.nodes.mtx", out);
    run_test_loopy_belief_propagation_mtx_files_edge_cuda("/home/mjt5v/Desktop/gunrock_benchmark_files/4000000_8000000.bif.edges.mtx", "/home/mjt5v/Desktop/gunrock_benchmark_files/4000000_8000000.bif.nodes.mtx", out);


    return 0;
}