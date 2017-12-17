#include "belief-propagation-kernels.hpp"

int main(void) {
    FILE * out = fopen("cuda_kernels_benchmark_loopy_node.csv", "w");
    fprintf(out, "File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)\n");
    fflush(out);

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


    return 0;
}