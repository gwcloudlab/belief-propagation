#include "../bnf-xml-parser/bnf-xml-wrapper.h"

int main(void) {
    FILE * out = fopen("c_benchmark_loopy_edge.csv", "w");
    fprintf(out, "File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Number of Iterations,BP Run Time(s)\n");
    fflush(out);

    run_test_loopy_belief_propagation_edge_xml_file("../benchmark_files/xml2/10_20.xml", out);
    run_test_loopy_belief_propagation_edge_xml_file("../benchmark_files/xml2/100_200.xml", out);
    run_test_loopy_belief_propagation_edge_xml_file("../benchmark_files/xml2/1000_2000.xml", out);
    run_test_loopy_belief_propagation_edge_xml_file("../benchmark_files/xml2/10000_20000.xml", out);
    run_test_loopy_belief_propagation_edge_xml_file("../benchmark_files/xml2/100000_200000.xml", out);
    run_test_loopy_belief_propagation_edge_xml_file("../benchmark_files/xml2/200000_400000.xml", out);
    run_test_loopy_belief_propagation_edge_xml_file("../benchmark_files/xml2/400000_600000.xml", out);
    run_test_loopy_belief_propagation_edge_xml_file("../benchmark_files/xml2/600000_1200000.xml", out);
    run_test_loopy_belief_propagation_edge_xml_file("../benchmark_files/xml2/800000_1600000.xml", out);
    run_test_loopy_belief_propagation_edge_xml_file("../benchmark_files/xml2/1000000_2000000.xml", out);

    return 0;
}