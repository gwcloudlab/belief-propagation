
#include "../bnf-xml-parser/bnf-xml-wrapper.h"

int main(void) {
    FILE * out = fopen("openacc_benchmark_non_loopy.csv", "w");
    fprintf(out, "File Name,Propagation Type,Number of Nodes,Number of Edges,Diameter,Max In-degree,Avg In-degree,Max Out-degree,Avg Out-degree,Number of Iterations,BP Run Time(s)\n");
    fflush(out);

    /*
    run_test_belief_propagation_xml_file_acc("../benchmark_files/xml2/10_20.xml", out);
    run_test_belief_propagation_xml_file_acc("../benchmark_files/xml2/100_200.xml", out);
    run_test_belief_propagation_xml_file_acc("../benchmark_files/xml2/1000_2000.xml", out);
    run_test_belief_propagation_xml_file_acc("../benchmark_files/xml2/10000_20000.xml", out);
    run_test_belief_propagation_xml_file_acc("../benchmark_files/xml2/100000_200000.xml", out);
    run_test_belief_propagation_xml_file_acc("../benchmark_files/xml2/200000_400000.xml", out);
    run_test_belief_propagation_xml_file_acc("../benchmark_files/xml2/400000_600000.xml", out);
    run_test_belief_propagation_xml_file_acc("../benchmark_files/xml2/600000_1200000.xml", out);
    run_test_belief_propagation_xml_file_acc("../benchmark_files/xml2/800000_1600000.xml", out);
    run_test_belief_propagation_xml_file_acc("../benchmark_files/xml2/1000000_2000000.xml", out);
     */
    run_test_belief_propagation_xml_file_acc("/mnt/raid0_huge/micheal/gunrock_benchmark_files/10_20.bif.xml", out);
    run_test_belief_propagation_xml_file_acc("/mnt/raid0_huge/micheal/gunrock_benchmark_files/100_200.bif.xml", out);
    run_test_belief_propagation_xml_file_acc("/mnt/raid0_huge/micheal/gunrock_benchmark_files/1000_2000.bif.xml", out);
    run_test_belief_propagation_xml_file_acc("/mnt/raid0_huge/micheal/gunrock_benchmark_files/10000_20000.bif.xml", out);
    run_test_belief_propagation_xml_file_acc("/mnt/raid0_huge/micheal/gunrock_benchmark_files/100000_200000.bif.xml", out);
    run_test_belief_propagation_xml_file_acc("/mnt/raid0_huge/micheal/gunrock_benchmark_files/200000_400000.bif.xml", out);
    run_test_belief_propagation_xml_file_acc("/mnt/raid0_huge/micheal/gunrock_benchmark_files/400000_800000.bif.xml", out);
    run_test_belief_propagation_xml_file_acc("/mnt/raid0_huge/micheal/gunrock_benchmark_files/800000_1600000.bif.xml", out);

    return 0;
}
