#include "bnf-xml-parser/bnf-xml-wrapper.h"

int main() {
    test_dog_files("../src/benchmark_files/");
    test_sample_xml_file("../src/benchmark_files/xml2/");

    return 0;
}