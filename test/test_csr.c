#include <csr-parser/csr-wrapper.h>

int main() {
    test_10_20_file("../src/benchmark_files/10_20.bif.edges.mtx", "../src/benchmark_files/10_20.bif.nodes.mtx");
    test_10_20_file("../src/benchmark_files/test_10_beliefs.edges.mtx", "../src/benchmark_files/test_10_beliefs.nodes.mtx");
}