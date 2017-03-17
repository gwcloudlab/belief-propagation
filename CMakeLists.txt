cmake_minimum_required(VERSION 2.8)

add_subdirectory(src/bnf-parser)
add_subdirectory(src/bnf-xml-parser)
add_subdirectory(src/example-belief-propagation)
add_subdirectory(src/benchmark_files)
add_subdirectory(src/c_benchmark)
add_subdirectory(src/openmp_benchmark)
add_subdirectory(src/cuda_benchmark)
add_subdirectory(src/openacc_benchmark EXCLUDE_FROM_ALL)

