project(benchmark_openmp)

cmake_minimum_required(VERSION 2.8)
find_package(OpenMP REQUIRED)
find_package(LibXml2 REQUIRED)

set(CMAKE_C_FLAGS_DEBUG  "${CMAKE_C_FLAGS_DEBUG} ${OpenMP_C_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native ${OpenMP_C_FLAGS}")

add_executable(openmp_node_benchmark loopy-node-belief-propagation.c ../bnf-xml-parser/bnf-xml-wrapper.c ../bnf-xml-parser/xml-expression.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/expression.c ../bnf-parser/Lexer.c ../bnf-parser/Parser.c ../graph/graph.c ../snap-parser/snap-parser.c  ../csr-parser/csr-parser.c ../csr-parser/csr-wrapper.c)
add_executable(openmp_edge_benchmark loopy-edge-belief-propagation.c ../bnf-xml-parser/bnf-xml-wrapper.c ../bnf-xml-parser/xml-expression.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/expression.c ../bnf-parser/Lexer.c ../bnf-parser/Parser.c ../graph/graph.c ../snap-parser/snap-parser.c  ../csr-parser/csr-parser.c ../csr-parser/csr-wrapper.c)

include_directories(${LIBXML2_INCLUDE_DIR})

target_link_libraries(openmp_node_benchmark ${LIBXML2_LIBRARIES})
target_link_libraries(openmp_edge_benchmark ${LIBXML2_LIBRARIES})