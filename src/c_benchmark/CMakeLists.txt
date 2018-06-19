project(benchmark_c)

cmake_minimum_required(VERSION 2.8)
find_package(LibXml2 REQUIRED)

set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")

add_executable(c_non_loopy_benchmark non-loopy-belief-propagation.c ../bnf-xml-parser/bnf-xml-wrapper.c ../bnf-xml-parser/xml-expression.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/expression.c ../bnf-parser/Lexer.c ../bnf-parser/Parser.c ../graph/graph.c ../snap-parser/snap-parser.c ../csr-parser/csr-parser.c ../csr-parser/csr-wrapper.c)
add_executable(c_loopy_node_benchmark loopy-node-belief-propagation.c ../bnf-xml-parser/bnf-xml-wrapper.c ../bnf-xml-parser/xml-expression.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/expression.c ../bnf-parser/Lexer.c ../bnf-parser/Parser.c ../graph/graph.c ../snap-parser/snap-parser.c  ../csr-parser/csr-parser.c ../csr-parser/csr-wrapper.c)
add_executable(c_loopy_edge_benchmark loopy-edge-belief-propagation.c ../bnf-xml-parser/bnf-xml-wrapper.c ../bnf-xml-parser/xml-expression.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/expression.c ../bnf-parser/Lexer.c ../bnf-parser/Parser.c ../graph/graph.c ../snap-parser/snap-parser.c  ../csr-parser/csr-parser.c ../csr-parser/csr-wrapper.c)

include_directories(${LIBXML2_INCLUDE_DIR})

target_link_libraries(c_non_loopy_benchmark ${LIBXML2_LIBRARIES} m)
target_link_libraries(c_loopy_node_benchmark ${LIBXML2_LIBRARIES} m)
target_link_libraries(c_loopy_edge_benchmark ${LIBXML2_LIBRARIES} m)