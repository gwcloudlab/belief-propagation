project(benchmark_openacc)

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
find_package(LibXml2 REQUIRED)

#set(CMAKE_C_COMPILER /opt/pgi/linux86-64/16.10/bin/pgcc)
set(CMAKE_C_COMPILER ${HOME}/pgi/linux86-64/16.10/bin/pgcc)
set(CMAKE_C_FLAGS_RELEASE "-O3 -acc -ta=nvidia")
set(CMAKE_C_FLAGS_DEBUG  "${CMAKE_C_FLAGS_DEBUG} -acc -Minfo=accel -ta=nvidia,time -Msafeptr")
SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

add_executable(openacc_non_loopy_benchmark non-loopy-belief-propagation.c ../bnf-xml-parser/bnf-xml-wrapper.c ../bnf-xml-parser/xml-expression.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/expression.c ../bnf-parser/Lexer.c ../bnf-parser/Parser.c ../graph/graph.c ../snap-parser/snap-parser.c)
add_executable(openacc_loopy_node_benchmark loopy-node-belief-propagation.c ../bnf-xml-parser/bnf-xml-wrapper.c ../bnf-xml-parser/xml-expression.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/expression.c ../bnf-parser/Lexer.c ../bnf-parser/Parser.c ../graph/graph.c ../snap-parser/snap-parser.c)
add_executable(openacc_loopy_edge_benchmark loopy-node-belief-propagation.c ../bnf-xml-parser/bnf-xml-wrapper.c ../bnf-xml-parser/xml-expression.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/expression.c ../bnf-parser/Lexer.c ../bnf-parser/Parser.c ../graph/graph.c ../snap-parser/snap-parser.c)

include_directories(${LIBXML2_INCLUDE_DIR})

target_link_libraries(openacc_non_loopy_benchmark ${LIBXML2_LIBRARIES})
target_link_libraries(openacc_loopy_node_benchmark ${LIBXML2_LIBRARIES})
target_link_libraries(openacc_loopy_edge_benchmark ${LIBXML2_LIBRARIES})