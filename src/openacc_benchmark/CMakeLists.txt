project(benchmark_openacc)

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
find_package(LibXml2 REQUIRED)

#set(CMAKE_C_COMPILER /opt/pgi/linux86-64/16.10/bin/pgcc)
set(CMAKE_C_COMPILER /home/trotsky/pgi/linux86-64/10.5/bin/pgcc)
set(CMAKE_C_FLAGS_RELEASE "-O3 -acc -ta=nvidia")
set(CMAKE_C_FLAGS_DEBUG  "${CMAKE_C_FLAGS_DEBUG} -acc -Minfo=accel -ta=nvidia,time")
SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

add_executable(openacc_benchmark main.c ../bnf-parser/Parser.c ../bnf-parser/Lexer.c ../bnf-parser/expression.c ../graph/graph.c ../edge/edge.c ../node/node.c ../bnf-xml-parser/xml-expression.c)
include_directories(${LIBXML2_INCLUDE_DIR})
target_link_libraries(openacc_benchmark ${LIBXML2_LIBRARIES})