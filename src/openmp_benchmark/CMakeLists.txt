cmake_minimum_required(VERSION 2.8)
find_package(OpenMP)

set(CMAKE_C_FLAGS_RELEASE "-O3 ${OpenMP_C_FLAGS}")

project(benchmark_openmp)

add_executable(openmp_benchmark main.c ../bnf-parser/Parser.c ../bnf-parser/Lexer.c ../bnf-parser/expression.c ../graph/graph.c ../edge/edge.c ../node/node.c)