cmake_minimum_required(VERSION 3.5)

set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_DEBUG  "-O0 -g -Wall")

project(benchmark_c)

add_executable(c_benchmark main.c ../bnf-parser/Parser.c ../bnf-parser/Lexer.c ../bnf-parser/expression.c ../graph/graph.c ../edge/edge.c ../node/node.c)