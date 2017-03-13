cmake_minimum_required(VERSION 3.5)
find_package(CUDA QUIET REQUIRED)

project(benchmark_openacc)

FIND_PROGRAM(M_pg_cc NAMES /c1/apps/pgi/linux86-64/10.5/bin/pgcc)
MESSAGE(STATUS "Mutatee pgcc: ${M_pg_cc}")
set(CMAKE_C_FLAGS_RELEASE "-O3 -acc -ta=nvidia")
set(CMAKE_C_FLAGS_DEBUG  "${CMAKE_C_FLAGS_DEBUG} -acc -Minfo=accel -ta=nvidia")
set (CMAKE_C_COMPILE_OBJECT "${M_pg_cc} -c <FLAGS> -o <OBJECT> -c <SOURCE>")

set (CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})
set (CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set (CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
set (CMAKE_CXX_COMPILE_OBJECT ${CMAKE_C_COMPILE_OBJECT})

add_executable(openacc_benchmark main.c ../bnf-parser/Parser.c ../bnf-parser/Lexer.c ../bnf-parser/expression.c ../graph/graph.c ../edge/edge.c ../node/node.c)