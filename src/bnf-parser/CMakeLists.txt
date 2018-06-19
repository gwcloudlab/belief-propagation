project(test-expression)

cmake_minimum_required(VERSION 2.8)
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

set(CMAKE_C_FLAGS_RELEASE "-O3")

BISON_TARGET(Parser Parser.y ${CMAKE_CURRENT_SOURCE_DIR}/Parser.c)
FLEX_TARGET(Lexer Lexer.l ${CMAKE_CURRENT_SOURCE_DIR}/Lexer.c )
ADD_FLEX_BISON_DEPENDENCY(Lexer Parser)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(test_expression main.c ${CMAKE_CURRENT_SOURCE_DIR}/Parser.c ${CMAKE_CURRENT_SOURCE_DIR}/Lexer.c expression.c bnf-wrapper.c ../graph/graph.c)