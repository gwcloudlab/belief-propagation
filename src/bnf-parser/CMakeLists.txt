cmake_minimum_required(VERSION 2.8)
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

project(test-expression)
BISON_TARGET(Parser Parser.y ${CMAKE_CURRENT_SOURCE_DIR}/Parser.c)
FLEX_TARGET(Lexer Lexer.l ${CMAKE_CURRENT_SOURCE_DIR}/Lexer.c )
ADD_FLEX_BISON_DEPENDENCY(Lexer Parser)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(main main.c ${CMAKE_CURRENT_SOURCE_DIR}/Parser.c ${CMAKE_CURRENT_SOURCE_DIR}/Lexer.c expression.c ../graph/graph.c ../edge/edge.c ../node/node.c)