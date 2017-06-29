cmake_minimum_required(VERSION 2.8)

include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/benchmark_files)

add_executable(test_simple_expression test_simple_expression.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/bnf-wrapper.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Parser.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Lexer.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/expression.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c)
add_executable(test_small_expressions test_small_expressions.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/bnf-wrapper.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Parser.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Lexer.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/expression.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c)
add_executable(test_medium_expressions test_medium_expressions.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/bnf-wrapper.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Parser.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Lexer.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/expression.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c)
add_executable(test_large_expressions test_large_expressions.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/bnf-wrapper.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Parser.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Lexer.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/expression.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c)
add_executable(test_very_large_expressions test_very_large_expressions.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/bnf-wrapper.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Parser.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Lexer.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/expression.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c)

add_test(test_simple_expression test_simple_expression)
add_test(test_small_expressions test_small_expressions)
add_test(test_medium_expressions test_medium_expressions)
add_test(test_large_expressions test_large_expressions)
add_test(test_very_large_expressions test_very_large_expressions)