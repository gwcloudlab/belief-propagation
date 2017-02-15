cmake_minimum_required(VERSION 2.8)
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)


project(example-belief-propagation)
add_executable(example-belief-propagation src/example-belief-propagation/example-belief-propagation.c src/graph/graph.c src/node/node.c src/edge/edge.c)

project(test-expression)
BISON_TARGET(Parser src/bnf-parser/Parser.y ../src/bnf-parser/Parser.c)
FLEX_TARGET(Lexer src/bnf-parser/Lexer.l ../src/bnf-parser/Lexer.c )
ADD_FLEX_BISON_DEPENDENCY(Lexer Parser)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(main src/bnf-parser/main.c ${BISON_Parser_OUTPUTS} ${FLEX_Lexer_OUTPUTS} src/bnf-parser/expression.c src/graph/graph.c src/edge/edge.c src/node/node.c)