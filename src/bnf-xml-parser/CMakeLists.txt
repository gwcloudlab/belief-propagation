cmake_minimum_required(VERSION 3.0)
find_package(LibXml2 REQUIRED)

project(test-xml-expression)

add_executable(test-xml-expression main.c xml-expression.c ../bnf-parser/Parser.c ../bnf-parser/Lexer.c ../bnf-parser/expression.c ../graph/graph.c ../edge/edge.c ../node/node.c)

include_directories(${LIBXML2_INCLUDE_DIR})
target_link_libraries(test-xml-expression ${LIBXML2_LIBRARIES})