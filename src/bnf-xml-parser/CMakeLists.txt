project(test-xml-expression)

cmake_minimum_required(VERSION 2.8)
find_package(LibXml2 REQUIRED)

add_executable(test-xml-expression main.c xml-expression.c bnf-xml-wrapper.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/Parser.c ../bnf-parser/Lexer.c ../bnf-parser/expression.c ../graph/graph.c)

include_directories(${LIBXML2_INCLUDE_DIR})
target_link_libraries(test-xml-expression ${LIBXML2_LIBRARIES} m)