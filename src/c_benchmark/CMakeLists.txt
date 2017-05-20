project(benchmark_c)

cmake_minimum_required(VERSION 2.8)
find_package(LibXml2 REQUIRED)

set(CMAKE_C_FLAGS_RELEASE "-O3")


add_executable(c_benchmark main.c ../bnf-parser/Parser.c ../bnf-parser/Lexer.c ../bnf-parser/expression.c ../graph/graph.c ../bnf-xml-parser/xml-expression.c)
include_directories(${LIBXML2_INCLUDE_DIR})
target_link_libraries(c_benchmark ${LIBXML2_LIBRARIES})