cmake_minimum_required(VERSION 2.8)
find_package(LibXml2 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(test_simple_expression test_simple_expression.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/bnf-wrapper.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Parser.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Lexer.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/expression.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c)
add_executable(test_small_expressions test_small_expressions.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/bnf-wrapper.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Parser.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Lexer.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/expression.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c)
add_executable(test_medium_expressions test_medium_expressions.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/bnf-wrapper.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Parser.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Lexer.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/expression.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c)
add_executable(test_large_expressions test_large_expressions.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/bnf-wrapper.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Parser.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Lexer.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/expression.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c)
add_executable(test_very_large_expressions test_very_large_expressions.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/bnf-wrapper.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Parser.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Lexer.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/expression.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c)
add_executable(test_snap_files test_snap_files.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c ${CMAKE_SOURCE_DIR}/src/snap-parser/snap-parser.c)
add_executable(test_page_rank test_page_rank.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c ${CMAKE_SOURCE_DIR}/src/snap-parser/snap-parser.c)
add_executable(test_csr test_csr.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c ${CMAKE_SOURCE_DIR}/src/csr-parser/csr-parser.c ${CMAKE_SOURCE_DIR}/src/csr-parser/csr-wrapper.c)

add_test(test_simple_expression test_simple_expression)
add_test(test_small_expressions test_small_expressions)
add_test(test_medium_expressions test_medium_expressions)
add_test(test_large_expressions test_large_expressions)
add_test(test_very_large_expressions test_very_large_expressions)
add_test(test_snap_files test_snap_files)
add_test(test_page_rank test_page_rank)

include_directories(${LIBXML2_INCLUDE_DIR})
add_executable(test_xml_expressions test_xml_expressions.c ${CMAKE_SOURCE_DIR}/src/bnf-xml-parser/xml-expression.c ${CMAKE_SOURCE_DIR}/src/bnf-xml-parser/bnf-xml-wrapper.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/bnf-wrapper.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Parser.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Lexer.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/expression.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c)
target_link_libraries(test_xml_expressions ${LIBXML2_LIBRARIES})

add_executable(test_viterbi test_viterbi.c ${CMAKE_SOURCE_DIR}/src/bnf-xml-parser/xml-expression.c ${CMAKE_SOURCE_DIR}/src/bnf-xml-parser/bnf-xml-wrapper.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/bnf-wrapper.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Parser.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/Lexer.c ${CMAKE_SOURCE_DIR}/src/bnf-parser/expression.c ${CMAKE_SOURCE_DIR}/src/graph/graph.c)
target_link_libraries(test_viterbi ${LIBXML2_LIBRARIES} -lm)