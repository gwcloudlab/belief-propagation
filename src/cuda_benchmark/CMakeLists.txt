project(cuda_benchmark)

cmake_minimum_required(VERSION 2.8)

find_package(LibXml2 REQUIRED)
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5)
    set(CUDA_HOST_COMPILER "gcc-5")
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo -gencode arch=compute_61,code=sm_61)


cuda_add_executable(cuda_node_benchmark belief-propagation-node.cu belief-propagation.cu ../bnf-xml-parser/bnf-xml-wrapper.c ../bnf-xml-parser/xml-expression.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/expression.c ../bnf-parser/Lexer.c ../bnf-parser/Parser.c ../graph/graph.c ../snap-parser/snap-parser.c  ../csr-parser/csr-parser.c)
cuda_add_executable(cuda_node_streaming_benchmark belief-propagation-node-streaming.cu belief-propagation.cu ../bnf-xml-parser/bnf-xml-wrapper.c ../bnf-xml-parser/xml-expression.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/expression.c ../bnf-parser/Lexer.c ../bnf-parser/Parser.c ../graph/graph.c ../snap-parser/snap-parser.c  ../csr-parser/csr-parser.c)
cuda_add_executable(cuda_edge_benchmark belief-propagation-edge.cu belief-propagation.cu ../bnf-xml-parser/bnf-xml-wrapper.c ../bnf-xml-parser/xml-expression.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/expression.c ../bnf-parser/Lexer.c ../bnf-parser/Parser.c ../graph/graph.c ../snap-parser/snap-parser.c  ../csr-parser/csr-parser.c)
cuda_add_executable(cuda_edge_streaming_benchmark belief-propagation-edge-streaming.cu belief-propagation.cu ../bnf-xml-parser/bnf-xml-wrapper.c ../bnf-xml-parser/xml-expression.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/expression.c ../bnf-parser/Lexer.c ../bnf-parser/Parser.c ../graph/graph.c ../snap-parser/snap-parser.c  ../csr-parser/csr-parser.c)
include_directories(${LIBXML2_INCLUDE_DIR})
target_link_libraries(cuda_node_benchmark ${LIBXML2_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} m)
target_link_libraries(cuda_node_streaming_benchmark ${LIBXML2_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} m)
target_link_libraries(cuda_edge_benchmark ${LIBXML2_LIBRARIES} m)
target_link_libraries(cuda_edge_streaming_benchmark ${LIBXML2_LIBRARIES} m)