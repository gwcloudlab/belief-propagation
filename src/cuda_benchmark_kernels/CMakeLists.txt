project(cuda_kernels_benchmark)

cmake_minimum_required(VERSION 2.8)

find_package(LibXml2 REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_HOST_COMPILER "gcc-5")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo -gencode arch=compute_35,code=sm_35)


cuda_add_executable(cuda_kernels_benchmark belief-propagation-kernels-benchmark.cu belief-propagation-kernels.cu ../bnf-xml-parser/bnf-xml-wrapper.c ../bnf-xml-parser/xml-expression.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/expression.c ../bnf-parser/Lexer.c ../bnf-parser/Parser.c ../graph/graph.c)
include_directories(${LIBXML2_INCLUDE_DIR})
target_link_libraries(cuda_kernels_benchmark ${LIBXML2_LIBRARIES})