project(cuda_kernels_benchmark)

cmake_minimum_required(VERSION 2.8)

find_package(LibXml2 REQUIRED)
find_package(CUDA REQUIRED)

if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5)
    set(CUDA_HOST_COMPILER "gcc-5")
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -gencode arch=compute_61,code=sm_61 -Xptxas -dlcm=cg -use_fast_math")


cuda_add_executable(cuda_kernels_benchmark belief-propagation-kernels-benchmark.cu belief-propagation-kernels.cu ../bnf-xml-parser/bnf-xml-wrapper.c ../bnf-xml-parser/xml-expression.c ../bnf-parser/bnf-wrapper.c ../bnf-parser/expression.c ../bnf-parser/Lexer.c ../bnf-parser/Parser.c ../graph/graph.c ../snap-parser/snap-parser.c  ../csr-parser/csr-parser.c)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${LIBXML2_INCLUDE_DIR})
target_link_libraries(cuda_kernels_benchmark ${LIBXML2_LIBRARIES} m)