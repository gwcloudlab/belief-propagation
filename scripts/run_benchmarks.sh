#!/bin/sh

ROOT_SRC_DIR=/home/mjt5v/CLionProjects/belief-propagation

# c benchmark
cd "$ROOT_SRC_DIR/cmake-build-release/src/c_benchmark/"
./c_loopy_edge_benchmark || true
./c_loopy_node_benchmark || true
./c_non_loopy_benchmark || true

# openmp benchmark
cd "$ROOT_SRC_DIR/cmake-build-release/src/openmp_benchmark"
./openmp_edge_benchmark || true
./openmp_node_benchmark || true

# openacc benchmark
cd "$ROOT_SRC_DIR/cmake-build-release/src/openacc_benchmark"
# make clean && make
./openacc_loopy_edge_benchmark || true
./openacc_loopy_node_benchmark || true

# cuda benchmark
cd "$ROOT_SRC_DIR/cmake-build-release/src/cuda_benchmark"
./cuda_edge_benchmark || true
./cuda_edge_streaming_benchmark || true
#./cuda_edge_openmpi_benchmark || true
./cuda_node_benchmark || true
./cuda_node_streaming_benchmark || true
#./cuda_node_openmpi_benchmark || true

# cuda-kernels benchmark
cd "$ROOT_SRC_DIR/cmake-build-release/src/cuda_benchmark_kernels"
./cuda_kernels_benchmark || true
