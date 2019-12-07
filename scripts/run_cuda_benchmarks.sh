#!/bin/sh

ROOT_SRC_DIR=/home/mjt5v/CLionProjects/belief-propagation

# cuda benchmark
cd "$ROOT_SRC_DIR/cmake-build-release/src/cuda_benchmark"
./cuda_edge_benchmark || true
# ./cuda_edge_streaming_benchmark || true
# ./cuda_edge_openmpi_benchmark || true
./cuda_node_benchmark || true
#./cuda_node_streaming_benchmark || true
#./cuda_node_openmpi_benchmark || true

# cuda-kernels benchmark
#cd "$ROOT_SRC_DIR/cmake-build-release/src/cuda_benchmark_kernels"
#./cuda_kernels_benchmark || true
