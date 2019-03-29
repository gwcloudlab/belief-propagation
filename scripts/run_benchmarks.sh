#!/bin/sh

ROOT_SRC_DIR=/home/mjt5v/CLionProjects/belief-propagation
echo "Starting scripts at $(date)"
# c benchmark
cd "$ROOT_SRC_DIR/cmake-build-release/src/c_benchmark/"
echo "Running C Loopy Edge: $(date)"
./c_loopy_edge_benchmark || true
echo "Running C Loopy Node $(date)"
./c_loopy_node_benchmark || true
echo "Running C Non Loopy $(date)"
./c_non_loopy_benchmark || true

# openmp benchmark
cd "$ROOT_SRC_DIR/cmake-build-release/src/openmp_benchmark"
echo "Running OpenMP Edge $(date)"
./openmp_edge_benchmark || true
echo "Running OpenMP Node $(date)"
./openmp_node_benchmark || true

# openacc benchmark
cd "$ROOT_SRC_DIR/cmake-build-release/src/openacc_benchmark"
# make clean && make
echo "Running OpenACC Edge $(date)"
./openacc_loopy_edge_benchmark || true
echo "Running OpenACC Node $(date)"
./openacc_loopy_node_benchmark || true

# cuda benchmark
cd "$ROOT_SRC_DIR/cmake-build-release/src/cuda_benchmark"
echo "Running CUDA Edge $(date)"
./cuda_edge_benchmark || true
echo "Running CUDA Edge Streaming $(date)"
./cuda_edge_streaming_benchmark || true
#./cuda_edge_openmpi_benchmark || true
echo "Running CUDA Node $(date)"
./cuda_node_benchmark || true
echo "Running CUDA Node Streaming $(date)"
./cuda_node_streaming_benchmark || true
#./cuda_node_openmpi_benchmark || true

# cuda-kernels benchmark
cd "$ROOT_SRC_DIR/cmake-build-release/src/cuda_benchmark_kernels"
echo "Running CUDA Kernels $(date)"
./cuda_kernels_benchmark || true

echo "Done $(date)"

