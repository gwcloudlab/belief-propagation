#!/usr/bin/env bash

ROOT_SRC_DIR=/home/mjt5v/CLionProjects/belief-propagation
OUTPUT_DIR="${ROOT_SRC_DIR}/cmake-build-release/openmp_num_threads"

for NUM_THREADS in 1 2 4 8
do
    rm -rf "${OUTPUT_DIR}/${NUM_THREADS}"
    export OMP_NUM_THREADS=${NUM_THREADS}
    echo "NUMBER OF OPENMP THREADS SET TO ${OMP_NUM_THREADS}"
    cd /home/mjt5v/CLionProjects/belief-propagation/
    cd "$ROOT_SRC_DIR/cmake-build-release/src/openmp_benchmark"
    echo "RUNNING EDGE BENCHMARK"
    ./openmp_edge_benchmark || true
    echo "RUNNING_NODE_BENCHMARK"
    ./openmp_node_benchmark || true
    mkdir -p "${OUTPUT_DIR}/${NUM_THREADS}"
    mv openmp_benchmark_loopy_edge.csv "${OUTPUT_DIR}/${NUM_THREADS}/openmp_benchmark_loopy_edge_${NUM_THREADS}.csv"
    mv openmp_benchmark_loopy_node.csv "${OUTPUT_DIR}/${NUM_THREADS}/openmp_benchmark_loopy_node_${NUM_THREADS}.csv"
done
