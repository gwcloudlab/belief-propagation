#!/bin/sh

#SBATCH -o loopy_node_cuda_kernels_benchmarks%j.out
#SBATCH -e loopy_node_cuda_kernels_benchmarks%j.err
# one hour timelimit
#SBATCH --time 7:00:00
# get gpu queue
#SBATCH -p gpu
# need 1 machine
#SBATCH -N 1
# name the job
#SBATCH -J LoopyNodeCudaKernelsBeliefPropagationBenchmarks

module load cuda/toolkit
module load libxml2
module load cmake

# build and run cuda benchmarks
cd ${HOME}/belief-propagation/src/cuda_benchmark_kernels
cmake . -DCMAKE_BUILD_TYPE=Release
make clean && make
rm -f *csv
nvprof --analysis-metrics -o cuda_kernels.nvprof ./cuda_kernels_benchmark