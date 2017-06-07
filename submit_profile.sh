#!/bin/sh

#SBATCH -o benchmarks%j.out
#SBATCH -e benchmarks%j.err
# one hour timelimit
#SBATCH --time 7:00:00
# get gpu queue
#SBATCH -p gpu
# need 1 machine
#SBATCH -N 1
# name the job
#SBATCH -J BeliefPropagationProfile

module load cuda/toolkit
module load libxml2
module load cmake

# build and run openacc benchmarks
cd ${HOME}/belief-propagation/src/openacc_benchmark
cmake . -DCMAKE_BUILD_TYPE=Release
make clean && make
rm -f *csv
nvprof --analysis-metrics -o openacc.nvprof ./openacc_benchmark

# build and run cuda benchmarks
cd ${HOME}/belief-propagation/src/cuda_benchmark
cmake . -DCMAKE_BUILD_TYPE=Release
make clean && make
rm -f *csv
nvprof --analysis-metrics -o cuda.nvprof ./cuda_benchmark

# build and run cuda kernels benchmarks
cd ${HOME}/belief-propagation/src/cuda_benchmark_kernels
cmake . -DCMAKE_BUILD_TYPE=Release
make clean && make
rm -f *csv
nvprof --analysis-metrics -o cuda_kernels.nvprof ./cuda_kernels_benchmark