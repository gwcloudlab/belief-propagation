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
#SBATCH -J BeliefPropagationBenchmarks

module load cuda/toolkit
module load libxml2
module load cmake

# build and run c benchmarks
cd /home/trotsky/belief-propagation/src/c_benchmark
cmake . -DCMAKE_BUILD_TYPE=Release
make clean && make
rm *csv
./c_benchmark

# build and run openmp benchmarks
cd /home/trotsky/belief-propagation/src/openmp_benchmark
cmake . -DCMAKE_BUILD_TYPE=Release
make clean && make
rm *csv
./openmp_benchmark

# build and run openacc benchmarks
cd /home/trotsky/belief-propagation/src/openacc_benchmark
cmake . -DCMAKE_BUILD_TYPE=Release
make clean && make
rm *csv
./openacc_benchmark

# build and run cuda benchmarks
cd /home/trotsky/belief-propagation/src/cuda_benchmark
cmake . -DCMAKE_BUILD_TYPE=Release
make clean && make
rm *csv
./cuda_benchmark

# build and run cuda kernels benchmarks
cd /home/trotsky/belief-propagation/src/cuda_benchmark_kernels
cmake . -DCMAKE_BUILD_TYPE=Release
make clean && make
rm *csv
./cuda_kernels_benchmark
