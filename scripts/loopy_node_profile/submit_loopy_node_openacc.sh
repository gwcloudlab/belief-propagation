#!/bin/sh

#SBATCH -o loopy_node_openacc_benchmarks%j.out
#SBATCH -e loopy_node_openacc_benchmarks%j.err
# one hour timelimit
#SBATCH --time 7:00:00
# get gpu queue
#SBATCH -p gpu
# need 1 machine
#SBATCH -N 1
# name the job
#SBATCH -J LoopyNodeOpenACCBeliefPropagationBenchmarks

module load cuda/toolkit
module load libxml2
module load cmake

# build and run openacc benchmarks
cd ${HOME}/belief-propagation/src/openacc_benchmark
cmake . -DCMAKE_BUILD_TYPE=Release
make clean && make
rm -f *csv
nvprof --analysis-metrics -o openacc_loopy_node.nvprof ./openacc_loopy_node_benchmark
