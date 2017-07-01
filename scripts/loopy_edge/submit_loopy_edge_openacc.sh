#!/bin/sh

#SBATCH -o loopy_edge_openacc_benchmarks%j.out
#SBATCH -e loopy_edge_openacc_benchmarks%j.err
# one hour timelimit
#SBATCH --time 7:00:00
# get gpu queue
#SBATCH -p gpu
# need 1 machine
#SBATCH -N 1
# name the job
#SBATCH -J LoopyEdgeOpenACCBeliefPropagationBenchmarks

module load cuda/toolkit
module load libxml2
module load cmake

# build and run openacc benchmarks
cd ${HOME}/belief-propagation/src/openacc_benchmark
cmake . -DCMAKE_BUILD_TYPE=Release
make clean && make
rm -f *csv
./openacc_loopy_edge_benchmark
