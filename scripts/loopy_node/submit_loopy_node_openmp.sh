#!/bin/sh

#SBATCH -o loopy_node_openmp_benchmark%j.out
#SBATCH -e loopy_node_openmp_benchmark%j.err
# one hour timelimit
#SBATCH --time 14:00:00
# get gpu queue
#SBATCH -p 128gb
# need 1 machine
#SBATCH -N 1
# name the job
#SBATCH -J LoopyNodeOpenMPBeliefPropagationBenchmarks

module load libxml2
module load cmake

# build and run c benchmarks
cd ${HOME}/belief-propagation/src/openmp_benchmark
cmake . -DCMAKE_BUILD_TYPE=Release
make clean && make
rm -f *csv
./openmp_node_benchmark
