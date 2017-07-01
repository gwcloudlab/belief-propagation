#!/bin/sh

#SBATCH -o non_loopy_benchmark%j.out
#SBATCH -e non_loopy_benchmark%j.err
# one hour timelimit
#SBATCH --time 14:00:00
# get gpu queue
#SBATCH -p 128gb
# need 1 machine
#SBATCH -N 1
# name the job
#SBATCH -J NonLoopyBeliefPropagationBenchmark

module load libxml2
module load cmake

# build and run c benchmarks
cd ${HOME}/belief-propagation/src/c_benchmark
cmake . -DCMAKE_BUILD_TYPE=Release
make clean && make
rm -f *csv
./c_non_loopy_benchmark
