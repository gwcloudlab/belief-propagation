#!/usr/bin/env bash

for FILE in loopy_edge_profile/*.sh
do
    sbatch "${FILE}"
done