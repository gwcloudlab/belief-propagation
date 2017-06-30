#!/usr/bin/env bash

for FILE in loopy_node_profile/*.sh
do
    sbatch "${FILE}"
done