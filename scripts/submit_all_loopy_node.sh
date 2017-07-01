#!/usr/bin/env bash

for FILE in loopy_node/*.sh
do
    sbatch "${FILE}"
done