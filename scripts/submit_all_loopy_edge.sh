#!/usr/bin/env bash

for FILE in loopy_edge/*.sh
do
    sbatch "${FILE}"
done