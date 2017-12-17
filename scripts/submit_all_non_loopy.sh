#!/usr/bin/env bash

for FILE in non_loopy/*.sh
do
    sbatch "${FILE}"
done