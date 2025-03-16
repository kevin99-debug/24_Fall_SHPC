#!/bin/bash

commands=(
  "sbatch sbatch_files/zero1.slurm"
  "sbatch sbatch_files/zero2.slurm"
  "sbatch sbatch_files/zero2-offload.slurm"
  "sbatch sbatch_files/zero3.slurm"
  "sbatch sbatch_files/zero3-offload.slurm"
  "sbatch sbatch_files/zero3-fp16.slurm"
)

stage_select=$1
if [ -z "$1" ]; then
  echo "Usage: $0 <zero-stages>"
  exit 1
fi


if (( stage_select < 1 || stage_select > ${#commands[@]} )); then
  echo "Error: Wrong number"
  exit 1
fi

echo "${commands[$((stage_select - 1))]}"
eval "${commands[$((stage_select - 1))]}"
