#!/bin/bash

#SBATCH --job-name=zero3
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=class1
#SBATCH --mem=200G
#SBATCH -o sbatch_output/%x-sbatch.out

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9723
export OMP_NUM_THREADS=4

srun --jobid $SLURM_JOBID bash -c 'torchrun \
 --nproc-per-node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
deepspeed_training.py \
--stage 3'
