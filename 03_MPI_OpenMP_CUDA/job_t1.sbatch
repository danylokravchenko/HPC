#!/bin/bash
#SBATCH --job-name=T1
#SBATCH --output=T1-result
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --reservation=gpucourse
#SBATCH --account=s-gpu-course
# GPU settings
#SBATCH --partition=pascal
#SBATCH --gres=gpu:1

ml CUDA/7.5.18

srun T1.o 1024
srun T1.o 2048
srun T1.o 4096
