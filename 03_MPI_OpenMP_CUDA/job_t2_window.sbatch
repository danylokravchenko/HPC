#!/bin/bash
#SBATCH --job-name=T2_window
#SBATCH --output=T2-window-result
#SBATCH --time=00:05:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000
#SBATCH --reservation=gpucourse
#SBATCH --account=s-gpu-course
# GPU settings
#SBATCH --partition=pascal
#SBATCH --gres=gpu:1

ml purge
ml CUDA/7.5.18
ml intel/2019a

srun --mpi=pmi2 ./T2_window.o 4096
