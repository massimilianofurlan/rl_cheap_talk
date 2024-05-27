#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00
#SBATCH --output=out/basecase.txt

module purge
module load Julia/1.9.0-linux-x86_64

srun julia --threads 48 --check-bounds no scripts/script1.jl -c basecase -n 6 -N 1000 --step_bias=0.005 -o out_basecase

