#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00
#SBATCH --output=out/gridsearch2.txt

module purge
module load Julia/1.9.0-linux-x86_64

srun julia --threads 48 scripts/script2.jl -c gridsearch -n 6 -o out_grid_search_new -N 1000
