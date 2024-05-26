#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00
#SBATCH --output=out/utilities.txt

module purge
module load Julia/1.9.0-linux-x86_64

srun julia --threads 48 scripts/script1.jl -c basecase -o out_fourthpower -n 6 -l fourth -N 1000
srun julia --threads 48 scripts/script1.jl -c basecase -o out_absolute -n 6 -l absolute -N 1000


