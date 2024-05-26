#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00
#SBATCH --output=out/states.txt

module purge
module load Julia/1.9.0-linux-x86_64

srun julia --threads 48 scripts/scrip1t.jl -c basecase -o out_3states -n 3 -N 1000
srun julia --threads 48 scripts/script1.jl -c basecase -o out_9states -n 9 -N 1000 


