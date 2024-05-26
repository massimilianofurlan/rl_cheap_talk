#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00
#SBATCH --output=out/messages.txt

module purge
module load Julia/1.9.0-linux-x86_64

srun julia --threads 48 scripts/script1.jl -c basecase -o out_more_messages -n 6 -N 1000 -m 9
srun julia --threads 48 scripts/script1.jl -c basecase -o out_less_messages -n 6 -N 1000 -m 3
