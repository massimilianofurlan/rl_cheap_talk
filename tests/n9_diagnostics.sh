#!/usr/bin/env bash
  # Diagnostics for the n=9 "non-convergence-to-equilibrium" issue at b=0.1.
  # Run from the rl_cheap_talk/ directory:  bash scripts/run_n9_diagnostics.sh
  set -euo pipefail
  
  JL="julia --threads 8 --check-bounds=no"
  
  # ---- (a) does MORE exploration restore exact Nash? ------------------------
  # slower temperature decay (lambda 1.25e-6 -> 6.25e-7 -> 3.125e-7), more episodes.
  $JL main.jl -n 9 -b 0.1 -N 50 -c slow_exploration   -o out_9states_slowexpl
  $JL main.jl -n 9 -b 0.1 -N 50 -c slower_exploration -o out_9states_slowerexpl
  
  # ---- (b) variance control: hold Var[theta]=1/12 fixed while increasing n --
  # if the break still appears at constant variance, it is the resolution/#messages,
  # not the state variance.
  $JL main.jl -n 6  -b 0.1 -N 50 -c basecase --scale_states -o out_var_n6
  $JL main.jl -n 9  -b 0.1 -N 50 -c basecase --scale_states -o out_var_n9
  $JL main.jl -n 12 -b 0.1 -N 50 -c basecase --scale_states -o out_var_n12
