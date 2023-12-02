## Ready-to-download simulation outcomes

All simulation outcomes (and relevant figures) used in the paper can be downloaded [here](https://livewarwickac-my.sharepoint.com/:f:/g/personal/u2251842_live_warwick_ac_uk/EmOVFRKxgNtGnIkBFbecMkABrXrcSSEhbfX3CzGJ3Az9RQ?e=4FYv0h). 

The following section lists instructions to replicate each figure. 


## Replication instructions 

### Figures 1 - 3
To run the simulations navigate to the project directory (main branch) and run 

```julia --threads auto scripts/1_script.jl -c basecase -o out_basecase -N 1000```

To generate the figures run

```julia --threads auto scripts/1_plots.jl -i out_basecase ```


### Figures 4 - 5
To run the simulations navigate to the project directory (random-matching branch) and run 

```julia --threads auto scripts/1_script.jl -c basecase -o out_rand_match -N 100```

To generate the figures run

```julia --threads auto scripts/2_plots.jl -i out_rand_match ```


### Figures 6 - 7
To run the simulations navigate to the project directory (main branch) and run 

```julia --threads auto scripts/2_script.jl -c gridsearch -o out_grid_search -N 100```

To generate the figures run

```julia --threads auto scripts/2_plots.jl -i out_grid_search ```


### Figure 8
To run the simulations navigate to the project directory (main branch) and run 

```
julia --threads auto scripts/1_script.jl -c basecase -o out_6states -n 6 -N 1000
julia --threads auto scripts/1_script.jl -c basecase -o out_11states -n 11 -N 1000
julia --threads auto scripts/1_script.jl -c basecase -o out_41states -n 41 -N 1000
```

To generate the figures run

```
julia --threads auto scripts/1_plots.jl -i out_6states
julia --threads auto scripts/1_plots.jl -i out_11states
julia --threads auto scripts/1_plots.jl -i out_41states
```

### Figure 9
To run the simulations navigate to the project directory (main branch) and run 

```
julia --threads auto scripts/1_script.jl -c basecase -o out_fourthpower -l fourth -N 1000
julia --threads auto scripts/1_script.jl -c basecase -o out_absolute -l absolute -N 1000
julia --threads auto scripts/1_script.jl -c basecase -o out_scaled -k 10 -N 1000
```

To generate the figures run

```
julia --threads auto scripts/1_plots.jl -i out_fourthpower
julia --threads auto scripts/1_plots.jl -i out_absolute
julia --threads auto scripts/1_plots.jl -i out_scaled
```


### Figure 10

To run the simulations navigate to the project directory (main branch) and run 

```
julia --threads auto scripts/1_script.jl -c basecase -o out_binomial -d binomial -N 1000
julia --threads auto scripts/1_script.jl -c basecase -o out_increasing -d increasing -N 1000
julia --threads auto scripts/1_script.jl -c basecase -o out_decreasing -d decreasing -N 1000
```

To generate the figures run

```
julia --threads auto scripts/1_plots.jl -i out_binomial 
julia --threads auto scripts/1_plots.jl -i out_increasing 
julia --threads auto scripts/1_plots.jl -i out_decreasing
```
