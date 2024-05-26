## Ready-to-download simulation outcomes

All simulation outcomes (and relevant figures) used in the paper can be downloaded [here](https://livewarwickac-my.sharepoint.com/:f:/g/personal/u2251842_live_warwick_ac_uk/EmOVFRKxgNtGnIkBFbecMkABrXrcSSEhbfX3CzGJ3Az9RQ?e=4FYv0h). 

The following section lists instructions to replicate each figure. 


## Replication instructions 

### Figures 1, 3, 4, 5 and 6 
To run the simulations navigate to the project directory and run 

```julia --threads auto scripts/script1.jl -c basecase -o out_basecase -n 6 -N 1000 --step_bias 0.005```

To generate the figures run

```julia --threads auto scripts/generate_plots_1.jl -i out_basecase --step_bias 0.005```


### Figures 2 and 7
To run the simulations navigate to the project directory and run 

```julia --threads auto scripts/script2.jl -c gridsearch -o out_grid_search -N 1000```

To generate the figures run

```julia --threads auto scripts/generate_plots_2.jl -i out_grid_search ```


### Figure 8
To run the simulations navigate to the project directory and run 

```
julia --threads auto scripts/script1.jl -c basecase -o out_more_messages -n 6 -N 1000 --step_bias 0.005 -m 9
julia --threads auto scripts/script1.jl -c basecase -o out_less_messages -n 6 -N 1000 --step_bias 0.005 -m 3
```

To generate the figures run

```
julia --threads auto scripts/generate_plots_1.jl -i out_more_messages --step_bias 0.005
julia --threads auto scripts/generate_plots_1.jl -i out_less_messages --step_bias 0.005
```

### Figure 9
To run the simulations navigate to the project directory and run 

```
julia --threads auto scripts/script1.jl -c basecase -o out_more_actions -n 6 -N 1000 --step_bias 0.005 -a 21
julia --threads auto scripts/script1.jl -c basecase -o out_less_actions -n 6 -N 1000 --step_bias 0.005 -a 9
```

To generate the figures run

```
julia --threads auto scripts/generate_plots_1.jl -i out_more_actions --step_bias 0.005
julia --threads auto scripts/generate_plots_1.jl -i out_less_actions --step_bias 0.005
```


### Figure 10

To run the simulations navigate to the project directory and run 

```
julia --threads auto scripts/script1.jl -c basecase -o out_3states -n 3 -N 1000 --step_bias 0.005
julia --threads auto scripts/script1.jl -c basecase -o out_9states -n 9 -N 1000 --step_bias 0.005
```

To generate the figures run

```
julia --threads auto scripts/generate_plots_1.jl -i out_3states --step_bias 0.005
julia --threads auto scripts/generate_plots_1.jl -i out_9states --step_bias 0.005
```


### Figure 11

To run the simulations navigate to the project directory and run 

```
julia --threads auto scripts/script1.jl -c basecase -o out_fourthpower -n 6 -l fourth -N 1000 --step_bias 0.005
julia --threads auto scripts/script1.jl -c basecase -o out_absolute -n 6 -l absolute -N 1000 --step_bias 0.005
```

To generate the figures run

```
julia --threads auto scripts/generate_plots_1.jl -i out_fourthpower --step_bias 0.005
julia --threads auto scripts/generate_plots_1.jl -i out_absolute --step_bias 0.005
```


### Figure 12

To run the simulations navigate to the project directory and run 

```
julia --threads auto scripts/script1.jl -c basecase -o out_increasing -n 6 -a 21 -d increasing -N 1000 --step_bias 0.005
julia --threads auto scripts/script1.jl -c basecase -o out_decreasing -n 6 -a 21 -d decreasing -N 1000 --step_bias 0.005
```

To generate the figures run

```
julia --threads auto scripts/generate_plots_1.jl -i out_increasing --step_bias 0.005
julia --threads auto scripts/generate_plots_1.jl -i out_decreasing --step_bias 0.005
```
