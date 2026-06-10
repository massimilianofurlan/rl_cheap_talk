for file in out_*
do
    [ "$file" = "out_grid_search" ] && continue
    # Run the Julia command with the current file
    julia --threads 8 scripts/generate_plots_1.jl -i "$file" --step_bias=0.005
done

# grid search (Figures 2 and 7) uses generate_plots_2
julia --threads 8 scripts/generate_plots_2.jl -i out_grid_search
