for file in out_*
do
    # Run the Julia command with the current file
    julia --threads 8 scripts/generate_plots_1.jl -i "$file" --step_bias=0.005
done
