# build all figures for every out_* directory (the grid figures are built by paper_figures.jl)
for file in out_*
do
    [ "$file" = "out_grid_search" ] && continue
    julia --threads 8 scripts/make_figures.jl -i "$file" --step_bias=0.005
done
