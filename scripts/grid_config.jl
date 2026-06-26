# (alpha, lambda) hyperparameter grid, shared by run_grid.jl (writes them to config.toml) and
# paper_figures.jl (looks them up when reading results). alpha doubles, lambda halves
const min_alpha  = 0.025f0
const n_alpha    = 5
const min_lambda = 0.00002f0
const n_lambda   = 5
const set_alpha::Vector{Float32}  = [min_alpha  * 2^(i-1) for i in 1:n_alpha]
const set_lambda::Vector{Float32} = [min_lambda / 2^(i-1) for i in 1:n_lambda]
