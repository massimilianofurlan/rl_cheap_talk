# builds the paper figures, at fixed cm sizes, into paper_figures/. reuses make_figures
# (make_figures.jl) for the per-directory figures; make_grid_figures (below) for the
# (alpha,lambda) grid. reads existing out_*/ data, run make_all_figures.sh if stale
# usage (from the rl_cheap_talk dir): julia --threads 8 scripts/paper_figures.jl [OUTDIR | -o OUTDIR]
#   e.g. -o ../revision/figures writes the tikz into the figures directory (default: paper_figures/)

using PGFPlotsX
using StatsBase
using Base.Threads
using Printf

include(joinpath(pwd(), "scripts/make_figures.jl"))   # provides make_figures (+ plots.jl, read_data.jl)
include(joinpath(pwd(), "scripts/grid_config.jl"))    # (alpha, lambda) grid (set_alpha / set_lambda)

# capture repo root now: generating pdfs (latex) changes the working dir, so pwd() is
# unreliable after the first figure. all paths below use ROOT
const ROOT = pwd()

# paper sizing: absolute cm, golden aspect, scale only axis.
# plots.jl reads these globals; plot_unit_scale=10 turns the build's fractional sizes into cm
ratio = (1 + sqrt(5)) / 2          # golden ratio: baseline panel 5.663cm x 3.5cm
plot_unit = "cm"
plot_unit_scale = 10
plot_scale_only_axis = true

# manuscript \linewidth = \textwidth = 6.5in = 16.51cm (11pt article, 1in margins, letterpaper).
# the policy_profiles_* figures are the only ones sized as fractions of \linewidth (plot_policy
# builds its own axis and ignores plot_unit), so pgfsave's standalone document renders them at
# the standalone \linewidth instead of the manuscript's. pin those fractions to absolute cm.
linewidth_cm = 16.51

# line styling for the paper figures (make_figures uses lighter defaults)
lw_value = "2.5pt"             # benchmark + grey equilibria lines
lw_modal = "1.3pt"             # modal outcome (blue) line
jump_value = "jump mark right"  # value lines step right (modal lines stay left, set in make_figures)

# build the two paper grid figures from input_dir (out_grid_search); grids are paper-only
#   grid_pmv_*  : per-panel size of the posterior-mean-variance 3x3 grid
#   grid_nash_* : per-panel size of the frequency-of-nash 5x5 grid
#   names       : nothing -> save both; Dict -> save renamed subset
function make_grid_figures(input_dir; step_bias = 0.01f0,
						grid_pmv_w = 0.375, grid_pmv_h = 0.375 * ratio^-1,
						grid_nash_w = 0.27, grid_nash_h = 0.27 * ratio^-1,
						save_dir = joinpath(input_dir, "tikz"),
						pdf_dir = joinpath(input_dir, "pdf"),
						names = nothing)

global set_biases = collect(0.00f0:step_bias:0.5f0)
n_biases, n_simulations = length(set_biases), 1000

posterior_mean_variance = fill(NaN, n_alpha, n_lambda, n_simulations, n_biases);
expected_reward_s = fill(NaN, n_alpha, n_lambda, n_simulations, n_biases);
expected_reward_r = fill(NaN, n_alpha, n_lambda, n_simulations, n_biases);
max_max_mass_on_suboptim = fill(NaN, n_alpha, n_lambda, n_simulations, n_biases);
max_absolute_error = fill(NaN, n_alpha, n_lambda, n_simulations, n_biases);
is_nash = fill(NaN, n_alpha, n_lambda, n_simulations, n_biases);
n_episodes = fill(NaN, n_alpha, n_lambda, n_simulations, n_biases);
is_converged = fill(NaN, n_alpha, n_lambda, n_simulations, n_biases);
ranks = fill(NaN, n_alpha, n_lambda, n_simulations, n_biases);
babbling_reward_r = fill(NaN, n_alpha, n_lambda, n_biases);

for dir in readdir(input_dir, join = true)
	isdir(dir) || continue
	basename(dir) in ("pdf", "temp", "tikz") && continue   # skip output folders from earlier runs
	println("\nCurrent dir: ", basename(dir))
	config, extracted_data = read_data(dir)
	config != nothing || continue # 
	alpha::Float32, expl_decay::Float32 = config["alpha_s"], config["expl_decay_s"]
	alpha_idx, lambda_idx = findfirst(set_alpha .== alpha), findfirst(set_lambda .== expl_decay)

	expected_reward_s[alpha_idx,lambda_idx,:,:] .= cat(extracted_data["expected_reward_s"]...,dims=2);
	expected_reward_r[alpha_idx,lambda_idx,:,:] .= cat(extracted_data["expected_reward_r"]...,dims=2);
	posterior_mean_variance[alpha_idx,lambda_idx,:,:] .= cat(extracted_data["posterior_mean_variance"]...,dims=2);
	max_max_mass_on_suboptim[alpha_idx,lambda_idx,:,:] .= cat(extracted_data["max_max_mass_on_suboptim"]...,dims=2);
	max_absolute_error[alpha_idx,lambda_idx,:,:] .= cat(extracted_data["max_absolute_error"]...,dims=2);
	is_nash[alpha_idx,lambda_idx,:,:] .= cat(extracted_data["max_max_mass_on_suboptim"]...,dims=2) .< 1f-2 
	n_episodes[alpha_idx,lambda_idx,:,:] .= cat(extracted_data["n_episodes"]...,dims=2);
	is_converged[alpha_idx,lambda_idx,:,:] .= cat(extracted_data["is_converged"]...,dims=2);
	babbling_reward_r[alpha_idx,lambda_idx,:] .= extracted_data["babbling_reward_r"]
end

mean_(x; dims=:) = dropdims(mean(x,dims=dims),dims=dims)
avg_expected_reward_s = mean_(expected_reward_s,dims=3)
avg_expected_reward_r = mean_(expected_reward_r,dims=3)
avg_posterior_mean_variance = mean_(posterior_mean_variance,dims=3)
avg_max_max_mass_on_suboptim = mean_(max_max_mass_on_suboptim,dims=3)
avg_max_absolute_error = mean_(max_absolute_error,dims=3)
avg_is_nash = mean_(is_nash,dims=3)
avg_n_episodes = mean_(n_episodes,dims=3)
avg_is_converged = mean_(is_converged,dims=3)

set_nash, best_nash = get_equilibria(1001)

format_scientific(x) = string(round(x / 10.0^floor(Int, log10(abs(x))), digits=2), " \\times 10^{", floor(Int, log10(abs(x))), "}")

pls_expected_reward_r = Axis[]
pls_posterior_mean_variance = Axis[]
pls_is_gamma_nash = Axis[]
for alpha_idx in 5:-1:1, lambda_idx in 1:1:5
	ylabel_ = lambda_idx == 1 ? string(raw"$",set_alpha[alpha_idx],raw"$") : ""
	xlabel_ = alpha_idx == 1 ? string(raw"$",format_scientific(set_lambda[lambda_idx]),raw"$") : ""
	ylabel_ = lambda_idx == 1 && alpha_idx == 3 ? string(raw"$\alpha$ \quad $",set_alpha[alpha_idx],raw"$") : ylabel_
	xlabel_ = alpha_idx == 1 && lambda_idx == 3 ? string(raw"$",format_scientific(set_lambda[lambda_idx]),raw"$ \\[5pt] $\lambda$") : xlabel_
	axis_style = lambda_idx == 3 && alpha_idx == 3 ? "axis line style={line width=0.75pt}" : ""

	# POSTERIOR MEAN VARIANCE
	pl_posterior_mean_variance = plot_dist(posterior_mean_variance[alpha_idx,lambda_idx,:,:];
									ylabel = ylabel_, xlabel = xlabel_,
									color = "blue",
									ymin=0, ymax=1, n_steps=65,
									additional = "ticks=none",
									width = grid_pmv_w, height = grid_pmv_h);
	pl_posterior_mean_variance = plot_interpolated_val!(pl_posterior_mean_variance, best_nash["posterior_mean_variance"]; color = "red", style = "solid, line width=2.5pt", opacity = 0.4, ymin=0, ymax=1, n_steps=65);
	babbling_posterior_mean_variance = fill(best_nash["posterior_mean_variance"][end],length(set_biases))
	pl_posterior_mean_variance = plot_val!(pl_posterior_mean_variance, babbling_posterior_mean_variance; color = "darkgray", style = "dotted");
	pl_posterior_mean_variance = plot_eq_bound!(pl_posterior_mean_variance,best_nash["posterior_mean_variance"]);

	# IS NASH
	pl_is_gamma_nash = plot_avg(is_nash[alpha_idx,lambda_idx,:,:];
									ylabel = ylabel_, xlabel = xlabel_,
									color = "green!80!black",
									ymin=0,
									ci_flag = false,
									additional = string("ticks=none, ",axis_style),
									width = grid_nash_w, height = grid_nash_h);
	pl_is_gamma_nash = plot_eq_bound!(pl_is_gamma_nash,best_nash["posterior_mean_variance"]);
	@pgf push!(pl_is_gamma_nash, HLine({"color = gray", "style = dashed, very thin", "on layer = axis background"}, 0))
	@pgf push!(pl_is_gamma_nash, HLine({"color = gray", "style = dashed, very thin", "on layer = axis background"}, 1))

	push!(pls_posterior_mean_variance, pl_posterior_mean_variance)
	push!(pls_is_gamma_nash, pl_is_gamma_nash)
end

pl_grid_posterior_mean_variance = @pgf GroupPlot(
							{ group_style = { group_size="3 by 3", raw"horizontal sep = 9.708pt", raw"vertical sep = 6pt" },
   							 }, pls_posterior_mean_variance[[1,3,5,11,13,15,21,23,25]]...);

pl_grid_frequency_nash = @pgf GroupPlot(
							{ group_style = { group_size="5 by 5", raw"horizontal sep = 8.09pt", raw"vertical sep = 5pt" },
   							 }, pls_is_gamma_nash...);

# save plots
tikz_dir = mkpath(save_dir)
pdf_dir = mkpath(pdf_dir)

save_plot(file_name, plot) = (pgfsave(joinpath(tikz_dir, "$file_name.tikz"), plot); pgfsave(joinpath(pdf_dir, "$file_name.pdf"), plot))

plots = [
    	("grid_posterior_mean_variance", pl_grid_posterior_mean_variance),
       	("grid_frequency_nash", pl_grid_frequency_nash),
	]

# when `names` is given, keep only those figures, renamed
names === nothing || (plots = [(names[k], p) for (k, p) in plots if haskey(names, k)])

counter = Atomic{Int}(0)
total_plots = length(plots)

Threads.@threads for (file_name, plot) in plots
    save_plot(file_name, plot)
    atomic_add!(counter, 1)
    print("\rGenerating plots $(counter[])/$total_plots")
    flush(stdout)
end
println()
return nothing

end  # make_grid_figures

const STEP_BIAS  = 0.005f0                                            # baseline + robustness

# output dir from args: -o/--outdir OUTDIR, -o=OUTDIR, or bare positional (default paper_figures/)
# relative paths resolve against ROOT (latex moves the working dir during pdf generation)
function parse_outdir(args, default)
	out = default
	skip = false
	for (k, a) in enumerate(args)
		if skip
			skip = false
		elseif (a == "-o" || a == "--outdir") && k < length(args)
			out = args[k+1]; skip = true
		elseif startswith(a, "-o=") || startswith(a, "--outdir=")
			out = split(a, "=", limit = 2)[2]
		elseif !startswith(a, "-")
			out = a
		end
	end
	return isabspath(out) ? out : abspath(joinpath(ROOT, out))
end

const FIG_DIR    = parse_outdir(ARGS, "paper_figures")               # tikz + pdf output
const PDF_DIR    = joinpath(FIG_DIR, "pdf")

# heights (fractional unit; x10 -> cm)
const H_MAIN  = 0.35         # 3.5000 cm  (baseline expected_rewards, posterior_mean_variance_modal)
const H_ROB   = 0.2625       # 2.6250 cm  (robustness expected_rewards panels)
const H_ERROR = 0.35 * ratio / 2   # 2.8316 cm  (optimization_errors 2x2 panels; = width/2)

# the single parameter sub-directory inside an out_* directory (what read_data expects)
function data_subdir(out_dir)
	subs = filter(isdir, readdir(out_dir, join = true))
	subs = setdiff(subs, joinpath.(out_dir, ["pdf", "temp", "tikz"]))
	length(subs) == 1 || @warn "expected one parameter sub-dir in $out_dir, found $(length(subs)); using the first" subs
	return first(subs)
end

mkpath(FIG_DIR); mkpath(PDF_DIR)

# ---- baseline figures (from out_basecase) ----
println("\n[paper figures] baseline (out_basecase)")
make_figures(data_subdir(joinpath(ROOT, "out_basecase"));
	step_bias = STEP_BIAS, main_height = H_MAIN, error_height = H_ERROR, expected_hsep = "72pt",
	# tighter layout for policy_profiles_learned_vs_equilibria
	cmp_panel = "$(round(0.20 * linewidth_cm, digits=4))cm", cmp_gap = "50pt", cmp_vsep = "8pt", cmp_tickfont = raw"\scriptsize",
	policy_panel = "$(round(0.25 * linewidth_cm, digits=4))cm",
	save_dir = FIG_DIR, pdf_dir = PDF_DIR,
	names = Dict(
		"expected_rewards"                      => "expected_rewards",
		"posterior_mean_variance_modal"         => "posterior_mean_variance_modal",
		"optimization_errors"                   => "optimization_errors",
		"policy_profiles_learned_sample"        => "policy_profiles_learned_sample",
		"policy_profiles_learned_vs_equilibria" => "policy_profiles_learned_vs_equilibria",
	))

# ---- robustness figures: out_* directory => paper figure name ----
robustness = [
	("out_3states",       "expected_rewards_3states"),
	("out_9states",       "expected_rewards_9states"),
	("out_less_messages", "expected_rewards_3messages"),
	("out_more_messages", "expected_rewards_9messages"),
	("out_less_actions",  "expected_rewards_9actions"),
	("out_more_actions",  "expected_rewards_21actions"),
	("out_power32",       "expected_rewards_power32"),
	("out_fourthpower",   "expected_rewards_fourth"),
	("out_increasing",    "expected_rewards_increasing"),
	("out_decreasing",    "expected_rewards_decreasing"),
]
for (out, paper_name) in robustness
	println("\n[paper figures] robustness ($out -> $paper_name)")
	make_figures(data_subdir(joinpath(ROOT, out));
		step_bias = STEP_BIAS, main_height = H_ROB, expected_hsep = "72pt",
		save_dir = FIG_DIR, pdf_dir = PDF_DIR,
		names = Dict("expected_rewards" => paper_name))
end

# ---- grids (from out_grid_search) ----
# grid panel sizes: 3x3 panel 4.3147cm x 2.6667cm ; 5x5 panel 2.5cm x 1.25cm
println("\n[paper figures] grids (out_grid_search)")
make_grid_figures(joinpath(ROOT, "out_grid_search");
	grid_pmv_w = 0.43147, grid_pmv_h = 0.26667,
	grid_nash_w = 0.25,   grid_nash_h = 0.125,
	save_dir = FIG_DIR, pdf_dir = PDF_DIR,
	names = Dict(
		"grid_posterior_mean_variance" => "grid_posterior_mean_variance",
		"grid_frequency_nash"          => "grid_frequency_nash",
	))

println("\n[paper figures] done -> $FIG_DIR")
