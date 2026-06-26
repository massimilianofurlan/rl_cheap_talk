# builds all figures for one output directory (from run_experiment.jl)
# usage: navigate to 'rl_cheap_talk', run
#   julia --threads NUM_THREADS scripts/make_figures.jl -i INPUT_DIR

using PGFPlotsX
using StatsBase
using Base.Threads
using Printf

include(joinpath(pwd(),"scripts/plots.jl"))
include(joinpath(pwd(),"scripts/read_data.jl"))

# preamble and default aspect ratio. `ratio` is a module global (plots.jl reads it);
# paper_figures.jl overrides it and the sizing globals for the paper figures
push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepgfplotslibrary{fillbetween}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usetikzlibrary{calc}")
ratio = 4/3

# build and save every figure for one parameter sub-dir `data_dir` (what read_data expects)
#   main_height / error_height : axis heights for the main line plots / the 2x2 error panels
#   save_dir / pdf_dir         : where to write the .tikz / .pdf
#   names                      : nothing -> save all under their own names; Dict -> renamed subset
# layout kwargs (cmp_*, expected_hsep): paper_figures.jl passes tighter values for the paper
function make_figures(data_dir; step_bias = 0.01f0,
						main_height = 0.35, error_height = 0.35 * ratio^-1,
						cmp_panel = raw"0.25\linewidth", cmp_gap = "50pt",
						cmp_vsep = "16pt", cmp_tickfont = raw"\normalsize",
						expected_hsep = "60pt",
						save_dir = joinpath(dirname(data_dir), "tikz"),
						pdf_dir = joinpath(dirname(data_dir), "pdf"),
						names = nothing)

# set of biases (global: read_data and helpers read it from module scope)
global set_biases = collect(0.00f0:step_bias:0.5f0)
# spec id from the out_* dir name (get_title_equation reads script_config["in_dir"])
global script_config = Dict("in_dir" => basename(dirname(data_dir)))

# read data and compute equilibria
_, extracted_data = read_data(data_dir)
n_biases_nash = 1001
set_nash, best_nash = get_equilibria(n_biases_nash)

# unpacking data
policy_s = cat(extracted_data["policy_s"]...,dims=4);
policy_r = cat(extracted_data["policy_r"]...,dims=4);
induced_actions = cat(extracted_data["induced_actions"]...,dims=4);
n_episodes = cat(extracted_data["n_episodes"]...,dims=2);
is_converged = cat(extracted_data["is_converged"]...,dims=2);
expected_reward_s = cat(extracted_data["expected_reward_s"]...,dims=2);
expected_reward_r = cat(extracted_data["expected_reward_r"]...,dims=2);
posterior_mean_variance = cat(extracted_data["posterior_mean_variance"]...,dims=2);
absolute_error_s = cat(extracted_data["absolute_error_s"]...,dims=2);
absolute_error_r = cat(extracted_data["absolute_error_r"]...,dims=2);
max_absolute_error = cat(extracted_data["max_absolute_error"]...,dims=2);
n_on_path_messages = cat(extracted_data["n_on_path_messages"]...,dims=2);
max_mass_on_suboptim_s = cat(extracted_data["max_mass_on_suboptim_s"]...,dims=2);
max_mass_on_suboptim_r = cat(extracted_data["max_mass_on_suboptim_r"]...,dims=2);
max_max_mass_on_suboptim = cat(extracted_data["max_max_mass_on_suboptim"]...,dims=2);
is_partitional = cat(extracted_data["is_partitional"]...,dims=2);
n_effective_messages = cat(extracted_data["n_effective_messages"]...,dims=2);

babbling_reward_s = extracted_data["babbling_reward_s"];
babbling_reward_r = extracted_data["babbling_reward_r"];
expected_reward_s_best = best_nash["expected_reward_s"];
expected_reward_r_best = best_nash["expected_reward_r"];
posterior_mean_variance_best = best_nash["posterior_mean_variance"];
expected_reward_s_nash = set_nash["expected_reward_s"];
expected_reward_r_nash = set_nash["expected_reward_r"];
posterior_mean_variance_nash = set_nash["posterior_mean_variance"];
induced_actions_nash = set_nash["induced_actions"];

n_simulations, n_biases = size(n_episodes)
set_biases_nash = range(0.0f0,0.5f0,n_biases_nash)


# COMPUTATIONS

# (a) gamma and epsilon nash flags
is_gamma_nash = max_max_mass_on_suboptim .< 1f-2
is_epsilon_nash = max_absolute_error .< 1f-4

# (b) compute modal policies for converged sessions and for sessions converged to a gamma nash 

modal_policy_s = fill(NaN32, n_states,n_messages,n_biases)
modal_policy_r = fill(NaN32, n_messages,n_actions,n_biases)
optimal_policy_s = fill(NaN32, n_states,n_messages,n_biases)
optimal_policy_r = fill(NaN32, n_messages,n_actions,n_biases)
optimal_induced_actions = fill(NaN32, n_states,n_actions,n_biases)
modal_induced_actions = fill(NaN32, n_states,n_actions,n_biases)
modal_posterior_mean_variance = fill(NaN32, n_biases)
modal_expected_reward_s = fill(NaN32, n_biases)
modal_expected_reward_r = fill(NaN32, n_biases)
freq_ia = zeros(Float32,n_biases)
for bias_idx in 1:n_biases
	global bias = set_biases[bias_idx]
	global reward_matrix_s, reward_matrix_r = gen_reward_matrix()
	# optimal (Pareto-efficient / receiver-preferred) equilibrium policy at this bias
	best_nash_i = get_best_nash()
	optimal_policy_s[:,:,bias_idx] = best_nash_i["best_policy_s"]
	optimal_policy_r[:,:,bias_idx] = best_nash_i["best_policy_r"]
	optimal_induced_actions[:,:,bias_idx] = best_nash_i["best_induced_actions"]
	# modal induced actions for converged sessions
	induced_actions_ = induced_actions[:,:,:,bias_idx]
	unique_induced_actions = unique(induced_actions_,dims=3)
	freq_induced_actions  =	[count(all(induced_actions_ .== induced_actions, dims=1:2)) for induced_actions in eachslice(unique_induced_actions, dims=3)]
	modal_induced_actions[:,:,bias_idx] = unique_induced_actions[:,:,argmax(freq_induced_actions)]
	freq_ia[bias_idx] = maximum(freq_induced_actions)
 	modal_induced_actions_idx = findfirst(all(induced_actions_ .== modal_induced_actions[:,:,bias_idx],dims=1:2)[:])
	modal_policy_s[:,:,bias_idx] = policy_s[:,:,modal_induced_actions_idx,bias_idx]
	modal_policy_r[:,:,bias_idx] = policy_r[:,:,modal_induced_actions_idx,bias_idx]
	modal_posterior_mean_variance[bias_idx] = get_posterior_mean_variance(modal_policy_s[:,:,bias_idx])
	modal_expected_reward_s[bias_idx], modal_expected_reward_r[bias_idx] = get_expected_rewards(modal_induced_actions[:,:,bias_idx])
end

# (c) compute range of existence of monotone partitional equilibria (identified by their induced actions)
unique_induced_actions_nash = unique(cat(unique(induced_actions_nash)...,dims=3),dims=3)
n_unique_induced_actions = size(unique_induced_actions_nash,3)
existence_range = zeros(Float32, n_unique_induced_actions,2)
set_biases_nash = range(0.0f0,0.5f0,n_biases_nash)
unique_posterior_mean_variance_nash = zeros(Float32, n_unique_induced_actions)
for ia_idx in 1:n_unique_induced_actions
	exists = falses(n_biases_nash)
	for bias_idx in 1:n_biases_nash
		nash_idx = all(unique_induced_actions_nash[:,:,ia_idx] .== induced_actions_nash[bias_idx], dims=1:2)[:]
		exists[bias_idx] = any(nash_idx) || continue
		unique_posterior_mean_variance_nash[ia_idx] = posterior_mean_variance_nash[bias_idx][nash_idx][1]
	end
	existence_range[ia_idx,:] .= set_biases_nash[[findfirst(exists), findlast(exists)]]
end
perm = sortperm(-unique_posterior_mean_variance_nash)
unique_posterior_mean_variance_nash = unique_posterior_mean_variance_nash[perm]
existence_range = existence_range[perm,:]



# GENERATE PLOTS

# N EPISODES
pl_n_episodes = plot_avg(n_episodes; title = "episodes to converge",color = "blue", legend_pos = "out_bottom");
pl_n_episodes = plot_eq_bound!(pl_n_episodes,posterior_mean_variance_best);


####  DISTRIBUTION  #####
#########################
# note that distributions are fitted on a grid of n_steps intervals
# overlaying a benchmark value to the grid without snapping values to the grid might be misleading (identical y-values are not aligned)
# for this reason, best_posterior_mean_variance and best_expected_reward_r are snapped to the grid
# best_expected_reward_s is not snapped to the grid as it looks fine (and actually better, because is curved)

# POSTERIOR MEAN VARIANCE
pl_posterior_mean_variance = plot_dist(posterior_mean_variance; title = "normalised posterior mean variance", 
													  legend = "simulations", 
													  color = "blue",
													  legend_pos = "out_bottom",
													  ymin=0, ymax=1, n_steps=65);
# interpolate benchmark points to heatmap grid (removes heatmap value distorsion)
pl_posterior_mean_variance = plot_interpolated_val!(pl_posterior_mean_variance, posterior_mean_variance_best; legend = "optimal", color = "red", style = "solid, line width=$(lw_value)", opacity = 0.4, ymin=0, ymax=1, n_steps=65);
#pl_posterior_mean_variance = plot_val!(pl_posterior_mean_variance, posterior_mean_variance_best; legend = "optimal", color = "red", style = "solid, line width=$(lw_value)", opacity = 0.4);
babbling_posterior_mean_variance = fill(posterior_mean_variance_best[end],length(set_biases))
pl_posterior_mean_variance = plot_val!(pl_posterior_mean_variance, babbling_posterior_mean_variance; legend = "babbling", color = "darkgray", style = "dotted");
pl_posterior_mean_variance = plot_eq_bound!(pl_posterior_mean_variance,posterior_mean_variance_best);

# EXPECTED REWARDS (SENDER)
group_pl_expected_reward_s = plot_dist(expected_reward_s; title = "ex-ante expected reward (sender)",
														  ylabel = get_title_equation(), ylabel_style = "",
														  color = "blue", n_steps = 65, height = main_height);
group_pl_expected_reward_s = plot_val!(group_pl_expected_reward_s, expected_reward_s_best; color="red", style = "solid, line width=$(lw_value)", opacity = 0.4);
group_pl_expected_reward_s = plot_val!(group_pl_expected_reward_s, babbling_reward_s; color="darkgray", style = "dotted");
group_pl_expected_reward_s = plot_eq_bound!(group_pl_expected_reward_s,posterior_mean_variance_best);

# EXPECTED REWARDS (RECEIVER)
ymin = minimum(quantile_(expected_reward_r, 0.05, dims = 1))
ymax = maximum(quantile_(expected_reward_r, 0.95, dims = 1))
group_pl_expected_reward_r = plot_dist(expected_reward_r; title = "ex-ante expected reward (receiver)",
														  legend = "simulations",
														  color = "blue",
														  ymin=ymin, ymax=ymax, n_steps = 65, height = main_height);
group_pl_expected_reward_r = plot_interpolated_val!(group_pl_expected_reward_r, expected_reward_r_best; legend = "optimal equilibrium", color = "red", style = "solid, line width=$(lw_value)", opacity = 0.4, ymin=ymin, ymax=ymax, n_steps=65);
group_pl_expected_reward_r = plot_val!(group_pl_expected_reward_r, babbling_reward_r; legend = "babbling equilibrium", color = "darkgray", style = "dotted");
group_pl_expected_reward_r = plot_eq_bound!(group_pl_expected_reward_r,posterior_mean_variance_best);

# EXPECTED REWARDS (GROUP) — legend captured by name, rendered via \pgfplotslegendfromname
# override the at={(xlabel.south)} from plot_val!'s out_bottom legend: rendered on its own
# there is no xlabel node
push!(group_pl_expected_reward_r.options, "legend style={legend columns = -1, legend to name={legend_expected_rewards}, column sep = 3.5pt, at={(0.98,0.98)}, anchor={north east}}")
pl_expected_rewards = @pgf GroupPlot({group_style={group_size="2 by 1", "horizontal sep" = expected_hsep },}, group_pl_expected_reward_s, group_pl_expected_reward_r);



######  MODE (ALL) ######
#########################
# printing values for modal policies of sender and receiver 
# all converged sessions are considered 

# POSTERIOR MEAN VARIANCE (with monotone partitional equilibria's values in grey)
pl_modal_posterior_mean_variance = init_tikz_axis(title="modal normalised posterior mean variance", xlabel=raw"$b$");
for i in 1:length(unique_posterior_mean_variance_nash)
    start_point, end_point = existence_range[i, :]
    mi_value = unique_posterior_mean_variance_nash[i] 
    if posterior_mean_variance_best[set_biases_nash .== end_point][1] == mi_value # replace grey with red on benchmark
    	end_point_idx = findfirst(posterior_mean_variance_best .== mi_value)
    	end_point_idx > 0 || continue	# totally replaced by benchmark
    	end_point = set_biases_nash[end_point_idx]
    end
    pl = @pgf Plot({no_marks, line_width=lw_value, const_plot, color="gray", opacity = 0.3, forget_plot=(i!=1)}, Coordinates([(start_point, mi_value),(end_point, mi_value)]))
    push!(pl_modal_posterior_mean_variance, pl)    
end
add_legend!(pl_modal_posterior_mean_variance, "equilibria", "out_bottom")
plot_val!(pl_modal_posterior_mean_variance, posterior_mean_variance_best; legend = "optimal", color = "red", style = "solid, line width=$(lw_value)", opacity = 0.4, jump_mark = "jump mark left");
plot_val!(pl_modal_posterior_mean_variance, modal_posterior_mean_variance; color="blue", style="solid, line width=$(lw_modal)", legend = "simulations", opacity = 0.5, jump_mark = "jump mark left");

# EXPECTED REWARDS (SENDER)
ymin, ymax = extrema(modal_expected_reward_s)
pl_modal_expected_reward_s = init_tikz_axis(title="ex-ante expected reward (sender)", xlabel=raw"$b$", ymin=ymin, ymax=ymax);
plot_val!(pl_modal_expected_reward_s, expected_reward_s_best; color = "red", style = "solid, line width=$(lw_value)", opacity = 0.4, jump_mark = "jump mark left");
plot_val!(pl_modal_expected_reward_s, modal_expected_reward_s; color="blue", style="solid, line width=$(lw_modal)", opacity = 0.5, jump_mark = "jump mark left");
plot_val!(pl_modal_expected_reward_s, babbling_reward_s; color="darkgray", style = "dotted", jump_mark = "jump mark left");
plot_eq_bound!(pl_modal_expected_reward_s,posterior_mean_variance_best);

# EXPECTED REWARDS (RECEIVER)
ymin, ymax = extrema(modal_expected_reward_r)
pl_modal_expected_reward_r = init_tikz_axis(title="ex-ante expected reward (receiver)", xlabel=raw"$b$", ymin=ymin, ymax=ymax);
plot_val!(pl_modal_expected_reward_r, expected_reward_r_best; legend = "optimal", color = "red", style = "solid, line width=$(lw_value)", opacity = 0.4, jump_mark = "jump mark left");
plot_val!(pl_modal_expected_reward_r, modal_expected_reward_r; color="blue", style="solid, line width=$(lw_modal)", legend = "simulations", opacity = 0.5, jump_mark = "jump mark left");
plot_val!(pl_modal_expected_reward_r, babbling_reward_r; color="darkgray", style = "dotted", jump_mark = "jump mark left");
plot_eq_bound!(pl_modal_expected_reward_r,posterior_mean_variance_best);

# EXPECTED REWARDS (GROUP)
push!(pl_modal_expected_reward_r.options, "legend style={legend columns = -1, legend to name={legend_expected_rewards}, column sep = 3.5pt}")
pl_expected_rewards_modal = @pgf GroupPlot({group_style={group_size="2 by 1", "horizontal sep" = expected_hsep},}, pl_modal_expected_reward_s, pl_modal_expected_reward_r);



####  SUBOPTIMALITY #####
#########################

# ABSOLUTE ERROR (SENDER)
y_max = max(maximum(mean(absolute_error_s,dims = 1)), maximum(mean(absolute_error_r, dims = 1)))
#y_max = max(maximum(quantile_(extracted_data["absolute_error_s"], 0.95, dims = 1)), maximum(quantile_(extracted_data["absolute_error_r"], 0.95, dims = 1)))
pl_absolute_error_s = plot_avg(absolute_error_s;
								title = raw"potential ex-ante gains (sender)\\[-8pt]", 
								additional = raw"tick scale binop=\times",
								color = "red",
								ymin=0, ymax = y_max,
								ci_flag = false,
								width = 0.35 * ratio, height = error_height);
pl_absolute_error_s = plot_eq_bound!(pl_absolute_error_s,posterior_mean_variance_best);

# ABSOLUTE ERROR (RECEIVER)
pl_absolute_error_r = plot_avg(absolute_error_r;
								title = raw"potential ex-ante gains (receiver)\\[-8pt]", 
								additional = raw"tick scale binop=\times",
								color = "blue",
								ymin=0, ymax = y_max,
								ci_flag = false,
								width = 0.35 * ratio, height = error_height);
pl_absolute_error_r = plot_eq_bound!(pl_absolute_error_r,posterior_mean_variance_best);


# MAXIMUM MASS ON SUBOTPIMAL (SENDER)
y_max = max(maximum(mean(max_mass_on_suboptim_s, dims = 1)), maximum(mean(max_mass_on_suboptim_r, dims = 1)))
#y_max = max(maximum(quantile_(extracted_data["max_mass_on_suboptim_s"], 0.95, dims = 1)), maximum(quantile_(extracted_data["max_mass_on_suboptim_r"], 0.95, dims = 1)))
pl_max_mass_on_suboptim_s = plot_avg(max_mass_on_suboptim_s;
								title = raw"maximum probability on \\[-2pt]suboptimal actions (sender)",
								color = "red",
								ymin = 0, ymax = y_max,
								ci_flag = false,
								width = 0.35 * ratio, height = error_height);
pl_max_mass_on_suboptim_s = plot_eq_bound!(pl_max_mass_on_suboptim_s,posterior_mean_variance_best);

# MAXIMUM MASS ON SUBOTPIMAL (RECEIVER)
pl_max_mass_on_suboptim_r = plot_avg(max_mass_on_suboptim_r;
								title = raw"maximum probability on \\[-2pt]suboptimal actions (receiver)",
								color = "blue",
								ymin = 0, ymax = y_max,
								ci_flag = false,
								width = 0.35 * ratio, height = error_height);
pl_max_mass_on_suboptim_r = plot_eq_bound!(pl_max_mass_on_suboptim_r,posterior_mean_variance_best);


# ABSOLUTE ERROR AND MAXIMUM MASS ON SUBOTPIMAL (GROUP)
pl_optimization_errors = @pgf GroupPlot({group_style={group_size="2 by 2",raw"horizontal sep = 50pt, vertical sep = 60pt"},}, pl_max_mass_on_suboptim_s, pl_max_mass_on_suboptim_r, pl_absolute_error_s, pl_absolute_error_r);


# IS GAMMA NASH
pl_is_gamma_nash = plot_avg(is_gamma_nash;
								title = raw"$max\{\gamma_S,\gamma_R\} < 0.01$",
								color = "green!80!black",
								legend_pos = "out_bottom",
								additional = "axis y discontinuity=parallel",
								ymin=0.55,
								ci_flag = false,
								width = 0.35 * ratio, height = error_height);
pl_is_gamma_nash = plot_eq_bound!(pl_is_gamma_nash,posterior_mean_variance_best);

# IS EPSILON NASH
pl_is_epsilon_nash = plot_avg(is_epsilon_nash;
								title = raw"$max\{\epsilon_S,\epsilon_R\} < 10^{-4}$",
								color = "magenta",
								legend_pos = "out_bottom",
								additional = "axis y discontinuity=parallel",
								ymin=0.55,
								ci_flag = false,
								width = 0.35 * ratio, height = error_height);
pl_is_epsilon_nash = plot_eq_bound!(pl_is_epsilon_nash,posterior_mean_variance_best);

pl_frequency_nash = @pgf GroupPlot({group_style={group_size="2 by 1", raw"horizontal sep = 60pt"},}, pl_is_gamma_nash, pl_is_epsilon_nash);



###  SENDER'S POLICY ####
#########################

# IS PARTITIONAL
pl_is_partitional = plot_dist(is_partitional; 
								title = "is partitional", 
								ymin = 0, ymax = 1);
pl_is_partitional = plot_avg!(pl_is_partitional, is_partitional; ci_flag = false);
pl_is_partitional = plot_eq_bound!(pl_is_partitional,posterior_mean_variance_best);


# group consecutive biases sharing the same induced type->action map (argmax action per type)
function distinct_equilibria(induced, biases)
	maps = [[argmax(induced[t,:,bi]) for t in 1:size(induced,1)] for bi in 1:length(biases)]
	groups = Tuple{Int,Float32,Float32}[]
	start = 1
	for i in 2:length(biases)
		maps[i] == maps[start] && continue
		push!(groups, (start, biases[start], biases[i-1]))
		start = i
	end
	push!(groups, (start, biases[start], biases[end]))
	return groups
end

# bias-range label cell (first column) and empty spacer column
range_title(lo, hi) = string(raw"$[", @sprintf("%.3f", lo), ", ", @sprintf("%.3f", hi), raw"]$")
range_cell(txt, header; width = raw"0.28\linewidth", height = raw"0.25\linewidth", font = raw"\scriptsize") = @pgf Axis({axis_lines = "none", xtick = raw"\empty", ytick = raw"\empty", clip = false, title = header, width = width, height = height, xmin = 0, xmax = 1, ymin = 0, ymax = 1}, string(raw"\node[anchor=center, font=", font, raw"] at (axis cs:0.5,0.5) {", txt, raw"};"))
spacer(; width = raw"0.02\linewidth") = @pgf Axis({hide_axis, "scale only axis", width = width, height = raw"1pt", xmin = 0, xmax = 1, ymin = 0, ymax = 1})


# OPTIMAL EQUILIBRIA
opt_groups = distinct_equilibria(optimal_induced_actions, set_biases)
group_policies_optimal = []
for (j, (idx, b_lo, b_hi)) in enumerate(opt_groups)
	bottom = j == length(opt_groups)
	hm_policy_s = plot_policy(optimal_policy_s[:,:,idx], bottom ? raw"$\theta$" : "", raw"$m$", bottom ? (0:0.5:1) : "", 1:n_messages, (n_states-1)/2, 1, j == 1 ? raw"$\pi_{\ast}^{S}$" : "");
	hm_policy_r = plot_policy(optimal_policy_r[:,:,idx], bottom ? raw"$m$" : "", raw"$a$", bottom ? (1:n_messages) : "", 0:0.25:1, 1, (n_actions-1)/4, j == 1 ? raw"$\pi_{\ast}^{R}$" : "");
	hm_induced = plot_policy(optimal_induced_actions[:,:,idx], bottom ? raw"$\theta$" : "", raw"$a$", bottom ? (0:0.5:1) : "", 0:0.25:1, (n_states-1)/2, (n_actions-1)/4, j == 1 ? raw"$\Theta \times A$" : "");
	n_on_path = findlast(sum(optimal_policy_s[:,:,idx], dims=1) .> 0.01)[2];
	@pgf push!(hm_policy_s, HLine({loosely_dashed, black}, n_messages - n_on_path + 0.5));
	@pgf push!(hm_policy_r, VLine({loosely_dashed, black}, n_on_path + 0.5));
	push!(group_policies_optimal, range_cell(range_title(b_lo, b_hi), j == 1 ? raw"$b$" : ""))
	push!(group_policies_optimal, hm_policy_s)
	push!(group_policies_optimal, hm_policy_r)
	push!(group_policies_optimal, hm_induced)
end
pl_policy_profiles_equilibria = @pgf GroupPlot({group_style = {group_size = "4 by $(length(opt_groups))", raw"horizontal sep = 45pt", raw"vertical sep = 16pt"},}, group_policies_optimal...);

# CONVERGED EQUILIBRIA
conv_groups = distinct_equilibria(modal_induced_actions, set_biases)
group_policies_converged = []
for (j, (idx, b_lo, b_hi)) in enumerate(conv_groups)
	bottom = j == length(conv_groups)
	hm_policy_s = plot_policy(modal_policy_s[:,:,idx], bottom ? raw"$\theta$" : "", raw"$m$", bottom ? (0:0.5:1) : "", 1:n_messages, (n_states-1)/2, 1, j == 1 ? raw"$\pi_{\infty}^{S}$" : "");
	hm_policy_r = plot_policy(modal_policy_r[:,:,idx], bottom ? raw"$m$" : "", raw"$a$", bottom ? (1:n_messages) : "", 0:0.25:1, 1, (n_actions-1)/4, j == 1 ? raw"$\pi_{\infty}^{R}$" : "");
	hm_induced = plot_policy(modal_induced_actions[:,:,idx], bottom ? raw"$\theta$" : "", raw"$a$", bottom ? (0:0.5:1) : "", 0:0.25:1, (n_states-1)/2, (n_actions-1)/4, j == 1 ? raw"$\Theta \times A$" : "");
	n_on_path = findlast(sum(modal_policy_s[:,:,idx], dims=1) .> 0.01)[2];
	@pgf push!(hm_policy_s, HLine({loosely_dashed, black}, n_messages - n_on_path + 0.5));
	@pgf push!(hm_policy_r, VLine({loosely_dashed, black}, n_on_path + 0.5));
	push!(group_policies_converged, range_cell(range_title(b_lo, b_hi), j == 1 ? raw"$b$" : ""))
	push!(group_policies_converged, hm_policy_s)
	push!(group_policies_converged, hm_policy_r)
	push!(group_policies_converged, hm_induced)
end
pl_policy_profiles_learned = @pgf GroupPlot({group_style = {group_size = "4 by $(length(conv_groups))", raw"horizontal sep = 45pt", raw"vertical sep = 16pt"},}, group_policies_converged...);

# COMPARISON: converged and optimal side by side, rows split wherever either changes
# two groupplots (learned | equilibria) so the middle gap (cmp_gap) is independent of column spacing
cmp_groups = distinct_equilibria(vcat(modal_induced_actions, optimal_induced_actions), set_biases)
ncmp = length(cmp_groups)
cells_learned, cells_equil = [], []
for (j, (idx, b_lo, b_hi)) in enumerate(cmp_groups)
	bottom = j == ncmp
	hm_modal_s = plot_policy(modal_policy_s[:,:,idx], bottom ? raw"$\theta$" : "", raw"$m$", bottom ? (0:0.5:1) : "", 1:n_messages, (n_states-1)/2, 1, j == 1 ? raw"$\pi_{\infty}^{S}$" : ""; tick_label_font = cmp_tickfont, width = cmp_panel, height = cmp_panel);
	hm_modal_r = plot_policy(modal_policy_r[:,:,idx], bottom ? raw"$m$" : "", raw"$a$", bottom ? (1:n_messages) : "", 0:0.25:1, 1, (n_actions-1)/4, j == 1 ? raw"$\pi_{\infty}^{R}$" : ""; tick_label_font = cmp_tickfont, ylabel_opts = raw"{rotate=-90, xshift=5pt}", width = cmp_panel, height = cmp_panel);
	hm_optimal_s = plot_policy(optimal_policy_s[:,:,idx], bottom ? raw"$\theta$" : "", raw"$m$", bottom ? (0:0.5:1) : "", 1:n_messages, (n_states-1)/2, 1, j == 1 ? raw"$\pi_{\ast}^{S}$" : ""; tick_label_font = cmp_tickfont, width = cmp_panel, height = cmp_panel);
	hm_optimal_r = plot_policy(optimal_policy_r[:,:,idx], bottom ? raw"$m$" : "", raw"$a$", bottom ? (1:n_messages) : "", 0:0.25:1, 1, (n_actions-1)/4, j == 1 ? raw"$\pi_{\ast}^{R}$" : ""; tick_label_font = cmp_tickfont, ylabel_opts = raw"{rotate=-90, xshift=5pt}", width = cmp_panel, height = cmp_panel);
	n_on_path_modal = findlast(sum(modal_policy_s[:,:,idx], dims=1) .> 0.01)[2];
	n_on_path_optimal = findlast(sum(optimal_policy_s[:,:,idx], dims=1) .> 0.01)[2];
	@pgf push!(hm_modal_s, HLine({loosely_dashed, black}, n_messages - n_on_path_modal + 0.5));
	@pgf push!(hm_modal_r, VLine({loosely_dashed, black}, n_on_path_modal + 0.5));
	@pgf push!(hm_optimal_s, HLine({loosely_dashed, black}, n_messages - n_on_path_optimal + 0.5));
	@pgf push!(hm_optimal_r, VLine({loosely_dashed, black}, n_on_path_optimal + 0.5));
	push!(cells_learned, range_cell(range_title(b_lo, b_hi), j == 1 ? raw"$b$" : ""; width = cmp_panel, height = cmp_panel, font = cmp_tickfont))
	push!(cells_learned, hm_modal_s)
	push!(cells_learned, hm_modal_r)
	push!(cells_equil, hm_optimal_s)
	push!(cells_equil, hm_optimal_r)
end
g_learned = @pgf GroupPlot({group_style = {group_name = "cmpL", group_size = "3 by $(ncmp)", raw"horizontal sep = 42pt", "vertical sep" = cmp_vsep},}, cells_learned...)
# place the equilibria group immediately to the right of the learned group, offset by cmp_gap
cells_equil[1]["at"] = "(cmpL c3r1.north east)"
cells_equil[1]["anchor"] = "north west"
cells_equil[1]["xshift"] = cmp_gap
g_equil = @pgf GroupPlot({group_style = {group_size = "2 by $(ncmp)", raw"horizontal sep = 42pt", "vertical sep" = cmp_vsep},}, cells_equil...)
pl_policy_profiles_learned_vs_equilibria = @pgf TikzPicture(g_learned, g_equil)

# COMPARISON: induced state-action distribution, learned vs optimal side by side
# like policy_profiles_learned_vs_equilibria but each half is a single Theta x A heatmap
# (modal learned vs optimal); rows split wherever either changes
sad_groups = distinct_equilibria(vcat(modal_induced_actions, optimal_induced_actions), set_biases)
nsad = length(sad_groups)
sad_gap = "50pt"   # horizontal gap between the learned and the equilibria halves
cells_sad_learned, cells_sad_equil = [], []
for (j, (idx, b_lo, b_hi)) in enumerate(sad_groups)
	bottom = j == nsad
	hm_sad_modal = plot_policy(modal_induced_actions[:,:,idx], bottom ? raw"$\theta$" : "", raw"$a$", bottom ? (0:0.5:1) : "", 0:0.25:1, (n_states-1)/2, (n_actions-1)/4, j == 1 ? raw"$(\Theta \times A)_{\infty}$" : ""; tick_label_font = raw"\scriptsize");
	hm_sad_optimal = plot_policy(optimal_induced_actions[:,:,idx], bottom ? raw"$\theta$" : "", raw"$a$", bottom ? (0:0.5:1) : "", 0:0.25:1, (n_states-1)/2, (n_actions-1)/4, j == 1 ? raw"$(\Theta \times A)_{\ast}$" : ""; tick_label_font = raw"\scriptsize", ylabel_opts = raw"{rotate=-90, xshift=5pt}");
	push!(cells_sad_learned, range_cell(range_title(b_lo, b_hi), j == 1 ? raw"$b$" : ""))
	push!(cells_sad_learned, hm_sad_modal)
	push!(cells_sad_equil, hm_sad_optimal)
end
g_sad_learned = @pgf GroupPlot({group_style = {group_name = "sadL", group_size = "2 by $(nsad)", raw"horizontal sep = 42pt", raw"vertical sep = 16pt"},}, cells_sad_learned...)
# place the equilibria heatmap immediately to the right of the learned half, offset by sad_gap
cells_sad_equil[1]["at"] = "(sadL c2r1.north east)"
cells_sad_equil[1]["anchor"] = "north west"
cells_sad_equil[1]["xshift"] = sad_gap
g_sad_equil = @pgf GroupPlot({group_style = {group_size = "1 by $(nsad)", raw"vertical sep = 16pt"},}, cells_sad_equil...)
pl_state_action_distribution_learned_vs_equilibria = @pgf TikzPicture(g_sad_learned, g_sad_equil)

# sampled biases retained for the policy-sample figure below
bias_idxs = trunc.(Int,[1 + (n_biases - 1) / 20 * (2^i - 1) for i in 0:4])


# LEARNED POLICIES (modal converged policy at sampled biases)
group_policies_s_sample, group_policies_r_sample = [], []
for bias_idx in bias_idxs
	hm_modal_policy_s = plot_policy(modal_policy_s[:,:,bias_idx], raw"$\theta$", bias_idx == 1 ? raw"$m$" : "", 0:0.5:1, bias_idx == 1 ? (1:n_messages) : "", (n_states-1)/2, 1, string(raw"$b=",set_biases[bias_idx],raw"$"));
	hm_modal_policy_r = plot_policy(modal_policy_r[:,:,bias_idx], raw"$m$",  bias_idx == 1 ? raw"$a$" : "", 1:n_messages, bias_idx == 1 ? (0:0.25:1) : "", 1, (n_actions-1)/4, "");
	n_on_path_messages_modal = findlast(sum(modal_policy_s[:,:,bias_idx], dims=1) .> 0.01)[2];
	@pgf push!(hm_modal_policy_r,VLine({loosely_dashed, black}, n_on_path_messages_modal+0.5));
	@pgf push!(hm_modal_policy_s,HLine({loosely_dashed, black}, n_messages-n_on_path_messages_modal+0.5));
	push!(group_policies_s_sample, hm_modal_policy_s)
	push!(group_policies_r_sample, hm_modal_policy_r)
end
pl_policy_profiles_learned_sample = @pgf GroupPlot({group_style={group_size="$(length(bias_idxs)) by 2", raw"horizontal sep = 10pt", raw"vertical sep = 35pt"},}, group_policies_s_sample..., group_policies_r_sample...);

pl_freq_modal_policy_profile = init_tikz_axis(title = "frequency of modal policy profile", ymin=0, ymax=1)
pl_freq_modal_policy_profile = plot_val!(pl_freq_modal_policy_profile,freq_ia/n_simulations)

# save plots
tikz_dir = mkpath(save_dir)
pdf_dir = mkpath(pdf_dir)

save_plot(file_name, plot) = (pgfsave(joinpath(tikz_dir, "$file_name.tikz"), plot); pgfsave(joinpath(pdf_dir, "$file_name.pdf"), plot))

plots = [
    	("posterior_mean_variance", pl_posterior_mean_variance),
    	("posterior_mean_variance_modal", pl_modal_posterior_mean_variance),
    	("expected_rewards", pl_expected_rewards),
    	("expected_rewards_modal", pl_expected_rewards_modal),
       	("policy_profiles_equilibria", pl_policy_profiles_equilibria),
       	("policy_profiles_learned", pl_policy_profiles_learned),
       	("policy_profiles_learned_vs_equilibria", pl_policy_profiles_learned_vs_equilibria),
       	("state_action_distribution_learned_vs_equilibria", pl_state_action_distribution_learned_vs_equilibria),
       	("policy_profiles_learned_sample", pl_policy_profiles_learned_sample),
    	("optimization_errors", pl_optimization_errors),
    	("frequency_nash", pl_frequency_nash),
    	("is_partitional", pl_is_partitional),
    	("n_episodes", pl_n_episodes),
    	("freq_modal_policy_profile", pl_freq_modal_policy_profile),
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

end  # make_figures


# command-line entry: build all figures for the -i dir
if abspath(PROGRAM_FILE) == @__FILE__
	script_config = parse_commandline("out_basecase", 0.01f0)
	input_dir = joinpath(pwd(), script_config["in_dir"])
	dirs = readdir(input_dir, join = true)
	dirs = dirs[isdir.(dirs)]
	dirs = setdiff(dirs, joinpath.(input_dir, ["pdf", "temp", "tikz"]))
	dir_id = 1
	if length(dirs) > 1
		println("\nlist of directories: ")
		for (i, dir) in enumerate(dirs)
			println(i, ": ", basename(dir))
		end
		while true
			print("select directory: ")
			global dir_id = parse(Int, readline())
			1 <= dir_id <= length(dirs) && break
		end
	end
	println("\nInput dir: ", script_config["in_dir"])
	make_figures(dirs[dir_id]; step_bias = script_config["step_bias"])
end
