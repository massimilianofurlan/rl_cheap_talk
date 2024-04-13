# This reads the simulations outcomes generated by 1_script.jl
# this file is loaded by 1_plots.jl

# load dependencies
using TOML
using JLD2
using StatsBase
using DataStructures
using LoopVectorization
using Random
using ArgParse

# load files and functions 
macro names(arg...) string.(arg) end
include(joinpath(pwd(),"analysis.jl"))
include(joinpath(pwd(),"rl_agents.jl"))

# functions

function parse_commandline()
    arg_settings = ArgParseSettings(allow_ambiguous_opts=true)
    @add_arg_table! arg_settings begin
        # GAME SETTINGS
        "--in_dir", "-i"
            arg_type = String
            help = "input directory"
        "--step_bias"
            arg_type = Float32
            help = "space between points in [0.0,0.5]"
            default = 0.01f0
    end
    parsed_args = parse_args(arg_settings)
    return parsed_args
end

function extract_data(config, results, best_nash, extracted_data)
	# extract and process data from data structures

	# read from files, some variavles must be global
	global bias = config["bias"]
	global reward_matrix_s, reward_matrix_r = k .* gen_reward_matrix()
	babbling_action = argmax(reward_matrix_r*p_t)
	babbling_reward_s::Float32 = p_t'reward_matrix_s[babbling_action,:]
	babbling_reward_r::Float32 = p_t'reward_matrix_r[babbling_action,:]
	babbling_aggregate_reward = babbling_reward_s + babbling_reward_r

	n_max_episodes = config["n_max_episodes"] 	
	Q_s = results["Q_s"]
	Q_r = results["Q_r"]
	n_episodes = results["n_episodes"]
	best_expected_reward_s = best_nash["best_expected_reward_s"]
	best_expected_reward_r = best_nash["best_expected_reward_r"]
	best_expected_aggregate_reward = best_expected_reward_s + best_expected_reward_r
	best_mutual_information = best_nash["best_mutual_information"]

	# print n_states and bias to console
	println("n_states: ", n_states, "\tbias: ", bias)

	# compute is_converged bool and frequence
    is_converged = n_episodes .< n_max_episodes
    n_converged = count(is_converged)
	freq_converged = n_converged / n_simulations

	# preallocate array
    expected_reward_s = Array{Float32,1}(undef, n_converged)
    expected_reward_r = Array{Float32,1}(undef, n_converged)
    expected_aggregate_reward = Array{Float32,1}(undef, n_converged)
    absolute_error_s = Array{Float32,1}(undef, n_converged)
    absolute_error_r = Array{Float32,1}(undef, n_converged)
    max_absolute_error = Array{Float32,1}(undef, n_converged)
    mutual_information = Array{Float32,1}(undef, n_converged)
	# compute metrics of interest for converged sessions
	Threads.@threads for z in 1:n_converged
		is_converged[z] == true || continue
        # get policies at convergence
       	policy_s = get_policy(Q_s[:,:,z], temp0_s*lambda_s^(n_episodes[z]-1))
        policy_r = get_policy(Q_r[:,:,z], temp0_r*lambda_r^(n_episodes[z]-1))
        # compute (ex-ante) expected rewards at convergence
        expected_reward_s[z], expected_reward_r[z] = get_expected_rewards(policy_s, policy_r)
        expected_aggregate_reward[z] = expected_reward_s[z] + expected_reward_r[z]  
        # compute best response to opponent's policy at convergence
        optimal_policy_s = get_best_reply_s(policy_r)
        optimal_policy_r = get_best_reply_r(policy_s)
        # compute expected rewards by best responding to opponent
        optimal_reward_s, _ = get_expected_rewards(optimal_policy_s, policy_r)
        _, optimal_reward_r = get_expected_rewards(policy_s, optimal_policy_r)
        # compute absolute expected error by (possibly) not best responding to opponent
        absolute_error_s[z] = optimal_reward_s - expected_reward_s[z] 
        absolute_error_r[z] = optimal_reward_r - expected_reward_r[z]
		max_absolute_error[z] = max(absolute_error_s[z], absolute_error_r[z])
        # compute communication metrics
        mutual_information[z] = get_mutual_information(policy_s)
    end

	mean_quant(x,α) = mean(x), quantile(x, [α/2, 1-α/2])
	avg_expected_reward_s = mean_quant(expected_reward_s,1.0-α)
	avg_expected_reward_r = mean_quant(expected_reward_r,1.0-α)
	avg_expected_aggregate_reward = mean_quant(expected_aggregate_reward,1.0-α)
	avg_mutual_information = mean_quant(mutual_information,1.0-α)
	avg_absolute_error_s = mean_quant(absolute_error_s,1.0-α)
	avg_absolute_error_r = mean_quant(absolute_error_r,1.0-α)
	avg_n_episodes = mean_quant(n_episodes,1.0-α)
	q_max_absolute_error = quantile(max_absolute_error, β)

	i = findfirst(set_biases .== bias)
	extracted_data["avg_n_episodes"][i] = avg_n_episodes
	extracted_data["freq_converged"][i] = freq_converged
	extracted_data["avg_expected_reward_s"][i] = avg_expected_reward_s
	extracted_data["avg_expected_reward_r"][i] = avg_expected_reward_r
	extracted_data["avg_expected_aggregate_reward"][i] = avg_expected_aggregate_reward
	extracted_data["avg_mutual_information"][i] = avg_mutual_information
	extracted_data["best_expected_reward_s"][i] = best_expected_reward_s
	extracted_data["best_expected_reward_r"][i] = best_expected_reward_r
	extracted_data["best_expected_aggregate_reward"][i] = best_expected_aggregate_reward
	extracted_data["best_mutual_information"][i] = best_mutual_information
	extracted_data["avg_absolute_error_s"][i] = avg_absolute_error_s
	extracted_data["avg_absolute_error_r"][i] = avg_absolute_error_r
	extracted_data["q_max_absolute_error"][i] = q_max_absolute_error
	extracted_data["babbling_reward_s"][i] = babbling_reward_s
	extracted_data["babbling_reward_r"][i] = babbling_reward_r
	extracted_data["babbling_aggregate_reward"][i] = babbling_aggregate_reward
end


# parse terminal config
const scrpt_config = parse_commandline()
# define set of biases 
const set_biases = Float32.(collect(0.00f0:scrpt_config["step_bias"]:0.5f0))
# define input dir 
const input_dir = joinpath(pwd(), scrpt_config["in_dir"])

# navigate to relevant subdir 
dirs = readdir(input_dir, join = true)
dirs = dirs[isdir.(dirs)]
dirs = setdiff(dirs,joinpath.(input_dir,["pdf", "temp", "tikz"]))
dir_id = 1
if length(dirs) > 1
	println("list of directories: ")
	for (dir_id, dir) in enumerate(dirs)
		println(dir_id,": ", split(dir,"/")[end])
	end
	while true
		print("select directory: ")
		global dir_id = parse(Int, readline())
		dir_id in keys(dirs) && break
	end
end
dir = dirs[dir_id]
const subdirs = readdir(dir, join=true)

# read configs that remain constant over all batches of simulations 
const config_ = load(joinpath(subdirs[2],"config.jld2"))	
const n_simulations = config_["n_simulations"]
const n_states = config_["n_states"]
const n_actions = config_["n_actions"]
const n_messages = config_["n_messages"]
const T = collect(0:1f0/(n_states-1):1) 
const A = collect(0:1f0/(n_actions-1):1)
const loss_type = config_["loss"]
const k::Float32 = config_["factor"]
const dist_type = config_["dist"]
const p_t = gen_distribution()
const temp0_s::Float32 = config_["temp0_s"]
const temp0_r::Float32 = config_["temp0_r"]
const lambda_s::Float32 = config_["lambda_s"]
const lambda_r::Float32 = config_["lambda_r"]

# percent of simulation outcomes to fall into confidence interval
const α = 0.95
# quantile for epsilon nash equilibria
const β = 0.9 

function read_data()
	# main loop, read files in subdirs, extract and process data, save to extracted_data
	extracted_data = DefaultDict(() -> Array{Any,1}(undef,length(set_biases)))
	for subdir in subdirs
		isdir(subdir) || continue
		config = load(joinpath(subdir,"config.jld2"))
		results = load(joinpath(subdir,"results.jld2"))
		best_nash = load(joinpath(subdir,"best_nash.jld2"))
		extract_data(config, results, best_nash, extracted_data)
	end
	extracted_data["best_reply_expected_reward_s"] = getindex.(extracted_data["avg_expected_reward_s"],1) .+ getindex.(extracted_data["avg_absolute_error_s"],1)
	extracted_data["best_reply_expected_reward_r"] = getindex.(extracted_data["avg_expected_reward_r"],1) .+ getindex.(extracted_data["avg_absolute_error_r"],1)
	return extracted_data
end