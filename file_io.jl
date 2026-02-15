
# load libraries
using ArgParse
using TOML
using PrettyTables
using JLD2

macro names(arg...) string.(arg) end

function parse_commandline()
    arg_settings = ArgParseSettings(allow_ambiguous_opts=true)
    @add_arg_table! arg_settings begin
        # GAME SETTINGS
        "--n_states", "-n"
            arg_type = Int64
            help = "number of states of the world"
            default = 6
            range_tester = x -> x > 0
        "--n_messages", "-m"
            arg_type = Int64
            help = "number of messages, set to -1 to have as many messages as in the sender-preferred equilibrium (default: n_states)"
            range_tester = x -> (x > 0 || x == -1)
        "--n_actions", "-a"
            arg_type = Int64
            help = "number of actions (default: 2*n_states-1)"
            range_tester = x -> x > 0
        "--bias", "-b"
            arg_type = Float32
            help = "sender's bias"
            default = 0.075f0
            range_tester = x -> x >= 0.0f0
        "--loss", "-l"
            arg_type = String
            help = "utility functions: \"quadratic\", \"fourth\" or \"absolute\""
            default = "quadratic"
            range_tester = x -> x in ["quadratic", "fourth", "absolute"]
        "--distribution", "-d"
            arg_type = String
            help = "distribution over states: \"uniform\", \"increasing\", \"decreasing\", \"vshaped\", \"binomial\""
            default = "uniform"
            dest_name = "dist"  
            range_tester = x -> x in ["uniform", "increasing", "decreasing" , "vshaped", "binomial"] 
        #"--noise", "-e"
        #    arg_type = Float32
        #    help = "error rate of communication channel (not implemented)"
        #    default = 0.0f0  
        #    range_tester = x -> x >= 0.0f0 
        # SIMULATION SETTINGS
        "--n_simulations", "-N"
            arg_type = Int64
            help = "number of simulations"
            default = 100  
            range_tester = x -> x > 0
        # IO SETTINGS
        "--quiet", "-q"
            help = "flag to not print input/output to terminal"
            action = :store_true
        "--save", "-s"
            help = "flag to save all results (except reward paths)"
            action = :store_true
        "--raw", "-r"
            help = "flag to exclusively run simulations and collect raw output"
            action = :store_true
        "--out_dir", "-o"
            arg_type = String
            help = "output directory name"
            default = "out"
        "--config", "-c"
            arg_type = String
            help = "config section name"
            default = "default"
    end
    parsed_args = parse_args(arg_settings)
    # default n_messages is n_states, if -1 then n_messages = n_messages_on_path
    if parsed_args["n_messages"] == nothing
        parsed_args["n_messages"] = parsed_args["n_states"]
    elseif parsed_args["n_messages"] == -1
        parsed_args["n_messages"] = get_N(parsed_args["bias"], parsed_args["n_states"])[1]
    end
    # default n_actions is 2 * n_states -1 
    if parsed_args["n_actions"] == nothing
        parsed_args["n_actions"] = 2 * parsed_args["n_states"] - 1
    end
    # raw mode is quiet and does not save_all
    if parsed_args["raw"] == true
        parsed_args["save"] = true
        parsed_args["quiet"] = true
    end
    return parsed_args
end

function show_experiment_details()
    # save experiment details to txt file, and show them on terminal
    open("$temp_dir/experiment_details.txt","w") do io
        println(io, "\n [GAME]")
        println(io, " n_states: \t ", n_states)
        println(io, " n_actions: \t ", n_actions)
        println(io, " n_messages: \t ", n_messages)
        println(io, " loss type: \t ", loss_type)
        println(io, " distribution: \t ", dist_type)
        println(io, " bias: \t\t ", bias)
        #println(io, " noise: \t ", noise)  
        #
        println(io, "\n [Q-LEARNING]")
        println(io, " n_simulations:  " , n_simulations)
        println(io, " max_episodes: \t " , n_max_episodes)
        println(io, " conv_threshold: " , convergence_threshold)
        println(io, " rtol: \t\t ", rtol)
        println(io, " q_init: \t " , q_init)        
        println(io, " alpha_s: \t " , alpha_s)
        println(io, " alpha_r: \t " , alpha_r)
        println(io, " policy_type: \t " , policy_type)
        println(io, " expl_decay_s: \t " , round.(expl_decay_s,digits=9))
        println(io, " expl_decay_r: \t " , round.(expl_decay_r,digits=9))
        println(io, " expl0_s: \t " , expl0_s)
        println(io, " expl0_r: \t " , expl0_r)
        println(io)
    end
    quiet || run(`cat $temp_dir/experiment_details.txt`)
end

function gen_dirs()
    current_dir = pwd()
    output_dir = joinpath(current_dir, config["out_dir"])
    temp_dir = joinpath(output_dir, "temp")
    # make sure output folder exists
    mkpath(output_dir)
    # overwrite temp folder if it exists
    isdir(temp_dir) && rm(temp_dir, recursive=true)
    mkdir(temp_dir)
    return output_dir, temp_dir
end

format_txt(val::AbstractFloat; digits = 5) = string(round.(Float64.(val[1]), digits=digits))
format_txt(val::Tuple; digits = 5) = string(round.(Float64.(val[1]), digits=digits), " (", round.(Float64.(val[2]), digits=6),")")
format_txt(val::AbstractArray; digits = 5) =  string(round.(Float64.(val), digits = digits))
format_entry(dict_val, key; std=false, digits = 5) = haskey(dict_val, key) ? (std ? format_txt(dict_val[key], digits=digits) : format_txt(dict_val[key][1], digits=digits)) : " -- "

function add_row(rows, dict_val, var_name; keys_ = ["converged", "not_converged"], std = true, text = var_name, digits = 5)
    row::Any = [text]
    for key in keys_
        row = hcat(row, format_entry(dict_val[key], var_name; std = std, digits = digits))
    end
    push!(rows, row)
end

function show_experiment_outcomes(set_nash, best_nash, statistics)
    !raw || return nothing
    
    best_nash_header = (["BEST NASH", ""])
    best_nash_table::Any = []
    push!(best_nash_table, ["n_messages_on_path" best_nash["n_messages_on_path"]])    
    push!(best_nash_table, ["best_posterior_mean_var" round(best_nash["best_posterior_mean_variance"], digits=4)])
    push!(best_nash_table, ["best_expe_rewards_s" round(best_nash["best_expected_reward_s"], digits=4)])
    push!(best_nash_table, ["best_expe_rewards_r" round(best_nash["best_expected_reward_r"], digits=4)])
    push!(best_nash_table, ["is_borderline" best_nash["is_borderline"]])

    statistics_header = (["", "CONVERGED", "NOT CONVERGED"])
    statistics_table::Any = []
    push!(statistics_table, ["[CONVERGENCE]" "" ""])
    add_row(statistics_table, statistics, "freq"; std = false)
    add_row(statistics_table, statistics, "avg_n_episodes")
    push!(statistics_table, ["[REWARD METRICS]" "" ""])
    add_row(statistics_table, statistics, "avg_expected_reward_s")
    add_row(statistics_table, statistics, "avg_expected_reward_r")
    push!(statistics_table, ["[POLICY METRICS]" "" ""])
    add_row(statistics_table, statistics, "avg_posterior_mean_variance")
    add_row(statistics_table, statistics, "avg_n_on_path_messages")
    add_row(statistics_table, statistics, "avg_n_effective_messages")
    add_row(statistics_table, statistics, "freq_partitional")
    push!(statistics_table, ["[Q METRICS]" "" ""])
    add_row(statistics_table, statistics, "freq_is_absorbing")
    add_row(statistics_table, statistics, "freq_is_greedy_s")
    add_row(statistics_table, statistics, "freq_is_greedy_r")
    add_row(statistics_table, statistics, "avg_margin_error_s", digits = 2)
    add_row(statistics_table, statistics, "avg_margin_error_r", digits = 2)
    push!(statistics_table, ["[EPSILON NASH]" "" ""])
    add_row(statistics_table, statistics, "avg_absolute_error_s", text = "avg_epsilon_s")
    add_row(statistics_table, statistics, "avg_absolute_error_r", text = "avg_epsilon_r")
    add_row(statistics_table, statistics, "quant_max_absolute_error", text = "epsilon_nash (.90, .95, 1)")
    push!(statistics_table, ["[GAMMA NASH]" "" ""])
    add_row(statistics_table, statistics, "avg_max_mass_on_suboptim_s", text = " avg_gamma_s")
    add_row(statistics_table, statistics, "avg_max_mass_on_suboptim_r", text = " avg_gamma_r")
    add_row(statistics_table, statistics, "quant_max_mass_on_suboptim", text = "gamma_nash (.25, .50, 0.75)")
    add_row(statistics_table, statistics, "freq_nash", text = "freq_nash (max_γ < 1f-2)"; std = false)
    open("$temp_dir/experiment_outcomes.txt","w") do io
        pretty_table(io, reduce(vcat, best_nash_table), header = best_nash_header, columns_width = [30,40], hlines = [0,1,6])
        pretty_table(io, reduce(vcat, statistics_table), header = statistics_header, columns_width = [30,40,40], hlines = [0,1,2,4,5,7,8,12,13,18,19,22,23,27])
    end
    quiet || run(`cat $temp_dir/experiment_outcomes.txt`)

    nash_idxs = (1:set_nash["n_nash"]...,0)
    nash_header = (["NASH"; nash_idxs...])
    nash_table::Any = []
    push!(nash_table, hcat(["posterior_mean_variance" format_txt.(set_nash["posterior_mean_variance"])... format_entry(statistics[0],"avg_posterior_mean_variance")]))
    push!(nash_table, hcat(["expe_rewards_s" format_txt.(set_nash["expected_reward_s"])... format_entry(statistics[0],"avg_expected_reward_s")]))
    push!(nash_table, hcat(["expe_rewards_r" format_txt.(set_nash["expected_reward_r"])... format_entry(statistics[0],"avg_expected_reward_r")]))
    push!(nash_table, hcat(["is_absorbing" set_nash["is_absorbing"]... format_entry(statistics[0],"freq_is_absorbing")]))
    add_row(nash_table, statistics, "freq", std=false, keys_=nash_idxs)
    add_row(nash_table, statistics, "freq_nash", std=false, text = "freq_nash (max_γ < 1f-2)", keys_=nash_idxs)
    add_row(nash_table, statistics, "avg_max_mass_on_suboptim_s", std=false, text = " avg_gamma_s", keys_=nash_idxs)
    add_row(nash_table, statistics, "avg_max_mass_on_suboptim_r", std=false, text = " avg_gamma_r", keys_=nash_idxs)
    add_row(nash_table, statistics, "freq_partitional", std=false, keys_=nash_idxs)
    add_row(nash_table, statistics, "freq_is_absorbing", std=false, keys_=nash_idxs)
    add_row(nash_table, statistics, "freq_is_greedy_s", std=false, keys_=nash_idxs)
    add_row(nash_table, statistics, "freq_is_greedy_r", std=false, keys_=nash_idxs)
    open("$temp_dir/nash_outcomes.txt","w") do io
        pretty_table(io, reduce(vcat, nash_table), header = nash_header, hlines = [0,1,5,6,13])
    end
    quiet || run(`cat $temp_dir/nash_outcomes.txt`)

    n_classes = findlast(map(x->haskey(statistics,x),-(1:1:n_simulations)))
    !isnothing(n_classes) && begin
    class_idxs = -(1:n_classes)
    class_header = (["NOT ALL NASH (CLASSES)"; -class_idxs...])
    class_table::Any = []
    add_row(class_table, statistics, "avg_posterior_mean_variance", keys_=class_idxs)
    add_row(class_table, statistics, "avg_expected_reward_s", keys_=class_idxs)
    add_row(class_table, statistics, "avg_expected_reward_r", keys_=class_idxs)
    add_row(class_table, statistics, "freq", std=false, keys_=class_idxs)
    add_row(class_table, statistics, "freq_nash", std=false, text = "freq_nash (max_γ < 1f-2)", keys_=class_idxs)
    add_row(class_table, statistics, "avg_max_mass_on_suboptim_s", std=false, text = " avg_gamma_s", keys_=class_idxs)
    add_row(class_table, statistics, "avg_max_mass_on_suboptim_r", std=false, text = " avg_gamma_r", keys_=class_idxs)
    add_row(class_table, statistics, "freq_partitional", std=false, keys_=class_idxs)
    add_row(class_table, statistics, "freq_is_absorbing", std=false, keys_=class_idxs)
    add_row(class_table, statistics, "freq_is_greedy_s", std=false, keys_=class_idxs)
    add_row(class_table, statistics, "freq_is_greedy_r", std=false, keys_=class_idxs)
    open("$temp_dir/class_outcomes.txt","w") do io
        pretty_table(io, reduce(vcat, class_table), header = class_header, hlines = [0,1,4,5,12])
    end
    quiet || run(`cat $temp_dir/class_outcomes.txt`)
    end

    open("$temp_dir/set_nash.txt", "w") do io
        write(io, "List of non-redundant monotone partitional equilibria (Frug, 2016) ordered from most informative to least informative.")
        for nash_idx in 1:set_nash["n_nash"]
            write(io, "\n\n\nEQUILIBRIUM $nash_idx:")
            write(io, "\n\nEx-ante Expected Reward Sender: \t$(set_nash["expected_reward_s"][nash_idx])")
            write(io, "\nEx-ante Expected Reward Receiver: \t$(set_nash["expected_reward_r"][nash_idx])")
            write(io, "\nPosterior Mean Variance: \t\t$(set_nash["posterior_mean_variance"][nash_idx])")
            write(io, "\nFixed Point: \t\t\t\t$(set_nash["is_absorbing"][nash_idx])")
            write(io, "\n\nPolicy Sender: ")
            policy_s = set_nash["policy_s"][:,:,nash_idx]
            replace_zero_policy_s = [x == 0 ? "" : x for x in policy_s]
            show(io, "text/plain", replace_zero_policy_s)
            write(io, "\n\nPolicy Receiver: ")
            policy_r = set_nash["policy_r"][:,:,nash_idx]
            replace_zero_policy_r = [x == 0 ? "" : x for x in policy_r]
            show(io, "text/plain", replace_zero_policy_r)
            write(io, "\n\nInduced Actions: ")
            induced_actions = set_nash["induced_actions"][:,:,nash_idx]
            replace_zero_induced_actions = [x == 0 ? "" : x for x in induced_actions]
            show(io, "text/plain", replace_zero_induced_actions)
            write(io, "\n\nInduced q_s:: ")
            show(io, "text/plain", set_nash["q_s"][:,:,nash_idx])
            write(io, "\n\nInduced q_r: ")
            show(io, "text/plain", set_nash["q_r"][:,:,nash_idx])
        end
    end
end


# others

function save__(set_nash::Dict, best_nash::Dict, results::Dict, statistics::Dict)
    save_ || return nothing
   
    game_key = join([n_states, n_actions, n_messages, bias, loss_type, dist_type], "_") # noise is omitted
    hyperparameters_key = join([alpha_s, alpha_r, expl_decay_s, expl_decay_r, expl0_s, expl0_r, q_init], "_")
    settings_key = join([n_simulations, n_max_episodes, convergence_threshold, rtol], "_") # irrelevanf for directory name
   
    save("$temp_dir/config.jld2", config)
    save("$temp_dir/set_nash.jld2", set_nash)
    save("$temp_dir/best_nash.jld2", best_nash)
    save("$temp_dir/results.jld2", results)    
    !raw && save("$temp_dir/statistics.jld2", statistics)
   
    out_dir = mkpath(joinpath(output_dir,hyperparameters_key,game_key))
    
    # move temp to folder (this overwrites the content of the destination)
    cp(temp_dir, out_dir, force=true)
end
