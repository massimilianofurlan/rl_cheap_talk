
using ArgParse
using TOML

macro names(arg...) string.(arg) end

function parse_commandline()
    arg_settings = ArgParseSettings(allow_ambiguous_opts=true)
    @add_arg_table! arg_settings begin
        # GAME SETTINGS
        "--n_states", "-n"
            arg_type = Int64
            help = "number of states of the world"
            default = 21
            range_tester = x -> x > 1
        "--n_messages", "-m"
            arg_type = Int64
            help = "number of messages (default: n_states)"
            range_tester = x -> x > 1
        "--bias", "-b"
            arg_type = Float32
            help = "sender's bias"
            default = 0.1f0
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
        "--factor", "-k"
            arg_type = Float32
            help = "utility scale factor"
            default = 1.0f0
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
        "--save_all", "-S"
            help = "flag to save all results (including reward paths)"
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
    # default n_messages is n_states
    if parsed_args["n_messages"] == nothing
        parsed_args["n_messages"] = parsed_args["n_states"]
    end
    # save_all implies saving
    if parsed_args["save_all"] == true
        parsed_args["save"] = true
    end
    # raw mode is quiet and does not save_all
    if parsed_args["raw"] == true
        parsed_args["save"] = true
        parsed_args["quiet"] = true
        parsed_args["save_all"] = false
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
        println(io, " scale factor: \t ", k)
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
        println(io, " lambda_s: \t " , round.(lambda_s,digits=8))
        println(io, " lambda_r: \t " , round.(lambda_r,digits=8))
        println(io, " temp_0: \t " , temp_0)
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

function add_row(rows, statistics, var_name; std = true, text = var_name)
    format_txt(var) = string(round.(Float64.(var[1]),digits=5), " (", round.(Float64.(var[2]),digits=5),")") 
    row::Any = [text]
    for key in ["converged", "not_converged"]
        if haskey(statistics[key], var_name)
            cell = std ? format_txt(statistics[key][var_name]) : string(round.(Float64.(statistics[key][var_name]), digits = 4))
            row = hcat(row, cell)
        else
            row = hcat(row, " -- ")
        end
    end
    push!(rows, row)
end

function show_experiment_outcomes(best_nash, statistics)
    !raw || return nothing
    
    best_nash_header = (["BEST NASH", ""])
    best_nash_table::Any = []
    push!(best_nash_table, ["best_expe_rewards_s" round(best_nash["best_expected_reward_s"], digits=4)])
    push!(best_nash_table, ["best_expe_rewards_r" round(best_nash["best_expected_reward_r"], digits=4)])
    push!(best_nash_table, ["best_expe_aggr_reward" round(best_nash["best_expected_aggregate_reward"], digits=4)])
    push!(best_nash_table, ["best_mutual_info" round(best_nash["best_mutual_information"], digits=4)])
    push!(best_nash_table, ["is_borderline" best_nash["is_borderline"]])

    statistics_header = (["", "CONVERGED", "NOT CONVERGED"])
    statistics_table::Any = []
    push!(statistics_table, ["[CONVERGENCE]" "" ""])
    add_row(statistics_table, statistics, "freq"; std = false)
    add_row(statistics_table, statistics, "avg_n_episodes")
    add_row(statistics_table, statistics, "avg_n_conv_diff", text = "n_r - n_s")
    push!(statistics_table, ["[REWARD METRICS]" "" ""])
    add_row(statistics_table, statistics, "avg_expected_reward_s")
    add_row(statistics_table, statistics, "avg_expected_reward_r")
    add_row(statistics_table, statistics, "avg_expected_aggregate_reward", text = "avg_expe_aggr_reward")
    add_row(statistics_table, statistics, "avg_mutual_information")
    push!(statistics_table, ["[EPSILON NASH]" "" ""])
    add_row(statistics_table, statistics, "avg_absolute_error_s")
    add_row(statistics_table, statistics, "avg_absolute_error_r")
    add_row(statistics_table, statistics, "quant_absolute_error", std = false, text = "epsilon_nash (.90, .95, 1)")
   
    open("$temp_dir/experiment_outcomes.txt","w") do io
        pretty_table(io, reduce(vcat, best_nash_table), header = best_nash_header, columns_width = [30,30], hlines = [0,1,6])
        pretty_table(io, reduce(vcat, statistics_table), header = statistics_header, columns_width = [30,30,30], hlines = [0,1,2,5,6,10,11,14])
    end

    quiet || run(`cat $temp_dir/experiment_outcomes.txt`)
end


# others

function save__(best_nash::Dict, results::Dict, statistics::Dict, rewards::Array{Float32,3})
    save_ || return nothing
   
    game_key = join([n_states, n_actions, n_messages, bias, loss_type, dist_type, k], "_") # noise is omitted
    hyperparameters_key = join([alpha_s, alpha_r, lambda_s, lambda_r, temp_0, q_init], "_")  
    settings_key = join([n_simulations, n_max_episodes, convergence_threshold, rtol], "_") # irrelevanf for directory name
   
    save("$temp_dir/config.jld2", config)
    save("$temp_dir/best_nash.jld2", best_nash)
    save("$temp_dir/results.jld2", results)    
    !raw && save("$temp_dir/statistics.jld2", statistics)
   
    out_dir = mkpath(joinpath(output_dir,hyperparameters_key,game_key))
    
    # move temp to folder (this overwrites the content of the destination)
    cp(temp_dir, out_dir, force=true)

    # shrink rewards and save directly on outputdir to avoid moving large files
    save_all == true && save("$out_dir/rewards.jld2", Dict("rewards" => Float16.(rewards[1:maximum(results["n_episodes"]),:,:])))
end
