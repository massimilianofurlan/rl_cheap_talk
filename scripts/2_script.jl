# This script runs a batch of N simulations for different levels of bias in [0.0,0.5]
# over a grid of reinforcement learning hyperparameter (learning rate and exploration decay rate)
# Usage:
# 1) configure the range of hyperparameters by editing scripts/2_config.toml 
# 2) navigate back to the project directory 'rl_cheap_talk' 
# 3) run 'julia --threads NUM_THREADS scripts/2_script.jl -c CONFIG_SECTION -o -OUT_DIR
#    replace NUM_THREADS with the desired number of threads 
#    replace CONFIG_SECTION with desired config section of config.toml (project directory)
#    replace OUT_DIR with the desired output dir
# Run 'julia scripts/1_script.jl --help' to se all the other options

using TOML
using ArgParse

include("nash.jl")

function parse_commandline()
    arg_settings = ArgParseSettings(allow_ambiguous_opts=true)
    @add_arg_table! arg_settings begin
        # GAME SETTINGS
        "--config", "-c"
            arg_type = String
            help = "config section"
            default = "gridsearch"
        "--out_dir", "-o"
            arg_type = String
            help = "output directory"
        "--n_simulations", "-N"
            arg_type = Int64
            help = "number of simulations"
            default = 100  
            range_tester = x -> x > 0
        "--n_states", "-n"
            arg_type = Int64
            help = "number of states of the world"
            default = 11
            range_tester = x -> x > 0
        "--n_messages", "-m"
            arg_type = Int64
            help = "number of messages (default: n_states)"
            range_tester = x -> (x > 0 || x == -1)
        "--n_actions", "-a"
            arg_type = Int64
            help = "number of actions (default: 2*n_states-1)"
            range_tester = x -> x > 0
	   "--step_bias"
            arg_type = Float32
            help = "space between points in [0.0,0.5]"
            default = 0.01f0
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
            range_tester = x -> x in ["uniform", "increasing", "decreasing" , "vshaped","binomial"] 
        "--factor", "-k"
            arg_type = Float32
            help = "utility scale factor"
            default = 1.0f0
    end
    parsed_args = parse_args(arg_settings)
    # default n_messages is n_states, if -1 then n_messages = n_messages_on_path
    if parsed_args["n_messages"] == nothing
        parsed_args["n_messages"] = parsed_args["n_states"]
    elseif parsed_args["n_messages"] == -1
        parsed_args["n_messages"] = get_N(parsed_args["bias"], parsed_args["n_states"])[1]
    end
    # default n_actions is 2 * n_states - 1
    if parsed_args["n_actions"] == nothing
        parsed_args["n_actions"] = 2 * parsed_args["n_states"] - 1
    end 
    return parsed_args
end

const n_cpus = Threads.nthreads()
const config = TOML.parsefile("scripts/2_config.toml")
const config_ = parse_commandline()

const n_simulations = config_["n_simulations"]
const n_states = config_["n_states"]
const n_messages = config_["n_messages"]
const n_actions = config_["n_actions"]
const set_biases = 0.0f0:config_["step_bias"]:0.5f0
const loss = config_["loss"]
const distr = config_["dist"]
const k = config_["factor"]
const out_dir = config_["out_dir"]
const config_section = config_["config"]

const set_alpha = range(config["min_alpha"],config["max_alpha"],config["n_alpha"])
const lin_range_n_ep = range(log(0.01)/log(config["min_lambda"]),log(0.01)/log(config["max_lambda"]),config["n_lambda"])
const set_lambda = round.(0.01.^(1 ./ lin_range_n_ep), digits = 8)

function modify_config_section(new_values)
    # modify config section with new values
    lines = readlines("config.toml")
    in_section = false
    for (i, line) in enumerate(lines)
        if startswith(line, "[$config_section]")
            in_section = true
        elseif startswith(line, "[") && endswith(line, "]")
            in_section = false
        elseif !startswith(line, "#") && contains(line, "=") && in_section
            key, value = split(line, "=", limit=2)
            key = strip(key)
            if haskey(new_values, key)
                lines[i] = "$key = $(new_values[key])"
            end
        end
    end
    # write the modified TOML file back
    open("config.toml", "w") do file
        write(file, join(lines, "\n"))
    end
end

for alpha in set_alpha
    for lambda in set_lambda
        # cycle over grid of alpha and lambda
        new_values = Dict("alpha_s" => alpha, "alpha_r" => alpha, "lambda_s" => lambda, "lambda_r" => lambda)
        modify_config_section(new_values)
        for bias in set_biases
            # cycle over bias
            println("ALPHA: ", alpha , "\tDECAY: ", lambda , "\tBIAS: ", bias)
            run(`julia  -t $n_cpus main.jl 
                        -n=$n_states -m=$n_messages -a=$n_actions -b=$bias -N=$n_simulations -l=$loss -d=$distr -k=$k 
                        -o=$out_dir -c=$config_section -r`
                )
        end
    end
end

rm("$out_dir/temp",recursive=true,force=true)

