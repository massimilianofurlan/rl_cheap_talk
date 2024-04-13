# This script runs a batch of N simulations for different levels of bias in [0.0,0.5]. 
# Usage:
# 1) navigate to the project directory 'rl_cheap_talk' 
# 2) run 'julia --threads NUM_THREADS scripts/1_script.jl -c CONFIG_SECTION -o -OUT_DIR
#    replace NUM_THREADS with the desired number of threads 
#    replace CONFIG_SECTION with desired config section of config.toml (project directory)
#    replace OUT_DIR with the desired output dir
# Run 'julia scripts/1_script.jl --help' to se all the other options

using TOML
using ArgParse

function parse_commandline()
    arg_settings = ArgParseSettings(allow_ambiguous_opts=true)
    @add_arg_table! arg_settings begin
        # GAME SETTINGS
        "--config", "-c"
            arg_type = String
            help = "config section"
	    default = "basecase"
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
    # default n_messages is n_states
    if parsed_args["n_messages"] == nothing
        parsed_args["n_messages"] = parsed_args["n_states"]
    end
    # default n_actions is 2 * n_states - 1
    if parsed_args["n_actions"] == nothing
        parsed_args["n_actions"] = 2 * parsed_args["n_states"] - 1
    end
    return parsed_args
end

const n_cpus = Threads.nthreads()
const config = parse_commandline()

const n_simulations = config["n_simulations"]
const n_states = config["n_states"]
const n_messages = config["n_messages"]
const n_actions = config["n_actions"]
const set_biases = 0.0f0:config["step_bias"]:0.5f0
const loss = config["loss"]
const distr = config["dist"]
const k = config["factor"]
const out_dir = config["out_dir"]
const config_section = config["config"]

for bias in set_biases
    # cycle over bias
	println("BIAS: ", bias)
	run(`cd ../..`)
	run(`julia  -t $n_cpus main.jl 
				-n=$n_states -m=$n_messages -a=$n_actions -b=$bias -N=$n_simulations -l=$loss -d=$distr -k=$k 
				-o=$out_dir -c=$config_section -r -q`
		)
end

rm("$out_dir/temp",recursive=true,force=true)
