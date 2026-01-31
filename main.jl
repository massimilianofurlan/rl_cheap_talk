
# load libraries
using ProgressMeter
using Random
using StatsBase
# load files
include("file_io.jl")
include("rl_agents.jl")
include("analysis.jl")
include("statistics.jl")
include("nash.jl")

# parse configurations from file and terminal
const term_parse = parse_commandline()
const file_parse = TOML.parsefile("config.toml")[term_parse["config"]]
const config = merge(file_parse, term_parse)
const output_dir, temp_dir = gen_dirs()

# SETTINGS
const quiet = config["quiet"]
const save_ = config["save"]
const raw = config["raw"]

# redirect output to devnull if quiet mode
redirect_stdout(quiet ? devnull : stdout)

# GAME
const n_agents = 2

# number of states of the world
const n_states = config["n_states"]
# number of actions for the receiver
const n_actions = config["n_actions"]
# number of possible messages
const n_messages = config["n_messages"]

# set of states
const T = collect(0:1f0/(n_states-1):1) 
# set of actions
const A = collect(0:1f0/(n_actions-1):1)
# set of messages
const M = n_messages < 26 ? collect('a':'z')[1:n_messages] : collect(1:n_messages)

# loss type 
const loss_type = config["loss"]
# senders' bias
const bias::Float32 = config["bias"]
# reward matrices
const reward_matrix_s, reward_matrix_r = gen_reward_matrix()
# prior distribution 
const dist_type = config["dist"]
const p_t = gen_distribution()
# communication channel
#const noise::Float32 = config["noise"]

# babbling rewards (receiver randomizes uniformly over pure best replies)
const babbling_actions = argmax_(reward_matrix_r*p_t)
const babbling_reward_s::Float32 = mean(p_t'reward_matrix_s[a,:] for a in babbling_actions)
const babbling_reward_r::Float32 = mean(p_t'reward_matrix_r[a,:] for a in babbling_actions)

# Q-LEARNING 

# number of simulations
const n_simulations = config["n_simulations"]
# maximum number of interactions
const n_max_episodes = config["n_max_episodes"]
# convergence requirement
const convergence_threshold = config["convergence_threshold"]

# initialization of Q-matrices
const q_init = config["q_init"]
# learning rate: Q <- (1-alpha) * Q + alpha * R
const alpha_s::Float32 = config["alpha_s"]
const alpha_r::Float32 = config["alpha_r"]
# exploration parameters (decay factor, initial exploration)
const lambda_s::Float32 = config["lambda_s"]
const lambda_r::Float32 = config["lambda_r"]
const temp0_s::Float32 = config["temp0_s"]
const temp0_r::Float32 = config["temp0_r"]
# exploration decay
const temp_s = [max(temp0_s * exp(-lambda_s*(t-1)), 1f-30) for t in 1:n_max_episodes]
const temp_r = [max(temp0_r * exp(-lambda_r*(t-1)), 1f-30) for t in 1:n_max_episodes]

# tolerance on convergence of policies
const rtol = 0.001f0
# tolerance on probabilities (off-path messages, partitional, effective messages)
const ptol = 0.001f0
# tolerance on gamma-nash (gamma < gtol)
const gtol = 0.01f0

show_experiment_details()

function main()
    println(stdout, "computing nash equilibria... ") 
    @time set_nash = get_monotone_partitional_equilibria()
    println(stdout, "computing ex-ante (receiver) optimal nash... ") 
    @time best_nash = get_best_nash()

    println(stdout, "running the simulation...")    
    # preallocate output arrays
    Q_s = Array{Float32,3}(undef, n_states, n_messages, n_simulations);
    Q_r = Array{Float32,3}(undef, n_messages, n_actions, n_simulations);
    n_episodes = Array{Int64,1}(undef, n_simulations);

    rngs = [MersenneTwister(z) for z in 1:n_simulations]
    progress = Progress(n_simulations, color=:white, showspeed=true)

    # main loop
    @time Threads.@threads for z in 1:n_simulations
        rng = rngs[z]
        # initialize Q-matrices
        Q0_s, Q0_r = init_agents(rng)
        # run simulation
        Q_s[:,:,z], Q_r[:,:,z], n_episodes[z] = run_simulation(Q0_s, Q0_r, rng=rng);
        quiet || next!(progress)
    end

    println(stdout, "analyzing outcomes...") 
    @time results = convergence_analysis(Q_s, Q_r, n_episodes); 

    println(stdout, "computing statistics...") 
    @time statistics = compute_statistics(set_nash, results)

    show_experiment_outcomes(set_nash, best_nash, statistics)

    println(save_ ? stdout : devnull, "saving data to file...") 
    @time save__(set_nash, best_nash, results, statistics)
end

Random.seed!(0)
main()

