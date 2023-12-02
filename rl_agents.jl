
# play and learn: training 

function run_simulation(;rng::MersenneTwister=MersenneTwister())
    # main function, runs simulation and returns Q matrices of the agents 
    Q_s, Q_r, policy_s, policy_r = init_agents()                            # initialize Q-matrices and policies of the agents
    policy_s_, policy_r_ = copy(policy_s), copy(policy_r)                   # copy of agents policies to asess convergence
    n_r, n_s = zeros(Int64, n_agents), zeros(Int64, n_agents)		        # n_s, s_r count episodes w/ similar policy
    ep, ep_s, ep_r = 1, ones(Int64, n_agents), ones(Int64, n_agents)	    # number of episodes played for each agent and globally
    @inbounds while ep < n_max_episodes
        i, j = rand(1:n_agents), rand(1:n_agents)                                         # randomly select a sender and a receiver
        t = sample_(rng, p_t)                                                             # draw state of the world from prior
        m = get_action(view(policy_s,:,:,i), view(Q_s,:,:,i), temp_s[ep_s[i]], t, rng)    # get action of sender (softmax)
        a = get_action(view(policy_r,:,:,j), view(Q_r,:,:,j), temp_r[ep_r[j]], m, rng)    # get action of receiver (softmax)
        reward_s, reward_r = reward_matrix_s[a,t], reward_matrix_r[a,t]                   # get utilities
        Q_s[:,:,j] = update_q(view(Q_s,:,:,i), t, m, reward_s, alpha_s)                   # update Q-matrix of sender
        Q_r[:,:,j] = update_q(view(Q_r,:,:,j), m, a, reward_r, alpha_r)                   # update Q-matrix of receiver
        n_s[i] = is_approx(view(policy_s,:,:,i), view(policy_s_,:,:,i)) ? n_s[i] + 1 : 0  # if policy approx unchanged increment else reset
        n_r[j] = is_approx(view(policy_r,:,:,j), view(policy_r_,:,:,j)) ? n_r[j] + 1 : 0  # if policy approx unchanged increment else reset
        n_s[i] == 0 && copy!(view(policy_s_,:,:,i), view(policy_s,:,:,i))                 # if policy changed 
        n_r[j] == 0 && copy!(view(policy_r_,:,:,j), view(policy_r,:,:,j))                 # if policy changed 
        min(minimum(n_s), minimum(n_r)) == convergence_threshold && break                 # break if policies have converged                
        ep, ep_s[i], ep_r[j] = ep+1, ep_s[i]+1, ep_r[j]+1
    end
    return Q_s, Q_r, ep_s, ep_r
end

# Q-learning

function init_agents()
    # initialize Q matrices and policies
    if q_init == "random"
        # randomly inizialize Q-matrices in interval [babbling_reward_i,0]
        Q_s = babbling_reward_s * rand(Float32, n_states, n_messages, n_agents)
        Q_r = babbling_reward_r * rand(Float32, n_messages, n_actions, n_agents)
    elseif q_init == "optimistic"
        # optimistic initialization (agents get take more than 0)
        Q_s = zeros(Float32, n_states, n_messages, n_agents)
        Q_r = zeros(Float32, n_messages, n_actions, n_agents)
    elseif q_init == "pessimistic"
        # pessimistic initialization
        Q_s = babbling_reward_s * ones(Float32, n_states, n_messages, n_agents)
        Q_r = babbling_reward_r * ones(Float32, n_messages, n_actions, n_agents)
    end
    policy_s = mapslices(x -> get_policy(x, temp_0), Q_s, dims = 1:2)
    policy_r = mapslices(x -> get_policy(x, temp_0), Q_r, dims = 1:2)
    return Q_s, Q_r, policy_s, policy_r
end

function get_action(policy::AbstractArray{Float32,2}, Q::AbstractArray{Float32,2}, temp::Float32, state::Int64, rng::MersenneTwister)
    # get action according to softmax distribution and update policy by reference
    # ∀c∈R softmax_i(x+c) = exp(x_i+c)/sum_i{exp(x_i+c)} = exp(x_i)/sum_i{exp(x_i)} = softmax_i(x)
    # exp.(x - max(x)) ensures softmax is numerically stable
    # warning: might result in NaN if temp gets extremely small (switch to Float64 in that case)
    n_states, n_actions = size(policy)
    @inbounds for state in 1:n_states
        cum_sum = 0.0f0
        max_val = maximum_(view(Q,state,:))
        @turbo for action in 1:n_actions
            policy[state, action] = exp((Q[state,action] - max_val) / temp)
            cum_sum += policy[state, action]
        end
        @turbo for action in 1:n_actions
            policy[state, action] /= cum_sum
        end   
    end 
    return sample_(rng, view(policy,state,:))
end

function update_q(Q::AbstractArray{Float32,2}, state::Int64, action::Int64, reward::Float32, alpha::Float32)
    # value iteration:  Q(s,a) <- Q(s,a) + alpha [ R - Q(s,a) ]
    @fastmath @inbounds Q[state,action] = (1 - alpha) * Q[state,action] + alpha * reward 
    return Q
end

# reward of the agents

function gen_reward_matrix(loss_type::String = loss_type)
    # reward function for Crowfard and Sobel on a unit line
    function reward(a,t,b)
        loss_type == "quadratic" && return -abs2(view(A,a) .- view(T,t) .- b)
        loss_type == "absolute" && return -abs(view(A,a) .- view(T,t) .- b)
        loss_type == "fourth" && return -(view(A,a) .- view(T,t) .- b)^4
    end
    reward_matrix_s = Array{Float32,2}(undef, n_actions, n_states)
    reward_matrix_r = Array{Float32,2}(undef, n_actions, n_states)
    for t in 1:n_states, a in 1:n_actions
        reward_matrix_s[a,t] = reward(a,t,bias)
        reward_matrix_r[a,t] = reward(a,t,0.0f0)
    end
    return reward_matrix_s, reward_matrix_r
end

# communication channel
#=
function get_signal(m::Int64, rng::MersenneTwister)
    # get signal via noisy channel (TODO: transition probability, or use sample_())
    if noise == 0.0 || rand(rng) > noise
        # message is transmitted correctly with probability 1-noise
        return m
    else
        # message is transmitted incorrectly with probability noise
        m_ = rand(rng, 1:n_messages - 1)
        return m_ >= m ? m_ + 1 : m_
    end
end
=#

# prior distribution

function gen_distribution(;dist_type::String = dist_type)
    # prior distribution over states of the world
    if dist_type == "uniform"
        p_t = ones(Float32, n_states) / n_states
    elseif dist_type == "increasing"
        p_t = Float32.((1:n_states))
        p_t ./= sum(p_t)
    elseif dist_type == "decreasing"
        p_t = Float32.((n_states:-1:1))
        p_t ./= sum(p_t)
    elseif dist_type == "vshaped"
        p_t = [abs(t - (n_states+1)/2) .+ 1.0 for t in 1:n_states]
        p_t ./= sum(p_t)
    elseif dist_type == "binomial"
        p = 0.5f0
        p_t::Array{Float32} = binomial.(n_states-1,0:n_states-1) .* (p).^(0:n_states-1) .* (1-p).^(n_states-1:-1:0)
    end
    return p_t
end

# generic functions

function is_approx(A::AbstractArray{Float32,2}, A_::AbstractArray{Float32,2})
    # fast isapprox(), discussion at https://discourse.julialang.org/t/faster-isapprox/101202/8
    norm_A, norm_A_, norm_diff = 0.0f0, 0.0f0, 0.0f0
    @turbo for i in eachindex(A)
        a, a_ = A[i], A_[i]
        norm_A += abs2(a)
        norm_A_ += abs2(a_)
        norm_diff += abs2(a - a_)
    end
    return norm_diff < abs2(rtol) * max(norm_A, norm_A_)
end

function argmax_(A::AbstractArray{Float32,1}; idxs = Array{Int64,1}(undef,length(A)))
    # fast argmax(), discussion at https://discourse.julialang.org/t/how-to-efficiently-find-the-set-of-maxima-of-an-array/73423/5  
    max_val = -Inf32
    n = 0
    @inbounds for i in eachindex(A)
        a = A[i]
        a < max_val && continue
        if a > max_val
            max_val = a
            n = 1
            idxs[n] = i
        else
            idxs[n+=1] = i
        end
    end
    return view(idxs,1:n)
end

function maximum_(A::AbstractArray{Float32,1})
    # fast maximum(), discussion at https://discourse.julialang.org/t/how-to-efficiently-find-the-set-of-maxima-of-an-array/73423/5  
    max_val = -Inf32
    @inbounds for i in eachindex(A)
        a = A[i]
        a < max_val && continue
        if a > max_val
            max_val = a
        end
    end
    return max_val
end

function sample_(rng::AbstractRNG, wv::AbstractArray{Float32,1})
    # fast weighted sampling, same as StatsBase.jl w/out probability weights
    # weights wv need to sum to 1.0
    t = rand(rng)
    n = length(wv)
    i = 1
    cw = wv[1]
    @fastmath @inbounds while cw < t && i < n
        i += 1
        cw += wv[i]
    end
    return i
end
