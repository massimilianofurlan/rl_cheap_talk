
# play and learn: training 

function run_simulation(rewards::AbstractArray{Float32,2}; rng::MersenneTwister=MersenneTwister())
    # main function, runs simulation and returns Q matrices of the agents 
    Q_s, Q_r, policy_s, policy_r = init_agents()                            # initialize Q-matrices and policies of the agents
    policy_s_, policy_r_ = copy(policy_s), copy(policy_r)                   # copy of agents policies to asess convergence
    n_r, n_s, ep = 0, 0, 1                                                  # n_s, s_r count episodes w/ similar policy
    @inbounds while ep < n_max_episodes
        t = sample_(rng, p_t)                                               # draw state of the world from prior
        m = get_action(policy_s, Q_s, temp_s[ep], t, rng)                   # get action of sender (softmax)
        #x = get_signal(m, rng)                                             # get noisy signal from channel
        a = get_action(policy_r, Q_r, temp_r[ep], m, rng)                   # get action of receiver (softmax)
        reward_s, reward_r = reward_matrix_s[a,t], reward_matrix_r[a,t]     # get utilities
        rewards[ep,:] .= reward_s, reward_r                                 # log utilities
        Q_s = update_q(Q_s, t, m, reward_s, alpha_s)                        # update Q-matrix of sender
        Q_r = update_q(Q_r, m, a, reward_r, alpha_r)                        # update Q-matrix of receiver
        n_s = is_approx(policy_s, policy_s_) ? n_s + 1 : 0                  # if policy approx unchanged increment else reset
        n_r = is_approx(policy_r, policy_r_) ? n_r + 1 : 0                  # if policy approx unchanged increment else reset
        n_s == 0 && copy!(policy_s_, policy_s)                              # if policy changed 
        n_r == 0 && copy!(policy_r_, policy_r)                              # if policy changed 
        min(n_s, n_r) == convergence_threshold && break                     # break if policies have converged                
        ep += 1
    end
    return Q_s, Q_r, ep, n_r - n_s
end

# Q-learning

function init_agents()
    # initialize Q matrices and policies
    if q_init == "random"
        # randomly inizialize Q-matrices in interval [babbling_reward_i,0]
        Q_s = babbling_reward_s * rand(Float32, n_states, n_messages)
        Q_r = babbling_reward_r * rand(Float32, n_messages, n_actions)
    elseif q_init == "optimistic"
        # optimistic initialization (agents get take more than 0)
        Q_s = zeros(Float32, n_states, n_messages)
        Q_r = zeros(Float32, n_messages, n_actions)
    elseif q_init == "pessimistic"
        # pessimistic initialization
        Q_s = babbling_reward_s * ones(Float32, n_states, n_messages)
        Q_r = babbling_reward_r * ones(Float32, n_messages, n_actions)
    end
    policy_s = get_policy(Q_s, temp0_s)
    policy_r = get_policy(Q_r, temp0_r)
    return Q_s, Q_r, policy_s, policy_r
end

function get_action(policy::Array{Float32,2}, Q::Array{Float32,2}, temp::Float32, state::Int64, rng::MersenneTwister)
    # get action according to softmax distribution and update policy by reference
    # ∀c∈R softmax_i(x+c) = exp(x_i+c)/sum_i{exp(x_i+c)} = exp(x_i)/sum_i{exp(x_i)} = softmax_i(x)
    # exp.(x - max(x)) ensures softmax is numerically stable
    # warning: might result in NaN if temp gets extremely small
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

 #=function get_action(policy::Array{Float32,2}, Q::Array{Float32,2}, epsilon::Float32, state::Int64, rng::MersenneTwister)
    # get action according to e-greedy (off-)policy (deterministic policy)
    # define A* = argmax_a Q(a,s) for given s in S
    # a ∈ A* with p = 1-ϵ/|A*| + ϵ/|A|  (|A*| of them)
    # a ∉ A* with p = ϵ / |A|           (|A|-|A*| of them)
    n_states, n_actions = size(policy)
    #@inbounds for state in 1:n_states
        optim_actions = argmax_(view(Q,state,:))
        p = epsilon / n_actions
        q = (1.0f0 - epsilon) / length(optim_actions)
        @inbounds for action in 1:n_actions
            policy[state, action] = p
            if action in optim_actions
                policy[state, action] += q
            end
        end
    #end
    return sample_(rng, view(policy,state,:))
end=#

function update_q(Q::Array{Float32,2}, state::Int64, action::Int64, reward::Float32, alpha::Float32)
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

function is_approx(A::Array{Float32,2}, A_::Array{Float32,2})
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
