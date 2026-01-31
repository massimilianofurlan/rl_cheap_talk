
# play and learn: training 

function run_simulation(Q_s::Array{Float32,2}, Q_r::Array{Float32,2}; rng::MersenneTwister=MersenneTwister())
    # main function, runs simulation and returns Q matrices of the agents 
    policy_s, policy_r = get_policy(Q_s, temp0_s), get_policy(Q_r, temp0_r) # initialize policies
    policy_s_, policy_r_ = copy(policy_s), copy(policy_r)                   # copy of agents policies to asess convergence
    n_r, n_s, ep = 0, 0, 1                                                  # n_s, s_r count episodes w/ similar policy
    while ep < n_max_episodes
        t = sample_(rng, p_t)                                               # draw state of the world from prior
        m = get_action(policy_s, Q_s, temp_s[ep], t, rng)                   # get action of sender (softmax)
        a = get_action(policy_r, Q_r, temp_r[ep], m, rng)                   # get action of receiver (softmax)
        reward_s, reward_r = reward_matrix_s[a,t], reward_matrix_r[a,t]     # get utilities
        Q_s = update_q(Q_s, t, m, reward_s, alpha_s)                        # update Q-matrix of sender
        Q_r = update_q(Q_r, m, a, reward_r, alpha_r)                        # update Q-matrix of receiver
        n_s = is_approx_unchanged(policy_s, policy_s_, n_s)                 # if policy approx unchanged increment else reset
        n_r = is_approx_unchanged(policy_r, policy_r_, n_r)                 # if policy approx unchanged increment else reset
        min(n_s, n_r) == convergence_threshold && break                     # break if policies have converged                
        ep += 1
    end
    return Q_s, Q_r, ep
end

function is_approx_unchanged(policy::Array{Float32,2}, policy_::Array{Float32,2}, n::Int64)
    # if policy is approximately equal to policy_ increment n else reset to 0 and set new current policy
    if is_approx(policy, policy_)
        n += 1
    else
        n = 0
        copy!(policy_, policy) 
    end
    return n
end

# Q-learning

function init_agents(rng::MersenneTwister)
    # initialize Q matrices and policies
    if q_init == "random"
        # randomly inizialize Q-matrices in interval [babbling_reward_i,0]
        Q_s = babbling_reward_s * rand(rng, Float32, n_states, n_messages)
        Q_r = babbling_reward_r * rand(rng, Float32, n_messages, n_actions)
    elseif q_init == "optimistic"
        # optimistic initialization (agents get take more than 0)
        Q_s = zeros(Float32, n_states, n_messages)
        Q_r = zeros(Float32, n_messages, n_actions)
    elseif q_init == "pessimistic"
        # pessimistic initialization
        Q_s = babbling_reward_s * ones(Float32, n_states, n_messages)
        Q_r = babbling_reward_r * ones(Float32, n_messages, n_actions)
    end
    return Q_s, Q_r
end

function get_action(policy::Array{Float32,2}, Q::Array{Float32,2}, temp::Float32, state::Int64, rng::MersenneTwister)
    # get action according to softmax distribution and update policy by reference
    # ∀c∈R softmax_i(x+c) = exp(x_i+c)/sum_i{exp(x_i+c)} = exp(x_i)/sum_i{exp(x_i)} = softmax_i(x)
    # exp.(x - max(x)) ensures softmax is numerically stable
    # warning: might result in NaN if temp gets extremely small
    n_states, n_actions = size(policy)
    @fastmath for state in 1:n_states
        cum_sum = 0.0f0
        max_val = maximum_(view(Q,state,:))
        @fastmath for action in 1:n_actions
            policy[state, action] = exp((Q[state,action] - max_val) / temp)
            cum_sum += policy[state, action]
        end
        @fastmath for action in 1:n_actions
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
    # value iteration:  Q(s,a) <-  (1 - alpha) * Q[state,action] + alpha * reward 
    @fastmath Q[state,action] += alpha * (reward - Q[state,action])
    return Q
end

# reward of the agents

function reward(a::Int, t::Int, b::Float32, loss_type::String)
    if loss_type == "quadratic"
        return -abs2(A[a] - T[t] - b)
    elseif loss_type == "absolute"
        return -abs(A[a] - T[t] - b)
    elseif loss_type == "fourth"
        return -(A[a] - T[t] - b)^4
    else
        error("Invalid loss type specified")
    end
end

function gen_reward_matrix()
    reward_matrix_s = Array{Float32,2}(undef, n_actions, n_states)
    reward_matrix_r = Array{Float32,2}(undef, n_actions, n_states)
    for t in 1:n_states, a in 1:n_actions
        reward_matrix_s[a,t] = reward(a,t,bias,loss_type)
        reward_matrix_r[a,t] = reward(a,t,0.0f0,loss_type)
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

function gen_distribution()
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

function norm_(A::Array{Float32,2})
    # fast l2 norm()
    norm_A = 0.0f0
    @fastmath for i in eachindex(A)
        norm_A += abs2(A[i])
    end
    return sqrt(norm_A)
end

function is_approx(A::Array{Float32,2}, A_::Array{Float32,2}; tol::Float32 = rtol)
    # fast isapprox(), discussion at https://discourse.julialang.org/t/faster-isapprox/101202/8
    norm_A, norm_A_, norm_diff = 0.0f0, 0.0f0, 0.0f0
    @fastmath for i in eachindex(A)
        a, a_ = A[i], A_[i]
        norm_A += abs2(a)
        norm_A_ += abs2(a_)
        norm_diff += abs2(a - a_)
    end
    return norm_diff < abs2(tol) * max(norm_A, norm_A_)
end

function argmax_(A::AbstractArray{Float32,1}; idxs = Array{Int64,1}(undef,length(A)), tol = 0.0f0)
    # fast argmax(), discussion at https://discourse.julialang.org/t/how-to-efficiently-find-the-set-of-maxima-of-an-array/73423/5
    max_val = -Inf32
    n = 0
    @fastmath for i in eachindex(A)
        a = A[i]
        a < max_val - tol && continue
        if a > max_val + tol
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
    # fast maximum(), related to discussion at https://discourse.julialang.org/t/how-to-efficiently-find-the-set-of-maxima-of-an-array/73423/5  
    max_val = -Inf32
    @fastmath for i in eachindex(A)
        a = A[i]
        a < max_val && continue
        if a > max_val
            max_val = a
        end
    end
    return max_val
end

function sample_(rng::AbstractRNG, wv::AbstractArray{Float32,1})
    # fast weighted sampling, same as in StatsBase.jl w/out probability weights
    # weights wv need to sum to 1.0 (no checks in place)
    t = rand(rng)
    n = length(wv)
    i = 1
    cw = wv[1]
    @fastmath while cw < t && i < n
        cw += wv[i+=1]
    end
    return i
end
