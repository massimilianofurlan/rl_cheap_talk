
# convergence analysis

function convergence_analysis(Q_s, Q_r, n_episodes, n_conv_diff)
    # analyze experiments at convergence 

    # no analysis in raw mode 
    dict_input = Dict(name => value for (name, value) in zip(@names(Q_s, Q_r, n_episodes, n_conv_diff), (Q_s, Q_r, n_episodes, n_conv_diff)))
    raw && return dict_input

    # preallocate arrays
    policy_s = Array{Float32,3}(undef, n_states, n_messages, n_simulations)
    policy_r = Array{Float32,3}(undef, n_messages, n_actions, n_simulations)
    induced_actions = Array{Float32,3}(undef, n_states, n_actions, n_simulations)
    expected_reward_s = Array{Float32,1}(undef, n_simulations)
    expected_reward_r = Array{Float32,1}(undef, n_simulations)
    expected_aggregate_reward = Array{Float32,1}(undef, n_simulations)
    absolute_error_s = Array{Float32,1}(undef, n_simulations)
    absolute_error_r = Array{Float32,1}(undef, n_simulations)
    mutual_information = Array{Float32,1}(undef, n_simulations)
    optimal_reward_s = Array{Float32,1}(undef, n_simulations)
    optimal_reward_r = Array{Float32,1}(undef, n_simulations)
    posterior = Array{Float32,3}(undef, n_states, n_messages, n_simulations)

    Threads.@threads for z in 1:n_simulations
        # get policies at convergence
        policy_s[:,:,z] = get_policy(Q_s[:,:,z], temp_s[n_episodes[z]])
        policy_r[:,:,z] = get_policy(Q_r[:,:,z], temp_r[n_episodes[z]])
        policy_s_ = policy_s[:,:,z]
        policy_r_ = policy_r[:,:,z]
        # compute induced actions at convergence
        induced_actions[:,:,z] = get_induced_actions(policy_s_, policy_r_)
        # compute (ex-ante) expected rewards at convergence
        expected_reward_s[z], expected_reward_r[z] = get_expected_rewards(policy_s_, policy_r_)
        expected_aggregate_reward[z] = expected_reward_s[z] + expected_reward_r[z]  
        # compute best response to opponent's policy at convergence
        optimal_policy_s = get_best_reply_s(policy_r_)
        optimal_policy_r = get_best_reply_r(policy_s_)
        # compute expected rewards by best responding to opponent
        optimal_reward_s[z], _ = get_expected_rewards(optimal_policy_s, policy_r_)
        _, optimal_reward_r[z] = get_expected_rewards(policy_s_, optimal_policy_r)
        # compute absolute expected error by (possibly) not best responding to opponent
        absolute_error_s[z] = expected_reward_s[z] - optimal_reward_s[z]
        absolute_error_r[z] = expected_reward_r[z] - optimal_reward_r[z]
        # compute communication metrics
        mutual_information[z] = get_mutual_information(policy_s_)
        # compute theoretical posterior belief 
        posterior[:,:,z] = get_posterior(policy_s_)
    end

    # convert results to dict
    results = (policy_s, policy_r, induced_actions, expected_reward_s, expected_reward_r, expected_aggregate_reward, 
                optimal_reward_s, optimal_reward_r, absolute_error_s, absolute_error_r, mutual_information, posterior,
                babbling_reward_s, babbling_reward_r)
    var_names = @names(policy_s, policy_r, induced_actions, expected_reward_s, expected_reward_r, expected_aggregate_reward, 
                optimal_reward_s, optimal_reward_r, absolute_error_s, absolute_error_r, mutual_information, posterior, 
                babbling_reward_s, babbling_reward_r)
    dict_results = Dict(name => value for (name, value) in zip(var_names, results))
    return merge(dict_input,dict_results)
end



# posterior beliefs

get_posterior(policy::Array{Float32,2}) = p_t .* policy ./ (p_t'*policy)

# off the path messages and their induced actions
#=
function get_off_path_message_action_pairs(policy_s::Array{Float32,2}, policy_r::Array{Float32,2})
    # get off path messages and their absolute induced actions. is_wlog if off path actions are induced in equilibrium
    off_path_messages = (p_t'*policy_s)' .< 1f-6  #iszero.((p_t'*policy_s)')
    off_path_induced_actions = policy_r[off_path_messages,:] .> 1f-6
    is_wlog = issubset(off_path_induced_actions, policy_r[.!off_path_messages,:] .> 1f-6)
    return off_path_messages, off_path_induced_actions, is_wlog
end
=#

# compute theoretical Q-matrices and Q-loss

function get_q_s(policy_r::Array{Float32,2})
    # compute theoretical Q-matrix of the sender
    return @turbo (policy_r*reward_matrix_s)'
end

function get_q_r(policy_s::Array{Float32,2}; opb = p_t)
    # compute theoretical Q-matrix of the receiver
    # conditional probability of being in state t given message m (bayes update)     
    p_tm = get_posterior(policy_s)
    # off-path belief coincide with opb (default is prior)
    off_path_messages = (p_t'*policy_s)' .< 1f-6
    p_tm[:,off_path_messages] .= opb
    return @turbo (reward_matrix_r * p_tm)'
end

# convert deterministic policy to distribution

function convert_policy(policy_::Array, n_actions::Int64)
    # convert deterministic policy to an equivalent full-support stochastic policy 
    policy = zeros(Float32, length(policy_), n_actions)
    @fastmath @inbounds for state in 1:length(policy_)
        if size(policy_[state]) == ()
            # if best action in state is unique, degenerate distribution
            policy[state,policy_[state]] = 1.0f0
        else
            # if best action in state is not unique, randomize across best actions
            policy[state,policy_[state]] .= 1.0f0 / length(policy_[state])
        end
    end
    return policy
end


# best response functions

function get_best_reply_r(policy_s::Array{Float32,2}; opb = p_t)
    # get best reply to sender's stochastic policy (default off-path belief is prior)
    q_r = get_q_r(policy_s; opb = opb)
    best_reply = argmax_.(q_r[m,:] for m in 1:n_messages)
    return convert_policy(best_reply, n_actions)
end

function get_best_reply_s(policy_r::Array{Float32,2})
    # get best reply to receiver's policy   
    q_s = get_q_s(policy_r)
    best_reply =  argmax_.(q_s[t,:] for t in 1:n_states)
    return convert_policy(best_reply, n_messages)
end


# policies

function get_policy(Q::Array{Float32,2}, temp::Float32)
    # derive soft-max policy from Q-matrix
    policy = similar(Q)
    @fastmath @inbounds for state in 1:size(Q,1)
        max_val = maximum_(view(Q,state,:))
        policy[state,:] = exp.((Q[state,:].-max_val)/temp)/sum(exp.((Q[state,:].-max_val)/temp))
    end
    return policy
end


# induced actions

function get_induced_actions(policy_s::Array{Float32,2}, policy_r::Array{Float32,2})
    # get distribution of induced actions given policy_s and policy_r
    return @turbo policy_s * policy_r
end


# rewards and mutual information

function get_expected_rewards(policy_s::Array{Float32,2}, policy_r::Array{Float32,2})
    # get on the path rewards given policy_s and policy_r
    induced_actions = get_induced_actions(policy_s, policy_r)
    @fastmath reward_s = p_t'*sum(induced_actions'.*reward_matrix_s, dims=1)[:]
    @fastmath reward_r = p_t'*sum(induced_actions'.*reward_matrix_r, dims=1)[:]
    return reward_s, reward_r
end

function get_mutual_information(policy::Array{Float32,2})
    # compute normalized mutual information between m and t 
    # marginal probability of receiving message m, p(m) = \sum_t p(m|t)p(t)
    @fastmath p_m = policy'p_t
    return @fastmath sum(policy[t,m]*p_t[t] * log2(policy[t,m]/p_m[m]) for t in 1:n_states, m in 1:n_messages if policy[t,m] != 0) / (-p_t' * log2.(p_t)) 
end
