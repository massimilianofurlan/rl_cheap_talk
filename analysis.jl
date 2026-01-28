
# convergence analysis
function convergence_analysis(Q_s, Q_r, n_episodes)
    # analyze experiments at convergence 

    # no analysis in raw mode 
    dict_input = Dict(name => value for (name, value) in zip(@names(Q_s, Q_r, n_episodes), (Q_s, Q_r, n_episodes)))
    raw && return dict_input

    # preallocate arrays
    policy_s = Array{Float32,3}(undef, n_states, n_messages, n_simulations)
    policy_r = Array{Float32,3}(undef, n_messages, n_actions, n_simulations)
    induced_actions = Array{Float32,3}(undef, n_states, n_actions, n_simulations)
    expected_reward_s = Array{Float32,1}(undef, n_simulations)
    expected_reward_r = Array{Float32,1}(undef, n_simulations)
    absolute_error_s = Array{Float32,1}(undef, n_simulations)
    absolute_error_r = Array{Float32,1}(undef, n_simulations)
    mutual_information = Array{Float32,1}(undef, n_simulations)
    residual_variance = Array{Float32,1}(undef, n_simulations)
    optimal_reward_s = Array{Float32,1}(undef, n_simulations)
    optimal_reward_r = Array{Float32,1}(undef, n_simulations)
    posterior = Array{Float32,3}(undef, n_states, n_messages, n_simulations)
    off_path_messages = Array{Bool,2}(undef, n_messages, n_simulations)
    n_off_path_messages = Array{Int64,1}(undef, n_simulations)
    n_effective_messages = Array{Int64,1}(undef, n_simulations)
    mass_on_suboptim_s = Array{Float32,2}(undef, n_states, n_simulations)
    mass_on_suboptim_r = Array{Float32,2}(undef, n_messages, n_simulations)
    max_mass_on_suboptim_s = Array{Float32,1}(undef, n_simulations)
    max_mass_on_suboptim_r = Array{Float32,1}(undef, n_simulations)
    max_mass_on_suboptim = Array{Float32,1}(undef, n_simulations)
    is_partitional = Array{Bool,1}(undef, n_simulations)
    is_converged = Array{Bool,1}(undef, n_simulations)
    is_nash = Array{Bool,1}(undef, n_simulations)

    Threads.@threads for z in 1:n_simulations
        # get policies at convergence
        policy_s[:,:,z] = get_policy(Q_s[:,:,z], temp_s[n_episodes[z]])
        #policy_r[:,:,z] = get_best_reply_r(policy_s[:,:,z])
        policy_r[:,:,z] = get_policy(Q_r[:,:,z], temp_r[n_episodes[z]])
        policy_s_ = policy_s[:,:,z]
        policy_r_ = policy_r[:,:,z]
        # compute induced actions at convergence
        induced_actions[:,:,z] = get_induced_actions(policy_s_, policy_r_)
        # compute (ex-ante) expected rewards at convergence
        expected_reward_s[z], expected_reward_r[z] = get_expected_rewards(policy_s_, policy_r_)
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
        residual_variance[z] = get_residual_variance(policy_s_)
        # compute theoretical posterior belief 
        posterior[:,:,z] = get_posterior(policy_s_)
        # get off path messages
        off_path_messages[:,z] = get_off_path_messages(policy_s_)
        # count number of messages that are off-path
        n_off_path_messages[z] = count(off_path_messages[:,z])
        # count number of messages that have no synonyms
        n_effective_messages[z] = count(get_effective_messages(policy_s_[:,.!off_path_messages[:,z]]))
        # compute mass on suboptim messages (actions) for each state (message)
        mass_on_suboptim_s[:,z] = get_mass_on_suboptim(policy_s_, optimal_policy_s)
        mass_on_suboptim_r[:,z] = get_mass_on_suboptim(policy_r_, optimal_policy_r)
        # check if is a γ-nash
        max_mass_on_suboptim_s[z] = maximum_(mass_on_suboptim_s[:,z])
        max_mass_on_suboptim_r[z] = maximum_(mass_on_suboptim_r[.!off_path_messages[:,z],z])
        max_mass_on_suboptim[z] = max(max_mass_on_suboptim_s[z], max_mass_on_suboptim_r[z])
        is_nash[z] = max_mass_on_suboptim[z] < gtol
        # check if policy is partitional
        is_partitional[z] = ispartitional(policy_s_)
        # check if agents have converged 
        is_converged[z] = n_episodes[z] < n_max_episodes
    end

    # convert results to dict
    results = (policy_s, policy_r, induced_actions, expected_reward_s, expected_reward_r, optimal_reward_s, optimal_reward_r,
                absolute_error_s, absolute_error_r, mutual_information, residual_variance, posterior, babbling_reward_s, babbling_reward_r,
                off_path_messages, n_off_path_messages, n_effective_messages, mass_on_suboptim_s, mass_on_suboptim_r, 
                max_mass_on_suboptim_s, max_mass_on_suboptim_r, max_mass_on_suboptim, is_nash, is_partitional, is_converged)
                
    var_names = @names(policy_s, policy_r, induced_actions, expected_reward_s, expected_reward_r, optimal_reward_s, optimal_reward_r,
                absolute_error_s, absolute_error_r, mutual_information, residual_variance, posterior, babbling_reward_s, babbling_reward_r,
                off_path_messages, n_off_path_messages, n_effective_messages, mass_on_suboptim_s, mass_on_suboptim_r, 
                max_mass_on_suboptim_s, max_mass_on_suboptim_r, max_mass_on_suboptim, is_nash, is_partitional, is_converged)
               
    dict_results = Dict(name => value for (name, value) in zip(var_names, results))
    return merge(dict_input,dict_results)
end


# compute theoretical Q-matrices and Q-loss

function get_q_s(policy_r::Array{Float32,2})
    # compute theoretical Q-matrix of the sender
    return @fastmath Matrix{Float32}((policy_r*reward_matrix_s)')
end

function get_q_r(policy_s::Array{Float32,2}; opb = p_t)
    # compute theoretical Q-matrix of the receiver
    # conditional probability of being in state t given message m
    p_tm = get_posterior(policy_s)
    # off-path belief coincide with opb (default is prior)
    off_path_messages = get_off_path_messages(policy_s, tol = 1f-6)
    p_tm[:,off_path_messages] .= opb
    return @fastmath Matrix{Float32}((reward_matrix_r * p_tm)')
end


# best response functions

function get_best_reply_r(policy_s::Array{Float32,2}; opb = p_t)
    # get best reply to sender's policy (default off-path belief is prior)
    q_r = get_q_r(policy_s; opb = opb)
    best_replies = argmax_.(q_r[m,:] for m in 1:n_messages; tol=1f-7) # precison up to 1f-7 to catch all indifferences 
    return convert_policy(best_replies, n_actions)
end

function get_best_reply_s(policy_r::Array{Float32,2})
    # get best reply to receiver's policy   
    q_s = get_q_s(policy_r)
    best_replies =  argmax_.(q_s[t,:] for t in 1:n_states; tol=1f-7)  # precison up to 1f-7 to catch all indifferences
    return convert_policy(best_replies, n_messages)
end

function convert_policy(best_replies::Array, n_actions::Int64)
    # convert set of pure best replies to an equivalent stochastic policy with full support over pure best replies
    policy = zeros(Float32, length(best_replies), n_actions)
    @fastmath for state in 1:length(best_replies)
        actions = best_replies[state]
        policy[state, actions] .= 1.0f0 / length(actions)   # randomize uniformly over optimal actions
    end
    return policy
end


# induced actions

function get_induced_actions(policy_s::Array{Float32,2}, policy_r::Array{Float32,2})
    # get distribution of induced actions given policy_s and policy_r
    return @fastmath policy_s * policy_r
end


# rewards and mutual information

function get_expected_rewards(policy_s::Array{Float32,2}, policy_r::Array{Float32,2})
    # get on the path rewards given policy_s and policy_r
    induced_actions = get_induced_actions(policy_s, policy_r)
    @fastmath reward_s = p_t'*sum(induced_actions'.*reward_matrix_s, dims=1)[:]
    @fastmath reward_r = p_t'*sum(induced_actions'.*reward_matrix_r, dims=1)[:]
    return reward_s, reward_r
end

function get_expected_rewards(induced_actions::Array{Float32,2})
    # get on the path rewards given induced actions
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

function get_residual_variance(policy::Array{Float32,2})
    # compute residual variance
    # expected value the state, E(t) = \sum_t p(t) t
    @fastmath e_t = p_t'T
    # variance of the state, V(t) = \sum_t p(t) (t - E(t))^2
    @fastmath v_t = sum(p_t[t] * (T[t]-e_t)^2 for t in 1:n_states)
    # conditional probability of being in state t given message m
    p_tm = get_posterior(policy)
    # marginal probability of receiving message m, p(m) = \sum_t p(m|t)p(t)
    @fastmath p_m = policy'p_t
    # expected value of the state conditional on m, E(t|m) = \sum_t p(t|m) t
    @fastmath e_tm = sum(p_tm[t,:]*T[t] for t in 1:n_states)
    # variance of the state conditional on m, V(t|m) = \sum_t p(t|m) (t - E(t|m))^2
    @fastmath v_tm = sum(p_tm[t,:] .* (T[t] .- e_tm).^2 for t in 1:n_states)
    # normalized expected ex-ante variance of the state conditional on m
    return @fastmath sum(p_m[m] * v_tm[m] for m in 1:n_messages if p_m[m] != 0) / v_t
end

# policy

function get_policy(Q::Array{Float32,2}, temp::Float32)
    # derive soft-max policy from Q-matrix
    policy = similar(Q)
    @fastmath for state in 1:size(Q,1)
        max_val = maximum_(view(Q,state,:))
        policy[state,:] = exp.((Q[state,:].-max_val)/temp)/sum(exp.((Q[state,:].-max_val)/temp))
    end
    return policy
end

function order_policies(policy_s, policy_r)
    # reorder messages so that lower message indeces are associated to lower states 
    m_= 1
    for t in 1:n_states 
        set_m = argmax_(policy_s[t,:])              # index of most frequently sent message in state t
        length(set_m) < n_messages || continue      # skip (inconsequential) reordering if all messages are synonym
        for m in set_m  
            m < m_ && continue                      # if m was not yet assigned a position, swap m with m_
            temp = policy_s[:,m_]
            policy_s[:,m_] = policy_s[:,m]
            policy_s[:,m] = temp
            temp = policy_r[m_,:]
            policy_r[m_,:] = policy_r[m,:]
            policy_r[m,:] = temp
            m_ += 1
        end
    end
    return policy_s, policy_r
end


# policy analysis

get_posterior(policy::Array{Float32,2}) = @fastmath p_t .* policy ./ (p_t'*policy)                              # posterior beliefs following each message
get_off_path_messages(policy_s::Array{Float32,2}; tol::Float32 = ptol) = @fastmath (p_t'*policy_s)' .<= tol     # bitmap off-path messages

function get_mass_on_suboptim(policy::Array{Float32,2}, optimal_policy::Array{Float32,2})
    # compute probability mass on suboptim actions for each states
    suboptim_bitmap = (policy .> 0) .& .!(optimal_policy .> 0)
    return @fastmath sum(suboptim_bitmap .* policy, dims=2)
end

function ispartitional(policy_s::Array{Float32,2}; tol::Float32 = ptol)
    # check if policy of the sender is partitional 
    supp_policy_s = (policy_s .> tol)
    @fastmath for message in 1:n_messages-1
        for message_ in message+1:n_messages
            states = supp_policy_s[:,message] .|| supp_policy_s[:,message_]
            flags = xor.(supp_policy_s[states,message],supp_policy_s[states,message_])
            (all(flags) || all(.!flags)) || return false
        end
    end
    return true
end

function get_effective_messages(policy_s::Array{Float32,2}; tol::Float32 = ptol)
    # bitmap of messages that have no synonyms
    n_on_path_messages = size(policy_s,2)
    has_no_synonyms = trues(n_on_path_messages)
    @fastmath for message1 in 1:n_on_path_messages-1
        has_no_synonyms[message1] || continue
        for message2 in message1+1:n_on_path_messages
            all(abs.(policy_s[:, message1] - policy_s[:, message2]) .< tol) || continue
            has_no_synonyms[message2] = false
        end
    end
    return has_no_synonyms
end


#=
function get_off_path_message_action_pairs(policy_s::Array{Float32,2}, policy_r::Array{Float32,2})
    # get off path messages and their induced actions. is_wlog if off path actions are induced in equilibrium
    off_path_messages = (p_t'*policy_s)' .< 1f-6  #iszero.((p_t'*policy_s)')
    off_path_induced_actions = policy_r[off_path_messages,:] .> 1f-6
    is_wlog = issubset(off_path_induced_actions, policy_r[.!off_path_messages,:] .> 1f-6)
    return off_path_messages, off_path_induced_actions, is_wlog
end

function isfixedpoint(Q_s::Array{Float32,2}, Q_r::Array{Float32,2}, policy_s::Array{Float32,2}, policy_r::Array{Float32,2}, induced_actions::Array{Float32,2})
    # is fixed point Q-system
    supp_s, supp_r =  policy_s .> ptol, policy_r .> ptol
    flag_s, flag_r = trues(n_messages), trues(n_actions)
    for t in 1:n_states
        for m in 1:n_messages
            supp_s[t,m] == true || continue
            for a in 1:n_actions
                supp_r[m,a] == true || continue
                # sender
                flag_s[m] *= abs(Q_s[t,m]-reward_matrix_s[a,t]) < 1f-6
                # receiver
                flag_r[a] *= abs(Q_r[m,a]-reward_matrix_r[a,t]) < 1f-6
            end
        end
    end
    return all(flag_s)*all(flag_r)
end

=#