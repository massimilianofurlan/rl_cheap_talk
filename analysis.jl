
# convergence analysis
function convergence_analysis(Q_s, Q_r, n_episodes)
    # analyze experiments at convergence 

    # no analysis in raw mode 
    dict_input = Dict(name => value for (name, value) in zip(@names(Q_s, Q_r, n_episodes), (Q_s, Q_r, n_episodes)))
    raw && return dict_input

    # preallocate arrays
    policy_s = Array{Float32,3}(undef, n_states, n_messages, n_simulations)
    policy_r = Array{Float32,3}(undef, n_messages, n_actions, n_simulations)
    q_s = Array{Float32,3}(undef, n_states, n_messages, n_simulations)
    q_r = Array{Float32,3}(undef, n_messages, n_actions, n_simulations)
    margin_error_s =  Array{Float32,2}(undef, n_states, n_simulations)
    margin_error_r =  Array{Float32,2}(undef, n_messages, n_simulations)
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
    is_absorbing = Array{Bool,1}(undef, n_simulations)
    is_greedy_s = Array{Bool,1}(undef, n_simulations)
    is_greedy_r = Array{Bool,1}(undef, n_simulations)

    Threads.@threads for z in 1:n_simulations
        # order Q-matrices
        order = get_order(Q_s[:,:,z])
        Q_s[:,:,z] = Q_s[:,order,z]
        Q_r[:,:,z] = Q_r[order,:,z]
        # get policies at convergence
        policy_s[:,:,z] = get_policy(Q_s[:,:,z], expl_s[n_episodes[z]])
        policy_r[:,:,z] = get_policy(Q_r[:,:,z], expl_r[n_episodes[z]])
        policy_s_ = policy_s[:,:,z]
        policy_r_ = policy_r[:,:,z]
        # get true q-matrices
        q_s[:,:,z] = get_q_s(policy_r_)
        q_r[:,:,z] = get_q_r(policy_s_)
        # get margin estimation error
        margin_error_s[:,z] = get_Q_margin(Q_s[:,:,z]) - get_Q_margin(q_s[:,:,z])
        margin_error_r[:,z] = get_Q_margin(Q_r[:,:,z]) - get_Q_margin(q_r[:,:,z])
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
        absolute_error_s[z] = optimal_reward_s[z] - expected_reward_s[z]
        absolute_error_r[z] = optimal_reward_r[z] - expected_reward_r[z]
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
        # check if converged policies are greedy wrt converged Q-values
        is_greedy_s[z] = is_greedy(Q_s[:,:,z], policy_s_)
        is_greedy_r[z] = is_greedy(Q_r[:,:,z], policy_r_)
        # check if Q_s,Q_r induce self-confirming policies
        is_absorbing[z] = is_greedy_absorbing(Q_s[:,:,z], Q_r[:,:,z])
    end

    # convert results to dict
    results = (Q_s, Q_r, policy_s, policy_r, q_s, q_r, margin_error_s, margin_error_r, induced_actions, expected_reward_s, expected_reward_r, 
                optimal_reward_s, optimal_reward_r, absolute_error_s, absolute_error_r, mutual_information, residual_variance, posterior, 
                babbling_reward_s, babbling_reward_r, off_path_messages, n_off_path_messages, n_effective_messages, mass_on_suboptim_s, mass_on_suboptim_r, 
                max_mass_on_suboptim_s, max_mass_on_suboptim_r, max_mass_on_suboptim, is_nash, is_partitional, is_converged, is_absorbing, is_greedy_s, is_greedy_r)
                
    var_names = @names(Q_s, Q_r, policy_s, policy_r, q_s, q_r, margin_error_s, margin_error_r, induced_actions, expected_reward_s, expected_reward_r, 
                optimal_reward_s, optimal_reward_r, absolute_error_s, absolute_error_r, mutual_information, residual_variance, posterior, 
                babbling_reward_s, babbling_reward_r, off_path_messages, n_off_path_messages, n_effective_messages, mass_on_suboptim_s, mass_on_suboptim_r, 
                max_mass_on_suboptim_s, max_mass_on_suboptim_r, max_mass_on_suboptim, is_nash, is_partitional, is_converged, is_absorbing, is_greedy_s, is_greedy_r)
               
    dict_results = Dict(name => value for (name, value) in zip(var_names, results))
    return merge(dict_input,dict_results)
end


# compute true Q-matrices and delta

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

function get_Q_margin(Q::AbstractMatrix{Float32})
    # margin[s] = max(Q[s,:]) - max{ x < max(Q[s,:]) } ; if all equal then return 0
    margin = zeros(Float32, size(Q,1))
    for s in 1:size(Q,1)
        best = -Inf32
        for q in @view Q[s, :]
            q > best && (best = q)
        end
        runnerup = -Inf32
        for q in @view Q[s, :]
            (q < best && q > runnerup) && (runnerup = q)
        end
        margin[s] = isfinite(runnerup) ? (best - runnerup) : 0f0
    end
    return margin
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

function get_mutual_information(policy_s::Array{Float32,2})
    # compute normalized mutual information between m and t 
    # marginal probability of receiving message m, p(m) = \sum_t p(m|t)p(t)
    @fastmath p_m = policy_s'p_t
    return @fastmath sum(policy_s[t,m]*p_t[t] * log2(policy_s[t,m]/p_m[m]) for t in 1:n_states, m in 1:n_messages if policy_s[t,m] != 0) / (-p_t' * log2.(p_t)) 
end

function get_residual_variance(policy_s::Array{Float32,2})
    # compute residual variance
    # expected value the state, E(t) = \sum_t p(t) t
    @fastmath e_t = p_t'T
    # variance of the state, V(t) = \sum_t p(t) (t - E(t))^2
    @fastmath v_t = sum(p_t[t] * (T[t]-e_t)^2 for t in 1:n_states)
    # conditional probability of being in state t given message m
    p_tm = get_posterior(policy_s)
    # marginal probability of receiving message m, p(m) = \sum_t p(m|t)p(t)
    @fastmath p_m = policy_s'p_t
    # expected value of the state conditional on m, E(t|m) = \sum_t p(t|m) t
    @fastmath e_tm = sum(p_tm[t,:]*T[t] for t in 1:n_states)
    # variance of the state conditional on m, V(t|m) = \sum_t p(t|m) (t - E(t|m))^2
    @fastmath v_tm = sum(p_tm[t,:] .* (T[t] .- e_tm).^2 for t in 1:n_states)
    # normalized expected ex-ante variance of the state conditional on m
    return @fastmath sum(p_m[m] * v_tm[m] for m in 1:n_messages if p_m[m] != 0) / v_t
end

# policy

function get_order(Q_s::Array{Float32,2})
    # returns a permutation of messages that associates higher messages with higher states as much as possible
    order = collect(1:n_messages)
    m_ = 1
    for t in 1:n_states
        set_m = argmax_(Q_s[t,:])                   # index of most frequently sent message in state t
        length(set_m) < n_messages || continue      # skip (inconsequential) reordering if all messages are synonym
        for m in set_m
            m < m_ && continue
            temp = order[m_]
            order[m_] = order[m]
            order[m] = temp
            m_ += 1
        end
    end
    return order
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

function is_greedy(Q::Array{Float32,2}, policy::Array{Float32,2})
    # check policy is greedy with respect to Q
    n_states, n_actions = size(Q)
    for state in 1:n_states
        max_val = maximum_(view(Q, state, :))
        for action in 1:n_actions
            if policy[state, action] > ptol
                abs(Q[state, action] - max_val) <= 1f-6 || return false
            end
        end
    end
    return true
end

function is_greedy_absorbing(Q_s::Array{Float32,2}, Q_r::Array{Float32,2})
    # necessary and sufficient for absorption: greedy argmax-sets are self-confirming given (Q_s,Q_r)
    # assumes π(·|s) to be greedy with uniform tie-breaking from Q(s,·)
    # for each agent checks that min{ min{supp r(s,a)} : a∈supp(π(·|s)) } >= max{ Q(s,b) : b∉supp(π(·|s)) }
    # and if |supp(π(·|s))|>1 then r(s,a) is constant over a∈supp(π(·|s)) and Q(s,a) equals that constant
    supp_policy_s = get_epsgreedy_policy(Q_s, 1f-7) .> ptol
    supp_policy_r = get_epsgreedy_policy(Q_r, 1f-7) .> ptol
    # sender
    for t in 1:n_states
        off_path = .!supp_policy_s[t, :]
        best_off_path_Q = any(off_path) ? maximum_(Q_s[t, off_path]) : -Inf32
        if count(supp_policy_s[t, :]) == 1
            for m in 1:n_messages
                supp_policy_s[t, m] || continue
                reward_supp = reward_matrix_s[supp_policy_r[m, :], t]
                worst_on_path = -maximum_(-reward_supp)
                worst_on_path >= best_off_path_Q - 1f-6 || return false
            end
        else
            best_on_path, worst_on_path = -Inf32, Inf32
            for m in 1:n_messages
                supp_policy_s[t, m] || continue
                reward_supp = reward_matrix_s[supp_policy_r[m, :], t]
                best_on_path = max(best_on_path, maximum_(reward_supp))
                worst_on_path = min(worst_on_path, -maximum_(-reward_supp))
            end
            best_on_path - worst_on_path <= 1f-6 || return false
            for m in 1:n_messages
                supp_policy_s[t, m] || continue
                abs(Q_s[t, m] - best_on_path) <= 1f-6 || return false
            end
        end
    end
    # receiver
    for m in 1:n_messages
        any(supp_policy_s[:, m]) || continue
        off_path = .!supp_policy_r[m, :]
        best_off_path_Q = any(off_path) ? maximum_(Q_r[m, off_path]) : -Inf32
        if count(supp_policy_r[m, :]) == 1
            for a in 1:n_actions
                supp_policy_r[m, a] || continue
                reward_supp = reward_matrix_r[a, supp_policy_s[:, m]]
                worst_on_path = -maximum_(-reward_supp)
                worst_on_path >= best_off_path_Q - 1f-6 || return false
            end
        else
            best_on_path, worst_on_path = -Inf32, Inf32
            for a in 1:n_actions
                supp_policy_r[m, a] || continue
                reward_supp = reward_matrix_r[a, supp_policy_s[:, m]]
                best_on_path = max(best_on_path, maximum_(reward_supp))
                worst_on_path = min(worst_on_path, -maximum_(-reward_supp))
            end
            best_on_path - worst_on_path <= 1f-6 || return false
            for a in 1:n_actions
                supp_policy_r[m, a] || continue
                abs(Q_r[m, a] - best_on_path) <= 1f-6 || return false
            end
        end
    end
    return true
end

#=
function get_off_path_message_action_pairs(policy_s::Array{Float32,2}, policy_r::Array{Float32,2})
    # get off path messages and their induced actions. is_wlog if off path actions are induced in equilibrium
    off_path_messages = (p_t'*policy_s)' .< 1f-6  #iszero.((p_t'*policy_s)')
    off_path_induced_actions = policy_r[off_path_messages,:] .> 1f-6
    is_wlog = issubset(off_path_induced_actions, policy_r[.!off_path_messages,:] .> 1f-6)
    return off_path_messages, off_path_induced_actions, is_wlog
end
=#
