
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
    posterior_mean_variance = Array{Float32,1}(undef, n_simulations)
    optimal_reward_s = Array{Float32,1}(undef, n_simulations)
    optimal_reward_r = Array{Float32,1}(undef, n_simulations)
    posterior = Array{Float32,3}(undef, n_states, n_messages, n_simulations)
    off_path_messages = Array{Bool,2}(undef, n_messages, n_simulations)
    n_off_path_messages = Array{Int64,1}(undef, n_simulations)
    mass_on_suboptim_s = Array{Float32,2}(undef, n_states, n_simulations)
    mass_on_suboptim_r = Array{Float32,2}(undef, n_messages, n_simulations)
    max_mass_on_suboptim_s = Array{Float32,1}(undef, n_simulations)
    max_mass_on_suboptim_r = Array{Float32,1}(undef, n_simulations)
    max_mass_on_suboptim = Array{Float32,1}(undef, n_simulations)
    is_partitional = Array{Bool,1}(undef, n_simulations)
    is_converged = Array{Bool,1}(undef, n_simulations)
    is_nash = Array{Bool,1}(undef, n_simulations)
    is_greedy_s = Array{Bool,1}(undef, n_simulations)
    is_greedy_r = Array{Bool,1}(undef, n_simulations)

    Threads.@threads for z in 1:n_simulations
        # get policies at convergence
        policy_s[:,:,z] = get_policy(Q_s[:,:,z], 1f-30)
        policy_r[:,:,z] = get_policy(Q_r[:,:,z], 1f-30)
        # order Q-matrices and policies
        order = get_order(policy_s[:,:,z])
        Q_s[:,:,z] = Q_s[:,order,z]
        Q_r[:,:,z] = Q_r[order,:,z]
        policy_s[:,:,z] = policy_s[:,order,z]
        policy_r[:,:,z] = policy_r[order,:,z]
        # local aliases avoid repeated slicing allocations
        Q_s_, Q_r_ = Q_s[:,:,z], Q_r[:,:,z]
        policy_s_, policy_r_ = policy_s[:,:,z], policy_r[:,:,z]
        # get true q-matrices
        q_s[:,:,z] = get_q_s(policy_r_)
        q_r[:,:,z] = get_q_r(policy_s_)
        # get margin estimation error
        margin_error_s[:,z] = get_Q_margin(Q_s_) - get_Q_margin(q_s[:,:,z])
        margin_error_r[:,z] = get_Q_margin(Q_r_) - get_Q_margin(q_r[:,:,z])
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
        # compute informativeness metric
        posterior_mean_variance[z] = get_posterior_mean_variance(policy_s_)
        # compute theoretical posterior belief 
        posterior[:,:,z] = get_posterior(policy_s_)
        # get off path messages
        off_path_messages[:,z] = get_off_path_messages(policy_s_)
        # count number of messages that are off-path
        n_off_path_messages[z] = count(off_path_messages[:,z])
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
	is_greedy_s[z] = is_greedy(Q_s_, expl_s[n_episodes[z]])
	is_greedy_r[z] = is_greedy(Q_r_, expl_r[n_episodes[z]])
    end

    # convert results to dict
    results = (Q_s, Q_r, policy_s, policy_r, q_s, q_r, margin_error_s, margin_error_r, induced_actions, expected_reward_s, expected_reward_r, 
                optimal_reward_s, optimal_reward_r, absolute_error_s, absolute_error_r, posterior_mean_variance, posterior, 
                babbling_reward_s, babbling_reward_r, off_path_messages, n_off_path_messages, mass_on_suboptim_s, mass_on_suboptim_r,
                max_mass_on_suboptim_s, max_mass_on_suboptim_r, max_mass_on_suboptim, is_nash, is_partitional, is_converged, is_greedy_s, is_greedy_r)

    var_names = @names(Q_s, Q_r, policy_s, policy_r, q_s, q_r, margin_error_s, margin_error_r, induced_actions, expected_reward_s, expected_reward_r,
                optimal_reward_s, optimal_reward_r, absolute_error_s, absolute_error_r, posterior_mean_variance, posterior,
                babbling_reward_s, babbling_reward_r, off_path_messages, n_off_path_messages, mass_on_suboptim_s, mass_on_suboptim_r,
                max_mass_on_suboptim_s, max_mass_on_suboptim_r, max_mass_on_suboptim, is_nash, is_partitional, is_converged, is_greedy_s, is_greedy_r)
               
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


# rewards and informativeness

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

function get_posterior_mean_variance(policy_s::Array{Float32,2})
    # compute normalized variance of the posterior mean V(E[T|M]) = V(T) - E[V(T|M)]
    @fastmath e_t = p_t'T
    @fastmath v_t = sum(p_t[t] * (T[t]-e_t)^2 for t in 1:n_states)
    @fastmath p_tm = get_posterior(policy_s)
    @fastmath p_m = policy_s'p_t
    @fastmath e_tm = sum(p_tm[t,:]*T[t] for t in 1:n_states)
    return @fastmath (v_t - sum(p_t[t] * policy_s[t,m] * (T[t] - e_tm[m])^2 for m in 1:n_messages, t in 1:n_states if p_m[m] != 0)) / v_t
end

# policy

function get_order(policy_s::Array{Float32,2})
    # returns permutation that sorts messages by posterior mean (off-path messages at the end)
    p_m = @fastmath policy_s' * p_t
    posterior_mean = fill(Inf32, n_messages)
    @fastmath for m in 1:n_messages
        p_m[m] <= ptol && continue
        posterior_mean[m] = 0f0
        for t in 1:n_states
            posterior_mean[m] += p_t[t] * policy_s[t,m] * T[t]
        end
        posterior_mean[m] /= p_m[m]
    end
    return sortperm(posterior_mean)
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

function is_greedy(Q::Array{Float32,2}, expl::Float32)
    # check policy at convergence temperature/exploration is greedy with respect to Q
    policy = get_policy(Q, expl)
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

