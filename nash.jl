
function get_N(bias, n_states)
    # based on Procedure 1 in Frug (2016)
    get_next(x) = max(1, x + ceil((n_states - 1)*4*bias) - 2)
    x = Array{Int,1}(undef, n_states)
    x[1] = 1
    for i in 2:n_states
        x[i] = get_next(x[i-1]) 
        sum(x[1:i]) > n_states && return i-1, x[1:i-1]
    end 
    return n_states, x
end

function convert_I_to_policy(I_, N; policy_s = zeros(Float32, n_states, n_messages))
    # convert I_ into a full support policy for the sender 
    policy_s = zeros(Float32, n_states, n_messages)
    for i in 1:length(I_)-1
        policy_s[sum(I_[1:i-1])+1:sum(I_[1:i]),i] .= 1
    end
    policy_s[sum(I_[1:end-1])+1:sum(I_),length(I_):end] .= 1/(n_messages-N+1)
    return policy_s
end

function get_exante_pareto_optimal()
    # get ex-ante pareto optimal equilibrium, based on Procedure 1 - Frug (2016)
    # only works if there are as many messages as states, the set of actions is enoguh rich, utility is quadratic and the prior is uniform
    N, x = get_N(bias, n_states)
    # residual 
    y = n_states - sum(x)
    # y = q * N + r
    q, r = div(y,N), rem(y,N)
    # y_i
    y_ = fill(q,N)
    y_[N-r+1:N] .= q+1
    # I_i
    I_ = y_ + x
    # converting partition to policy_s
    best_policy_s = convert_I_to_policy(I_, N)
    # get best response to policy_s
    best_policy_r = get_best_reply_r(best_policy_s)
    return best_policy_s, best_policy_r, N
end

function is_best_reply(policy, policy_opponent, get_best_reply)
    # checks if policy is a best reply to policy_opponent
    # best reply to opponent policy
    best_reply = get_best_reply(policy_opponent)
    # compute (bitmap) support of policy and best response to policy_opponent
    supp_policy, supp_best_reply = (policy .> 0), (best_reply .> 0)
    # policy is a best response to policy_opponent iff supp(policy) ⊆ supp(best_reply)
    return all(supp_best_reply - supp_policy .>= 0)
end

function is_exact_nash(policy_s, policy_r)
    # checks if (policy_s, policy_r) is an exact equilibrium
    is_best_reply_s = is_best_reply(policy_s, policy_r, get_best_reply_s)
    is_best_reply_r = is_best_reply(policy_r, policy_s, get_best_reply_r)
    return is_best_reply_s && is_best_reply_r
end

function get_exante_receiver_optimal()
    # get ex-ante receiver optimal equilibrium by scanning over all possible partitional equilibria
    # Frug (2016) shows receiver's optimal equilibrium is partitional
    # if prior is not uniform, returns the ex-ante receiver-preferred equilibrium among partitional equilibria
    policy_s = zeros(Float32, n_states, n_messages, Threads.nthreads());
    policy_r = zeros(Float32, n_messages, n_actions, Threads.nthreads());
    expected_reward_r = fill(-Inf32, Threads.nthreads()); # makes it thread-safe
    n_messages_on_path = zeros(Int64, Threads.nthreads())
    Threads.@threads for i in 2^(n_states-1)-1 : -1 : 0
        thread_idx = Threads.threadid()
        # construct (partitional) policy for the sender
        cut = digits(i, base=2, pad=(n_states-1))                                           # i-th possible partition
        N = count(cut.==1) + 1                                                              # number of messages
        N <= n_messages || continue
        I_ = N > 1 ? findall(cut .== 1) : [n_states]                
        I_ = [i > 1 ? I_[i] - I_[i-1] : I_[i] for i in 1:length(I_)]
        N > 1 && push!(I_,n_states - sum(I_))
        policy_s_ = convert_I_to_policy(I_, N, policy_s = view(policy_s,:,:,thread_idx))    # sender's candidate policy
        policy_r_ = get_best_reply_r(policy_s_)                                             # receiver's candidate policy
        is_best_reply(policy_s_, policy_r_, get_best_reply_s) || continue
        expected_reward_s_, expected_reward_r_ = get_expected_rewards(policy_s_, policy_r_)
        expected_reward_r_ > expected_reward_r[thread_idx] || continue                      # if nash is preferred by receiver continue
        expected_reward_r[thread_idx] = expected_reward_r_
        policy_s[:,:,thread_idx] = copy(policy_s_)
        policy_r[:,:,thread_idx] = copy(policy_r_)
        n_messages_on_path[thread_idx] = N
    end
    best_idx = argmax(expected_reward_r)
    return policy_s[:,:,best_idx], policy_r[:,:,best_idx], n_messages_on_path[best_idx]
end

function get_best_nash()
    best_policy_s, best_policy_r, n_messages_on_path = undef, undef, undef
    if dist_type == "uniform" && loss_type == "quadratic" && n_messages >= n_states && mod(n_actions - 1, n_states - 1) == 0 && div(n_actions - 1, n_states - 1) > 1 
        best_policy_s, best_policy_r, n_messages_on_path = get_exante_pareto_optimal()
    else
        best_policy_s, best_policy_r, n_messages_on_path = get_exante_receiver_optimal()
    end
    # get induced actions
    best_induced_actions = get_induced_actions(best_policy_s, best_policy_r)
    # get expected rewards
    best_expected_reward_s, best_expected_reward_r = get_expected_rewards(best_policy_s, best_policy_r)
    # compute mutual information
    best_mutual_information = get_mutual_information(best_policy_s) 
    best_posterior = get_posterior(best_policy_s)
    # check if a marginal change in bias changes the pareto optimal nash equilibrium
    is_borderline = get_N(bias, n_states) != get_N(bias+1e-3, n_states)

    # convert pareto optimum variables to dict
    best_nash = (best_induced_actions, best_policy_s, best_policy_r, best_expected_reward_s, best_expected_reward_r, best_mutual_information, best_posterior, n_messages_on_path, is_borderline)
    var_names = @names(best_induced_actions, best_policy_s, best_policy_r, best_expected_reward_s, best_expected_reward_r, best_mutual_information, best_posterior, n_messages_on_path, is_borderline)
    dict_pareto_optimum = Dict(name => value for (name, value) in zip(var_names, best_nash))
    return dict_pareto_optimum
end


function get_monotone_equilibria()
    # brute force over space of monotone strategies for senders
    n_policies_s = 2^(n_states-1)
    policy_s = Array{Float32,3}(undef,n_states,n_messages,n_policies_s)
    policy_r = Array{Float32,3}(undef,n_messages,n_actions,n_policies_s)
    induced_actions = Array{Float32,3}(undef,n_states,n_actions,n_policies_s)
    expected_reward_s = Array{Float32,1}(undef,n_policies_s)
    expected_reward_r = Array{Float32,1}(undef,n_policies_s)
    mutual_information = Array{Float32,1}(undef,n_policies_s)
    is_nash = zeros(Bool,n_policies_s)
    for idx in 2^(n_states-1) : -1 : 1
        cut = digits(idx-1, base=2, pad=(n_states-1))                                   # i-th possible partition
        N = count(cut.==1) + 1                                                          # number of messages
        N <= n_messages || continue
        I_ = N > 1 ? findall(cut .== 1) : [n_states]                
        I_ = [i > 1 ? I_[i] - I_[i-1] : I_[i] for i in 1:length(I_)]
        N > 1 && push!(I_,n_states - sum(I_))
        policy_s_ = convert_I_to_policy(I_, N)                                          # sender's candidate policy
        policy_r_ = get_best_reply_r(policy_s_)                                         # receiver's candidate policy
        is_best_reply(policy_s_, policy_r_, get_best_reply_s) || continue                                          
        # if here, (policy_s_, policy_r_) is an equilibrium
        policy_s[:,:,idx] .= policy_s_
        policy_r[:,:,idx] .= policy_r_
        induced_actions[:,:,idx] .= get_induced_actions(policy_s_,policy_r_)    
        expected_reward_s[idx], expected_reward_r[idx] = get_expected_rewards(policy_s_, policy_r_)
        mutual_information[idx] = get_mutual_information(policy_s_)
        is_nash[idx] = true
    end
    policy_s = policy_s[:,:,is_nash]
    policy_r = policy_r[:,:,is_nash]
    induced_actions = induced_actions[:,:,is_nash]
    expected_reward_s = expected_reward_s[is_nash] 
    expected_reward_r = expected_reward_r[is_nash]
    mutual_information = mutual_information[is_nash]
    n_nash = count(is_nash)

    perm = sortperm(-mutual_information)
    policy_s = policy_s[:,:,perm]
    policy_r = policy_r[:,:,perm]
    induced_actions = induced_actions[:,:,perm]
    expected_reward_s = expected_reward_s[perm]
    expected_reward_r = expected_reward_r[perm]
    mutual_information = mutual_information[perm]

    set_nash = (policy_s, policy_r, induced_actions, expected_reward_s, expected_reward_r, mutual_information, n_nash, bias)
    var_names = @names(policy_s, policy_r, induced_actions, expected_reward_s, expected_reward_r, mutual_information, n_nash, bias)
    dict_nash = Dict(name => value for (name, value) in zip(var_names, set_nash))
    return dict_nash
end
