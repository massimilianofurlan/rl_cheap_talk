
function compute_statistics(set_nash, results)
    # compute statistics

    # no statistics in raw mode
    raw && return Dict()

    # get sessions that have converged
    is_converged = results["is_converged"]
    # get sessions that have converged to a γ-nash
    is_nash = is_converged .& results["is_nash"]

    # compute statistics for converged and not converged sessions separately
    statistics_converged = compute_group_statistics(results, is_converged)
    statistics_not_converged = compute_group_statistics(results, .!is_converged) 

    # get set of nash equilibria
    nash_induced_actions, n_nash = set_nash["induced_actions"], set_nash["n_nash"]
    # get induced actions
    induced_actions = results["induced_actions"]

    # assign nash id
    nash_ids = get_nash_ids(induced_actions, nash_induced_actions; tol=0.01f0)
    nash_ids[.!is_nash] .= 0
    # compute statistics for each nash_id separately (only converged sessions)
    statistics_nash = Dict(i => Dict{String, Any}() for i in 0:n_nash)
    for nash_id in 0:n_nash
        statistics_nash[nash_id] = compute_group_statistics(results, nash_ids .== nash_id)
    end

    # assign similarity classes
    class_ids = get_class_ids(induced_actions; tol=0.01f0)
    n_classes = maximum(class_ids)
    # compute statistics for each similarity class separately (only if not all gamma-nash and frequence > 0.025) 
    statistics_not_nash = Dict()
    curr_id = -1
    for class_id in 1:n_classes
        class_statitsitcs = compute_group_statistics(results, class_ids .== class_id)
        if class_statitsitcs["freq"] >= 0.025 && class_statitsitcs["freq_nash"] < 1.0
            statistics_not_nash[curr_id] = class_statitsitcs
            curr_id -= 1
        end
    end

    dict_statistics = Dict("converged" => statistics_converged, 
                           "not_converged" => statistics_not_converged,
                           statistics_nash...,
                           statistics_not_nash...)
    return dict_statistics
end


function compute_group_statistics(results, group)
    # compute statistics for sessions in group
    n_group = count(group)
    n_group == 0 && return Dict()

    # read experiment outcomes from file
    n_episodes = results["n_episodes"][group]
    expected_reward_s = results["expected_reward_s"][group]
    expected_reward_r = results["expected_reward_r"][group]
    absolute_error_s = results["absolute_error_s"][group]
    absolute_error_r = results["absolute_error_r"][group]
    mutual_information = results["mutual_information"][group]
    residual_variance = results["residual_variance"][group]
    n_off_path_messages = results["n_off_path_messages"][group]
    max_mass_on_suboptim_s = results["max_mass_on_suboptim_s"][group]
    max_mass_on_suboptim_r = results["max_mass_on_suboptim_r"][group]
    max_mass_on_suboptim = results["max_mass_on_suboptim"][group]
    n_effective_messages = results["n_effective_messages"][group]
    is_partitional = results["is_partitional"][group]
    is_nash = results["is_nash"][group]
    is_absorbing = results["is_absorbing"][group]
    margin_error_s = results["margin_error_s"][:,group]
    margin_error_r = results["margin_error_r"][:,group]

    # average episodes played, frequence in group
    freq = count(group) / n_simulations
    avg_n_episodes = trunc.(Int,mean_std(n_episodes))
    # average expected rewards
    avg_expected_reward_s = mean_std(expected_reward_s)
    avg_expected_reward_r = mean_std(expected_reward_r)
    # average expected error
    avg_absolute_error_s = mean_std(absolute_error_s)
    avg_absolute_error_r = mean_std(absolute_error_r)
    # average mutual info
    avg_mutual_information = mean_std(mutual_information)
    # average residual variance
    avg_residual_variance = mean_std(residual_variance)
    # on-path and off-path messages
    avg_n_on_path_messages = mean_std(n_messages .- n_off_path_messages)
    # number of messages with no sysnonims
    avg_n_effective_messages = mean_std(n_effective_messages)
    # freq partitional policy 
    freq_partitional = mean_std(is_partitional)
    # average margin estimation error
    avg_margin_error_s = dropdims(mean(margin_error_s, dims=2), dims=2)
    avg_margin_error_r = dropdims(mean(margin_error_r, dims=2), dims=2)

    # epsilon-nash 
    max_absolute_error = max.(absolute_error_s, absolute_error_r)                  # smallest ϵ that makes each simulation an ϵ-approximate equilibrium
    quant_max_absolute_error = quantile(max_absolute_error, [0.9, 0.95, 1.0])      # value of ϵ that makes 90, 95 and 100% of simulation an ϵ-approximate equilibrium
   
    # average gamma
    avg_max_mass_on_suboptim_s = mean_std(max_mass_on_suboptim_s) 
    avg_max_mass_on_suboptim_r = mean_std(max_mass_on_suboptim_r) 
    # gamma-nash
    quant_max_mass_on_suboptim = quantile(max_mass_on_suboptim, [0.25,0.5,0.75])
    # frequence (γ < 1f-2)-nash 
    freq_nash = count(is_nash) / n_group
    # frequence is fixed point
    freq_is_absorbing = count(is_absorbing) / n_group


    statistics = (group, freq, avg_n_episodes, avg_expected_reward_s, avg_expected_reward_r, avg_absolute_error_s, avg_absolute_error_r,
                  avg_max_mass_on_suboptim_s, avg_max_mass_on_suboptim_r, avg_mutual_information, avg_residual_variance, avg_n_on_path_messages, avg_n_effective_messages, 
                  max_absolute_error, quant_max_absolute_error, max_mass_on_suboptim, quant_max_mass_on_suboptim, freq_nash, freq_partitional, freq_is_absorbing, avg_margin_error_s, avg_margin_error_r)
    var_names = @names(group, freq, avg_n_episodes, avg_expected_reward_s, avg_expected_reward_r, avg_absolute_error_s, avg_absolute_error_r,
                  avg_max_mass_on_suboptim_s, avg_max_mass_on_suboptim_r,  avg_mutual_information, avg_residual_variance, avg_n_on_path_messages, avg_n_effective_messages, 
                  max_absolute_error, quant_max_absolute_error, max_mass_on_suboptim, quant_max_mass_on_suboptim, freq_nash, freq_partitional, freq_is_absorbing, avg_margin_error_s, avg_margin_error_r)
    dict_statistics = Dict(name => value for (name, value) in zip(var_names, statistics))
    return dict_statistics
end

mean_std(x; dims = :) = (mean(x, dims = dims), std(x, dims = dims, corrected = false))

function get_nash_ids(induced_actions::Array{Float32,3}, nash_induced_actions::Array{Float32,3}; tol::Float32 = rtol)
    # returns nash ids
    n_nash = size(nash_induced_actions, 3)
    nash_ids = zeros(Int, n_simulations)
    @inbounds @fastmath for z in 1:n_simulations
        for nash_id in 1:n_nash
            is_approx(nash_induced_actions[:,:,nash_id], induced_actions[:,:,z]; tol = tol) || continue
            nash_ids[z] = nash_id
            break
        end
    end
    return nash_ids
end

function get_class_ids(induced_actions::Array{Float32,3}; tol::Float32 = rtol)
    # returns similarity class ids
    class_ids, curr_id = zeros(Int, n_simulations), 0
    @inbounds @fastmath for i in 1:n_simulations
        class_ids[i] == 0 && (curr_id += 1; class_ids[i] = curr_id)
        for j in i+1:n_simulations
            class_ids[j] == 0 || continue
            is_approx(induced_actions[:,:,i], induced_actions[:,:,j]; tol=tol) || continue
            # assign to the same class if similar
            class_ids[j] = class_ids[i]
        end
    end
    return class_ids
end

#=function get_nash_dists(induced_actions::Array{Float32,3}, nash_induced_actions::Array{Float32,3})
    # compute distance in l2-norm form each partitonal equilibrium
    n_nash = size(nash_induced_actions,3)
    nash_dists = Array{Float32,2}(undef, n_nash, n_simulations)
    @inbounds @fastmath for z in 1:n_simulations
        for nash_id in 1:n_nash
            nash_dists[nash_id, z] = norm_(induced_actions[:,:,z] - nash_induced_actions[:,:,nash_id])
        end
    end
    return nash_dists
end=#


