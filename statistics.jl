
function compute_statistics(results)
    # compute statistics

    # no statistics in raw mode
    raw && return Dict()

    # get sessions that have converged
    is_converged = results["n_episodes"] .< n_max_episodes

    # compute statistics for converged and not converged sessions separately
    statistics_converged = compute_group_statistics(results, is_converged)
    statistics_not_converged = compute_group_statistics(results, .!is_converged) 

    dict_statistics = Dict("converged" => statistics_converged, 
                           "not_converged" => statistics_not_converged)
    return dict_statistics
end


function compute_group_statistics(results, group)
    # compute statistics for sessions in group
    n_group = count(group)
    n_group == 0 && return Dict()

    # read experiment outcomes from file
    n_episodes = results["n_episodes"][group]
    n_conv_diff = results["n_conv_diff"][group]
    expected_reward_s = results["expected_reward_s"][group]
    expected_reward_r = results["expected_reward_r"][group]
    expected_aggregate_reward = results["expected_aggregate_reward"][group]
    absolute_error_s = results["absolute_error_s"][group]
    absolute_error_r = results["absolute_error_r"][group]
    mutual_information = results["mutual_information"][group]
    off_path_messages = results["off_path_messages"][:,group]
    mass_on_suboptim_s = results["mass_on_suboptim_s"][:,group]
    mass_on_suboptim_r = results["mass_on_suboptim_r"][:,group]
    is_partitional = results["is_partitional"][group]
    n_effective_messages = results["n_effective_messages"][group]

    # average episodes played, frequence in group
    freq = count(group) / n_simulations
    avg_n_episodes = trunc.(Int,mean_std(n_episodes))
    avg_n_conv_diff = trunc.(Int,mean_std(n_conv_diff))
    # average expected rewards
    avg_expected_reward_s = mean_std(expected_reward_s)
    avg_expected_reward_r = mean_std(expected_reward_r)
    avg_expected_aggregate_reward = mean_std(expected_aggregate_reward)
    # average expected error
    avg_absolute_error_s = mean_std(absolute_error_s)
    avg_absolute_error_r = mean_std(absolute_error_r)
    # average mutual info
    avg_mutual_information = mean_std(mutual_information)
    # on-path and off-path messages
    avg_n_on_path_messages = mean_std(n_messages .- count(off_path_messages, dims=1))
    # number of messages with no sysnonims
    avg_n_effective_messages = mean_std(n_effective_messages)
    # freq partitional policy 
    freq_partitional = mean_std(is_partitional)

    # epsilon-nash 
    min_absolute_error = min.(absolute_error_s,absolute_error_r)                  # smallest ϵ that makes each simulation an ϵ-approximate equilibrium
    quant_min_absolute_error = quantile(min_absolute_error, 1.0 .- [0.9,0.95,1])  # value of ϵ that makes 90, 95 and 100% of simulation an ϵ-approximate equilibrium
   
    # gamma
    max_mass_on_suboptim_s = maximum(mass_on_suboptim_s, dims=1)[:]
    max_mass_on_suboptim_r = maximum(.!off_path_messages .* mass_on_suboptim_r, dims=1)[:]
    # average gamma (action adjusted)
    avg_max_mass_on_suboptim_s = mean_std(max_mass_on_suboptim_s * n_messages) 
    avg_max_mass_on_suboptim_r = mean_std(max_mass_on_suboptim_r * n_actions) 
    # gamma-nash
    max_mass_on_suboptim = max.(max_mass_on_suboptim_s, max_mass_on_suboptim_r)
    quant_max_mass_on_suboptim = quantile(max_mass_on_suboptim, [0.25,0.5,0.75])
    
    # exact nash (γ .< 1f-3)
    freq_nash = count(max_mass_on_suboptim .< 1f-3) / n_simulations


    statistics = (group, freq, avg_n_episodes, avg_n_conv_diff, avg_expected_reward_s, avg_expected_reward_r, avg_expected_aggregate_reward, 
                  avg_absolute_error_s, avg_absolute_error_r, avg_max_mass_on_suboptim_s, avg_max_mass_on_suboptim_r, avg_mutual_information, 
                  avg_n_on_path_messages, avg_n_effective_messages, min_absolute_error, quant_min_absolute_error, max_mass_on_suboptim, quant_max_mass_on_suboptim, 
                  freq_nash, freq_partitional)
    var_names = @names(group, freq, avg_n_episodes, avg_n_conv_diff, avg_expected_reward_s, avg_expected_reward_r, avg_expected_aggregate_reward, 
                  avg_absolute_error_s, avg_absolute_error_r, avg_max_mass_on_suboptim_s, avg_max_mass_on_suboptim_r,  avg_mutual_information, 
                  avg_n_on_path_messages, avg_n_effective_messages, min_absolute_error, quant_min_absolute_error, max_mass_on_suboptim, quant_max_mass_on_suboptim, 
                  freq_nash, freq_partitional)
    dict_statistics = Dict(name => value for (name, value) in zip(var_names, statistics))
    return dict_statistics
end

mean_std(x; dims = :) = (mean(x, dims = dims), std(x, dims = dims, corrected = false))


