
quantile_(data, alpha; dims = :) = mapslices(x -> quantile(x, alpha), data, dims=1)

# figure sizing (module globals). defaults give \linewidth output; paper_figures.jl flips to
# absolute cm w/ scale only axis. plot_policy heatmaps build their own axis (untouched)
plot_unit = raw"\linewidth"
plot_unit_scale = 1
plot_scale_only_axis = false

# line styling (module globals). paper_figures.jl overrides these for the paper figures;
# titles and labels are not set here
lw_value = "1.8pt"             # benchmark + grey equilibria lines (paper: 2.5pt)
lw_modal = "1pt"               # modal outcome (blue) line (paper: 1.3pt)
jump_value = "jump mark left"  # step direction of value lines (paper: jump mark right)

function init_tikz_axis(;title="", xlabel="", ylabel="", ylabel_style="rotate=-90", y_tick_label_style="{/pgf/number format/fixed, /pgf/number format/precision=4}", additional="", xmin=-0.0025, xmax=0.5025, ymin=0, ymax=1, width=0.35*ratio, height=0.35)
	pl = @pgf Axis(
	    {	#
	    	title = title,
	    	xlabel = xlabel, ylabel = ylabel,
	    	xtick = 0:0.1:0.5, #ytick = "",
	        xmin = xmin, xmax = xmax,
	        ymin = ymin, ymax = ymax, 
	        #
			y_tick_label_style = y_tick_label_style,
	 		ylabel_style = ylabel_style,
	        scaled_y_ticks = true,
	        #
			width = string(width*plot_unit_scale, plot_unit),
	        height = string(height*plot_unit_scale, plot_unit),
	        #
	        clip = true,
	        enlarge_y_limits=0.1,
			align = "center",
	    }
	);
    plot_scale_only_axis && push!(pl.options, "scale only axis")
    !isempty(additional) && push!(pl.options, additional)
	return pl
end

function add_legend!(pl, legend, pos)
   	push!(pl, LegendEntry(legend))
    if pos == "out_bottom" 
    	push!(pl.options, raw"xlabel style={name=xlabel}")
		push!(pl.options, raw"legend style={at={(xlabel.south)},anchor=north,legend columns = -1, column sep = 5pt}")
    elseif pos == "out_right"
    	push!(pl.options, raw"legend style={at={(axis description cs:1.1,0.5)},anchor=west}")
    end
end


function plot_avg(data; legend = "", title = "", xlabel=raw"$b$", ylabel = "", ylabel_style = "rotate=-90", color = "blue",
						y_tick_label_style = "{/pgf/number format/fixed, /pgf/number format/precision=4}",
						ymin = minimum(quantile_(data, 0.05, dims = 1)), 
						ymax = maximum(quantile_(data, 0.95, dims = 1)), 
						width = 0.35*ratio, height = 0.35,
						legend_pos = "", additional = "", ci_flag = true)

	# initialize tikz plot
	pl = init_tikz_axis(title = title, xlabel = xlabel, ylabel = ylabel, ylabel_style = ylabel_style, y_tick_label_style = y_tick_label_style, additional = additional, ymin = ymin, ymax = ymax, width = width, height = height)
	# plot average 
	pl = plot_avg!(pl, data; legend = legend, color = color, legend_pos = legend_pos, ci_flag = ci_flag)
	return pl
end

function plot_avg!(pl, data; α = 0.1, legend = "", color = "blue", legend_pos = "", ci_flag = true)
	# process data
 	n_simulations, n_biases = size(data)
 	set_biases_ = range(0.0, 0.5, n_biases)
	average = mean(data, dims = 1)[:]
    ci_flag && (confidence_interval = quantile_(data, [α/2, 1-α/2], dims = 1))
    # plot data
    @pgf pl_avg = Plot({color = color, style = "solid, thick"}, Table(x = set_biases_, y = average));
	push!(pl, pl_avg);
	if ci_flag == true
		@pgf pl_ub = Plot({"name path=f", no_marks, thin, "draw=none", "forget plot"}, Table(x = set_biases_, y = confidence_interval[2,:]));
		@pgf pl_lb = Plot({"name path=g", no_marks, thin, "draw=none", "forget plot"}, Table(x = set_biases_, y = confidence_interval[1,:]));
		@pgf pl_fill = Plot({color = color, fill = color, opacity = 0.1, "forget plot"}, raw"fill between [of=f and g]");
		push!(pl, pl_ub);
		push!(pl, pl_lb);
		push!(pl, pl_fill);
	end
    !isempty(legend) && add_legend!(pl, legend, legend_pos)
	return pl
end


function plot_dist(data; legend = "", title = "", xlabel = raw"$b$", ylabel = "", ylabel_style = "rotate=-90", color = "blue", 
						y_tick_label_style = "{/pgf/number format/fixed, /pgf/number format/precision=4}",
						ymin = minimum(quantile_(data, 0.05, dims = 1)), 
						ymax = maximum(quantile_(data, 0.95, dims = 1)), 
						width = 0.35*ratio, height = 0.35,
						n_steps = 65, legend_pos = "", additional = "")
	# x-axis margin = half the bias step, so heatmap edge cells fit inside the axis box
	xpad = 0.5 / (size(data, 2) - 1) / 2
	# initialize tikz plot
	pl = init_tikz_axis(title = title, xlabel = xlabel, ylabel = ylabel, ylabel_style = ylabel_style, y_tick_label_style = y_tick_label_style, additional = additional, xmin = -xpad, xmax = 0.5 + xpad, ymin = ymin, ymax = ymax, width = width, height = height)
	pl = plot_dist!(pl, data, ymin = ymin, ymax = ymax, color = color, n_steps = n_steps, legend = legend, legend_pos = legend_pos)
	return pl
end

function plot_dist!(pl, data; ymin = minimum(quantile_(data, 0.05, dims = 1)),
							  ymax = maximum(quantile_(data, 0.95, dims = 1)), 
							  color = "blue", n_steps = 65, legend = "", legend_pos = "")
	# input: data matrix of dimension n_simulations x n_bias

	# process data
 	n_simulations, n_biases = size(data)
 	set_biases_ = range(0.0,0.5,n_biases)
	step_size = (ymax - ymin) / (n_steps-4)
	points = range(ymin-3*step_size/2,ymax+3*step_size/2,n_steps)

	# add heatmap options 
	push!(pl.options,
		#colorbar,
		#colorbar_sampled,
		"point meta max = $n_simulations",
		"point meta min = 0",
		"colormap={whiteblue}{color(0cm)=($(color)!0), color(1cm)=($(color)!50)}",
		"colorbar style = {scaled ticks=false, /pgf/number format/fixed, /pgf/number format/precision = 5}",
	)

    # plot data
	h_data = hcat([fit(Histogram, data[:,b], points).weights for b in 1:n_biases]...)
	x, y = repeat(set_biases_, outer = n_steps-1), repeat(points[2:end], inner = n_biases)
	@pgf coord = Table({"meta=value"},["x" => x, "y" => y, "value" => vec(h_data')])
    push!(coord.options,"y expr=\\thisrow{y} - $step_size * 1/3")
    @pgf pl_hm = Plot({ "matrix plot*", mark = "", point_meta = "explicit", "mesh/cols" = n_biases, on_layer = "axis background", forget_plot}, coord)
	push!(pl, pl_hm);
    !isempty(legend) && add_legend!(pl, legend, legend_pos)
	!isempty(legend) && push!(pl, "\\addlegendimage{only marks, mark=square*, fill=$color, draw=$color, opacity = 0.6}")
	return pl
end

function plot_val!(pl, data; legend = "", color = "red", style = "solid", opacity = 1.0, blend_mode="multiply", jump_mark = jump_value, legend_pos = "out_bottom")
	# plot value on top of existing plot. jump_mark holds the value to the right of each bias
	# ("jump mark right") or to the left ("jump mark left", used for the modal-outcome lines)
	val = getindex.(data,1)
    set_biases_ = range(0.0,0.5,length(data))
    @pgf pl_val = Plot({axis_on_top, color = color, style = style, opacity = opacity, blend_mode=blend_mode}, Table(x = set_biases_, y = val));
    push!(pl_val.options, jump_mark)
	push!(pl, pl_val);
    !isempty(legend) && add_legend!(pl, legend, legend_pos)
	return pl
end

function plot_interpolated_val!(pl, data; legend = "", color = "red", style = "solid", opacity = 1.0, blend_mode="multiply",
							 			  ymin = minimum(quantile_(data, 0.05, dims = 1)),
							 			  ymax = maximum(quantile_(data, 0.95, dims = 1)), n_steps=50)
	# plot value on top of existing plot
	val = getindex.(data,1)
    set_biases_ = range(0.0,0.5,length(data))
	# interpolate values to distribution grid 
	step_size = (ymax - ymin) / (n_steps-4)
	points = range(ymin-3*step_size/2,ymax+3*step_size/2,n_steps)
    val = map(x -> (i = findfirst(points .>= x); i === nothing ? x : points[i] - step_size/2), val)
	# plot
    @pgf pl_val = Plot({axis_on_top, color = color, style = style, opacity = opacity, jump_mark_right, blend_mode=blend_mode}, Table(x = set_biases_, y = val));
	push!(pl, pl_val);
    !isempty(legend) && push!(pl, LegendEntry(legend))
	return pl
end

function plot_eq_bound!(pl,posterior_mean_variance_best)
	# plot shaded areas to indicate where babbling is the unique equilibrium and where full communication is an equilibrium
	set_biases_ = range(0.0,0.5,length(posterior_mean_variance_best))
	# last index at which informativeness is maximal
	mi1_idx = findlast(posterior_mean_variance_best .== maximum(posterior_mean_variance_best))
	# first index at which informativeness is minimal
	mi0_idx = findfirst(posterior_mean_variance_best .== minimum(posterior_mean_variance_best))
	@pgf pl_best_lx_bound = VLine({"draw=none", "name path=blx"}, 0.0)
	@pgf pl_best_rx_bound = VLine({"draw=none", "name path=brx"}, set_biases_[mi1_idx])
	@pgf pl_best_fill = Plot({color = "gray", fill = "gray", opacity = 0.08}, raw"fill between [of=blx and brx]");
	@pgf pl_worst_lx_bound = VLine({"draw=none", "name path=wlx"}, set_biases_[mi0_idx])
	@pgf pl_worst_rx_bound = VLine({"draw=none", "name path=wrx"}, 0.5)
	@pgf pl_worst_fill = Plot({color = "gray", fill = "gray", opacity = 0.08}, raw"fill between [of=wlx and wrx]");
	push!(pl, pl_best_lx_bound);
	push!(pl, pl_best_rx_bound);
	push!(pl, pl_best_fill);
	push!(pl, pl_worst_lx_bound);
	push!(pl, pl_worst_rx_bound);
	push!(pl, pl_worst_fill);
end

function plot_policy(policy, xlabel, ylabel, xticklabel, yticklabel, xstep, ystep, title; tick_label_font = raw"\normalsize", ylabel_opts = raw"{rotate=-90}", width = raw"0.25\linewidth", height = raw"0.25\linewidth")
 	x = repeat(1:size(policy,1), outer = size(policy,2))
	y = repeat(1:size(policy,2), inner = size(policy,1))
	coord = Coordinates(x, y; meta = vec(policy[:,end:-1:1]))

	plot = @pgf Axis(
	    {
	        title = title,
			xlabel = xlabel, ylabel = ylabel,
	        xtick = (1:xstep:size(policy,1)),
	        ytick = size(policy,2):-ystep:0,
	  	    xticklabels = string.(xticklabel),
	        yticklabels = string.(yticklabel),
	        #
	        xtick_style="{draw=none}",
    		ytick_style="{draw=none}",
	        "tick label style" = string("{font=", tick_label_font, "}"),
	        ylabel_style = ylabel_opts,
	        #
			colormap= "{reversed blackwhite}{gray(0cm)=(1); gray(1cm)=(0)}",
	        point_meta_min = 0.00,
			point_meta_max = 1.0,
			#
	        y_dir = "normal",
	        enlargelimits = false,
			width = width,
            height = height,
	    },
	    PlotInc(
	        {
	            matrix_plot,
	            mark = "",
	            point_meta = "explicit",
	            "mesh/cols" = size(policy,1)
	        },
	        coord,
	    ),
	)	
	return plot
end

# regime identifier for the robustness figures, shown as the sender panel's ylabel; empty for
# the baseline. fourth/absolute collapse u_S/u_R into a single u_i(.) form
function get_title_equation()
	labels = Dict(
		"out_3states"       => raw"$|\Theta| = 3$",
		"out_9states"       => raw"$|\Theta| = 9$",
		"out_less_messages" => raw"$|M| = 3$",
		"out_more_messages" => raw"$|M| = 9$",
		"out_less_actions"  => raw"$|A| = 9$",
		"out_more_actions"  => raw"$|A| = 21$",
		"out_increasing"    => raw"$p(\theta_k)=2k / (n(n+1))$",
		"out_decreasing"    => raw"$p(\theta_k)=2 (n-k+1) / (n(n+1))$",
		"out_fourthpower"   => raw"$u_i(\theta,a) = -( \, \cdot \, )^4$",
		"out_absolute"      => raw"$u_i(\theta,a) = -| \, \cdot \, |$",
	)
	return get(labels, script_config["in_dir"], "")
end
