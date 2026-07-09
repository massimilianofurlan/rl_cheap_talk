
# Analytic location of the babbling-branch bifurcations of the logit agent-QRE
# correspondence, via the message-relabelling symmetry.
#
# The babbling profile (uniform sender, rho = 0) is a logit agent QRE at every
# precision lambda: it is the branch through the centroid. Informative branches are
# BORN on it, at the precisions where its linearisation loses a unit eigenvalue --
# a pitchfork forced by the S_n symmetry on messages. This file locates those
# precisions, lambda*, without any continuation or proximity threshold. It does so
# two ways and checks they agree:
#
#   (1) FULL Jacobian.  D_rho g(0, lambda) is the rho-block of the finite-difference
#       Jacobian of the QRE residual at rho = 0 (the same residual the figure traces).
#       A bifurcation is a lambda at which an eigenvalue of D_rho T_lambda(0) crosses 1.
#
#   (2) SYMMETRY-REDUCED scalar equation.  A single-message-contrast perturbation
#       decouples the linearisation into a small operator M(lambda) on type-loadings,
#       built from the symmetric receiver response pi_bar^R(lambda) = softmax(lambda V_bar^R).
#       The condition becomes lambda^2 * sigma(lambda) = 1 for an eigenvalue sigma of M.
#
# The bias b enters M only through a term that is constant in the type theta, so
# lambda* is predicted to be BIAS-INDEPENDENT; we verify this by scanning b in (1).
#
# What this does and does not give. lambda* is where informative branches ATTACH to
# the babbling branch (and hence where a reachable branch connects to the centroid
# component in qre.jl). It is NOT the bias where a branch DETACHES from the centroid
# component; detachment is a nonlocal fold of the informative branch, not a local
# rho = 0 eigenvalue crossing, and is left to the continuation in qre.jl.
#
# All arithmetic mirrors qre.jl (float32 game, float64 linear algebra). Run e.g.
#   julia qre_bifurcation.jl -n 3
#
# Result (n=3, |A|=5, uniform, quadratic). The babbling branch has a SINGLE bifurcation
# in (0, 80], at lambda* = 4.2415, and it is a DOUBLE eigenvalue crossing (null dim 2,
# eigenvalues real to 1e-17): the six-dim rho-space is three copies of the two-dimensional
# standard irrep of S_3, so every eigenvalue is doubled. The reduced scalar equation
# reproduces lambda* to 6e-6, and lambda* is EXACTLY bias-independent: the relevant
# eigenvalue sigma is constant to 10 digits across b (the ~1e-3 spread printed in section
# (3) is float32 finite-difference noise). Mechanism: b enters M only as a rank-1 term
# along the constant-in-type vector 1, which is a right eigenvector of M for eigenvalue 0
# and is orthogonal (biorthogonal) to the left eigenvector of the relevant sigma, so sigma
# is unchanged. So all informative branches are born at one bias-independent bifurcation;
# which partition a branch limits to, and the bias at which it detaches from the centroid
# component, are nonlinear/global facts not visible in this local spectrum.

using Random
using LinearAlgebra
include("file_io.jl")
include("rl_agents.jl")
include("analysis.jl")
include("nash.jl")

# parse configuration (structural game settings; bias is set per call below)
const term_parse = parse_commandline()
const file_parse = TOML.parsefile("config.toml")[term_parse["config"]]
const config = merge(file_parse, term_parse)

# GAME (as in qre.jl)
const n_states = config["n_states"]
const n_actions = config["n_actions"]
const n_messages = config["n_messages"]
const T = collect(0:1f0/(n_states-1):1)
const A = collect(0:1f0/(n_actions-1):1)
const M = n_messages < 26 ? collect('a':'z')[1:n_messages] : collect(1:n_messages)
const loss_type = config["loss"]
const dist_type = config["dist"]
const p_t = gen_distribution()
const policy_type = "softmax"
const get_policy = get_softmax_policy
const ptol = 0.001f0
const rtol = 0.001f0
const gtol = 0.01f0
# bias and reward matrices are reset for each bias (get_q_s/get_q_r read these globals)
bias = config["bias"]
reward_matrix_s, reward_matrix_r = gen_reward_matrix()

const n_rho = n_states * (n_messages - 1)


# QRE map and residual (identical to qre.jl) ---------------------------------

function get_policy_from_rho(rho)
    scores = hcat(zeros(Float32, n_states), Float32.(reshape(rho, n_states, n_messages-1)))
    return get_softmax_policy(scores, 1f0)
end

get_rho_from_policy_(policy_s) =
    Float64.(vec(log.(max.(policy_s,1f-30))[:,2:end] .- log.(max.(policy_s,1f-30))[:,1:1]))

function quantal_response_map(rho, lambda)
    temp = Float32(1 / max(lambda, 1e-8))
    policy_s = get_policy_from_rho(rho)
    policy_r = get_softmax_policy(get_q_r(policy_s), temp)
    return get_rho_from_policy_(get_softmax_policy(get_q_s(policy_r), temp))
end

residual(y) = quantal_response_map(y[1:n_rho], y[n_rho+1]) .- y[1:n_rho]

function jacobian_rho(lambda; h = 1e-3)
    # rho-block of the central-difference Jacobian of the residual at the babbling
    # point rho = 0: this is D_rho g(0, lambda) = D_rho T_lambda(0) - I
    y0 = vcat(zeros(n_rho), Float64(lambda))
    J = Matrix{Float64}(undef, n_rho, n_rho)
    for k in 1:n_rho
        yp = copy(y0); yp[k] += h
        ym = copy(y0); ym[k] -= h
        J[:,k] = (residual(yp) .- residual(ym)) ./ (2h)
    end
    return J
end


# (1) bifurcations from the full babbling Jacobian ---------------------------

# eigenvalues of D_rho T_lambda(0) = jacobian_rho(lambda) + I; a bifurcation is where
# a real eigenvalue crosses 1 (equivalently D_rho g is singular)
map_eigs(lambda) = eigvals(jacobian_rho(lambda) + I)

function sorted_real_eigs(lambda)
    # real parts of the eigenvalues of D_rho T, sorted descending, and the largest
    # absolute imaginary part (to confirm the crossing eigenvalues are real)
    ev = map_eigs(lambda)
    return sort(real.(ev); rev = true), maximum(abs.(imag.(ev)))
end

function find_bifurcations(; lambda_max = 80.0, dl = 0.2)
    # every lambda in (0, lambda_max] at which some eigenvalue of D_rho T crosses 1,
    # found by tracking the k-th largest real eigenvalue on a grid and bisecting each
    # sign change of (eigenvalue - 1)
    grid = collect(dl:dl:lambda_max)
    E = hcat([sorted_real_eigs(l)[1] for l in grid]...)          # n_rho x n_grid
    raw = Float64[]
    for k in 1:n_rho, i in 1:length(grid)-1
        (E[k,i]-1) * (E[k,i+1]-1) < 0 || continue                # crossing of eigenvalue k
        a, b, fa = grid[i], grid[i+1], E[k,i]-1
        for _ in 1:60
            m = 0.5*(a+b); fm = sorted_real_eigs(m)[1][k] - 1
            (fa*fm <= 0) ? (b = m) : (a = m; fa = fm)
        end
        push!(raw, 0.5*(a+b))
    end
    # cluster crossings that coincide (a degenerate eigenvalue makes several cross together)
    isempty(raw) && return Tuple{Float64,Int}[]
    ls = sort(raw); clusters = Tuple{Float64,Int}[]; start = ls[1]; count = 1
    for j in 2:length(ls)
        if ls[j] - ls[j-1] < 1e-3
            count += 1
        else
            push!(clusters, ((start+ls[j-1])/2, count)); start = ls[j]; count = 1
        end
    end
    push!(clusters, ((start+ls[end])/2, count))
    return clusters
end

function nullspace_structure(lambda)
    # near-null right singular vectors of D_rho g at a bifurcation, reshaped to
    # (type x message-contrast): which types differentiate which messages
    F = svd(jacobian_rho(lambda))
    dim = count(F.S .< 1e-2 * maximum(F.S))
    vecs = [reshape(F.V[:, end-j+1], n_states, n_messages-1) for j in 1:max(dim,1)]
    return dim, vecs
end


# (2) symmetry-reduced scalar equation ---------------------------------------

function symmetric_receiver(lambda)
    # pi_bar^R(a) = softmax_a(lambda * V_bar^R(a)), the receiver's logit response to the
    # prior belief (bias-independent: the receiver payoff has no bias)
    Vbar = Float64.(reward_matrix_r * p_t)                        # V_bar^R(a) = -sum_theta p(theta)(a-theta)^2
    w = exp.(lambda .* (Vbar .- maximum(Vbar)))
    return w ./ sum(w)
end

function reduced_operator(lambda)
    # M(lambda): x (type-loading) |-> chi, whose eigenvalue sigma gives a babbling
    # bifurcation via lambda^2 * sigma = 1. From a single-message-contrast perturbation
    # of the symmetric branch (see header).
    piR = symmetric_receiver(lambda)
    Rr = Float64.(reward_matrix_r); Rs = Float64.(reward_matrix_s); pt = Float64.(p_t)
    Mmat = Matrix{Float64}(undef, n_states, n_states)
    for j in 1:n_states
        x = zeros(n_states); x[j] = 1.0
        xc = x .- dot(pt, x)                                      # centre by the prior
        psi = Rr * (pt .* xc)                                     # psi(a) = -sum_theta p(theta) xc(theta)(a-theta)^2
        psit = psi .- dot(piR, psi)                              # centre by pi_bar^R
        Mmat[:, j] = Rs' * (piR .* psit)                          # chi(theta)
    end
    return Mmat
end

function analytic_bifurcations(; lambda_max = 80.0, dl = 0.2)
    # solve lambda^2 * sigma(lambda) = 1 for the largest real eigenvalue sigma of M(lambda)
    grid = collect(dl:dl:lambda_max)
    f(l) = l^2 * maximum(real.(eigvals(reduced_operator(l)))) - 1
    fv = f.(grid); roots = Float64[]
    for i in 1:length(grid)-1
        fv[i]*fv[i+1] < 0 || continue
        a, b, fa = grid[i], grid[i+1], fv[i]
        for _ in 1:60
            m = 0.5*(a+b); fm = f(m)
            (fa*fm <= 0) ? (b = m) : (a = m; fa = fm)
        end
        push!(roots, 0.5*(a+b))
    end
    return roots
end


# report ---------------------------------------------------------------------

set_bias!(b) = (global bias = Float32(b); global reward_matrix_s, reward_matrix_r = gen_reward_matrix())

function main()
    println("game: n_states=$n_states  n_actions=$n_actions  n_messages=$n_messages  ",
            "loss=$loss_type  dist=$dist_type   (rho dimension n_rho=$n_rho)\n")

    set_bias!(bias)
    println("== (1) babbling bifurcations from the full Jacobian at rho=0 (b=$(round(bias,digits=3))) ==")
    bifs = find_bifurcations()
    if isempty(bifs)
        println("  none in (0, 80].")
    else
        for (l, mult) in bifs
            dim, _ = nullspace_structure(l)
            imax = sorted_real_eigs(l)[2]
            println("  lambda* = $(round(l, digits=4))   eigenvalues crossing 1: $mult   ",
                    "null dim: $dim   max|Im eig|: $(round(imax, sigdigits=2))")
        end
    end

    println("\n== (2) symmetry-reduced scalar equation  lambda^2 * sigma(lambda) = 1 ==")
    roots = analytic_bifurcations()
    if isempty(roots)
        println("  no real positive root in (0, 80].")
    else
        for l in roots; println("  lambda* = $(round(l, digits=4))"); end
    end
    if !isempty(bifs) && !isempty(roots)
        d = minimum(abs(first(bifs[1]) - r) for r in roots)
        println("  |primary full - nearest reduced| = $(round(d, sigdigits=2))   ",
                d < 1e-2 ? "(agree)" : "(DISAGREE -- check derivation)")
    end

    println("\n== (3) bias-independence: primary lambda* across b ==")
    println("     b        lambda*(full)")
    b0 = bias
    for b in 0.0:0.05:0.5
        set_bias!(b)
        bb = find_bifurcations()
        println("   $(rpad(round(b,digits=2),6))   $(isempty(bb) ? NaN : round(first(bb[1]), digits=4))")
    end
    set_bias!(b0)
    return nothing
end

main()
