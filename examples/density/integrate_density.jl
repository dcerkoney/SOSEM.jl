# using AbstractTrees
using CodecZlib
using CompositeGrids
using ElectronGas
using ElectronLiquid
using ElectronLiquid.UEG
using FeynmanDiagram
using JLD2
using Measurements
using MCIntegration
using Lehmann
using LinearAlgebra
using Parameters
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np

function build_occupation_with_ct(orders; renorm_mu=true, renorm_lambda=true, isFock=false)
    DiagTree.uidreset()
    valid_partitions = Vector{PartitionType}()
    diagparams = Vector{DiagParaF64}()
    diagtrees = Vector{DiagramF64}()
    exprtrees = Vector{ExprTreeF64}()
    # Build all counterterm partitions at the given orders (lowest order (Fock) is N = 1)
    partitions = DiagGen.counterterm_partitions(
        orders;
        n_lowest=0,
        renorm_mu=renorm_mu,
        renorm_lambda=renorm_lambda,
    )
    @debug "Partitions: $partitions"
    for p in partitions
        # Build diagram tree for this partition
        @debug "Partition (n_loop, n_ct_mu, n_ct_lambda): $p"
        diagparam, diagtree = build_diagtree(; n_loop=p[1])

        # Build tree with counterterms (∂λ(∂μ(DT))) via automatic differentiation
        dμ_diagtree = DiagTree.derivative([diagtree], BareGreenId, p[2]; index=1)
        dλ_dμ_diagtree = DiagTree.derivative(dμ_diagtree, BareInteractionId, p[3]; index=2)
        if isempty(dλ_dμ_diagtree)
            @warn("Ignoring partition $p with no diagrams")
        else
            if isFock && (p != (1, 0, 0)) # the Fock diagram itself should not be removed
                DiagTree.removeHartreeFock!(dλ_dμ_diagtree)
            end
            @debug "\nDiagTree:\n" * repr_tree(dλ_dμ_diagtree)
            # Compile to expression tree and save results for this partition
            exprtree = ExprTree.build(dλ_dμ_diagtree)
            push!(valid_partitions, p)
            push!(diagparams, diagparam)
            push!(exprtrees, exprtree)
            append!(diagtrees, dλ_dμ_diagtree)
        end
    end
    return valid_partitions, diagparams, diagtrees, exprtrees
end

function build_diagtree(; n_loop=0)
    # Instantaneous Green's function (occupation number) diagram parameters
    #
    # NOTE: Differentiation and Fock filter do not commute, so we need to manually zero
    #       out the Fock insertions after differentiation (via DiagTree.removeHartreeFock!)
    DiagTree.uidreset()
    diagparam = DiagParaF64(;
        type=GreenDiag,
        hasTau=true,
        firstLoopIdx=2,
        innerLoopNum=n_loop,
        firstTauIdx=2,
        totalTauNum=n_loop + 1,
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa interaction
        filter=[NoHartree],
    )
    # # Instantaneous Green's function (occupation number) diagram parameters
    # diagparam = DiagParaF64(;
    #     type=GreenDiag,
    #     innerLoopNum=n_loop,
    #     firstTauIdx=2,
    #     totalTauNum=n_loop + 1,
    #     hasTau=true,
    # )

    # Loop basis vector for external momentum
    k = DiagTree.getK(diagparam.totalLoopNum, 1)

    # NOTE: there is only 1 external time (instantaneous G)
    extT = (1, 1)
    diagtree = Parquet.green(diagparam, k, extT; name=:nₖ)

    @debug "\nDiagTree:\n" * repr_tree(diagtree)
    return diagparam, diagtree
end

function integrate_density_with_ct(
    mcparam::UEG.ParaMC,
    diagparams::Vector{DiagParaF64},
    exprtrees::Vector{ExprTreeF64};
    alpha=3.0,
    neval=1e5,
    print=-1,
    solver=:vegasmc,
)
    @assert all(p.totalTauNum ≤ mcparam.order + 1 for p in diagparams)

    # We assume that each partition expression tree has a single root
    @assert all(length(et.root) == 1 for et in exprtrees)

    # List of expression tree roots, external times, and total
    # loop numbers for each tree (to be passed to integrand)
    # roots = [et.root[1] for et in exprtrees]
    totalLoopNums = [p.totalLoopNum for p in diagparams]

    # Temporary array for combined K-variables.
    # We use the maximum necessary loop basis size for K pool.
    maxloops = maximum(totalLoopNums)
    varK = zeros(3, maxloops)

    # Build adaptable MC integration variables
    (K, T) = density_mc_variables(mcparam, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T)
    # We do not integrate the incoming external time and nₖ is instantaneous, hence n_τ = totalTauNum - 1
    dof = [[p.totalLoopNum, p.totalTauNum - 1] for p in diagparams]
    println(dof)

    # Total density is a scalar
    obs = zeros(length(dof))  # observable for each partition

    # External times are fixed for left/right measurement of the discontinuity at τ = 0
    T.data[1] = 0  # τin = 0 (= τout⁺)

    # Phase-space factors
    phase_factors = [1.0 / (2π)^(mcparam.dim * nl) for nl in totalLoopNums]

    # Total prefactors
    prefactors = -phase_factors

    return integrate(
        integrand;
        solver=solver,
        measure=measure,
        neval=neval,
        print=print,
        # MC config kwargs
        userdata=(mcparam, exprtrees, totalLoopNums, prefactors, varK),
        var=(K, T),
        dof=dof,
        obs=obs,
    )
end

"""Build variable pools for the density."""
function density_mc_variables(mcparam::UEG.ParaMC, alpha::Float64)
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Offset T pool by 1 for fixed external times (instantaneous Green's function ⟹ τin = τout⁺)
    T = Continuous(0.0, mcparam.β; offset=1, alpha=alpha)
    return (K, T)
end

"""Measurement for multiple diagram trees (counterterm partitions)."""
function measure(vars, obs, weights, config)
    # Measure the weight of each partition
    for o in 1:(config.N)
        obs[o] += weights[o]
    end
    return
end

"""Integrand for the total density n."""
function integrand(vars, config)
    # We sample internal momentum/times, and external momentum index
    K, T = vars
    R, Theta, Phi = K

    # Unpack userdata
    mcparam, exprtrees, totalLoopNums, prefactors, varK = config.userdata

    @debug "totalLoopNums = $totalLoopNums" maxlog = 3
    @debug "config.N = $(config.N)" maxlog = 3

    # Evaluate the integrand for each partition
    integrand = Vector(undef, config.N)
    for i in 1:(config.N)
        phifactor = 1.0
        for j in 1:totalLoopNums[i]  # config.dof[i][1]
            r = R[j] / (1 - R[j])
            θ = Theta[j]
            ϕ = Phi[j]
            varK[1, j] = r * sin(θ) * cos(ϕ)
            varK[2, j] = r * sin(θ) * sin(ϕ)
            varK[3, j] = r * cos(θ)
            phifactor *= r^2 * sin(θ) / (1 - R[j])^2
        end
        # @assert T.data[1] == 0
        @debug "K = $(varK)" maxlog = 3
        @debug "T = $(T.data)" maxlog = 3

        # Evaluate the expression tree (additional = mcparam)
        ExprTree.evalKT!(exprtrees[i], varK, T.data, mcparam)

        # Evaluate the density integrand n for this partition
        root = exprtrees[i].root[1]  # there is only one root per partition
        weight = exprtrees[i].node.current
        integrand[i] = phifactor * prefactors[i] * weight[root]
    end
    return integrand
end

"""MC integration of the total density."""
function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    # Debug mode
    if isinteractive()
        ENV["JULIA_DEBUG"] = Main
    end

    # Total loop order N
    orders = [0, 1, 2, 3]
    max_order = maximum(orders)
    sort!(orders)

    # Settings
    alpha = 3.0
    print = 0
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 1e10

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Remove Fock insertions?
    isFock = false

    # UEG parameters for MC integration
    param = ParaMC(;
        order=max_order,
        rs=1.0,
        beta=40.0,
        mass2=1.0,
        isDynamic=false,
        isFock=isFock,  # remove Fock insertions
    )
    @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)"

    # Build diagram/expression trees for the occupation number to order
    # ξᴺ in the renormalized perturbation theory (includes CTs in μ and λ)
    partitions, diagparams, diagtrees, exprtrees = build_occupation_with_ct(
        orders;
        renorm_mu=renorm_mu,
        renorm_lambda=renorm_lambda,
        isFock=isFock,
    )

    println("Integrating partitions: $partitions")
    println("diagtrees: $diagtrees")
    println("exprtrees: $exprtrees")

    res = integrate_density_with_ct(
        param,
        diagparams,
        exprtrees;
        alpha=alpha,
        neval=neval,
        print=print,
        solver=solver,
    )

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end
    if isFock
        ct_string *= "_noFock"
    end

    # Save to JLD2 on main thread
    if !isnothing(res)
        savename =
            "results/data/density_n=$(param.order)_rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_neval=$(neval)_$(solver)$(ct_string)_no_green4"
        jldopen("$savename.jld2", "a+"; compress=true) do f
            key = "$(UEG.short(param))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (orders, param, partitions, res)
        end
    end
end

main()
