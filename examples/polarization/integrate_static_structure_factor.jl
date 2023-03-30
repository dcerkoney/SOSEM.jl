using AbstractTrees
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

function build_polarization_with_ct(
    orders;
    renorm_mu=true,
    renorm_lambda=true,
    isFock=false,
)
    DiagTree.uidreset()
    valid_partitions = Vector{PartitionType}()
    diagparams = Vector{DiagParaF64}()
    diagtrees = Vector{DiagramF64}()
    exprtrees = Vector{ExprTreeF64}()
    # Build all counterterm partitions at the given orders (lowest order (Π₀) is N = 1)
    partitions = DiagGen.counterterm_partitions(
        orders;
        n_lowest=1,
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
            isFock && DiagTree.removeHartreeFock!(dλ_dμ_diagtree)
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
    # Polarization diagram parameters
    #
    # NOTE: Differentiation and Fock filter do not commute, so we need to manually zero
    #       out the Fock insertions after differentiation (via DiagTree.removeHartreeFock!)
    DiagTree.uidreset()
    diagparam = DiagParaF64(;
        type=PolarDiag,
        hasTau=true,
        innerLoopNum=n_loop,
        filter=[Proper, NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa interaction
    )
    # Build diagram tree dataframe
    diag_df = Parquet.polarization(diagparam; name=:Π)
    # Merge spins, Π_σ = Π↑↑ + Π↑↓
    diagtree = mergeby(diag_df.diagram)[1]
    @debug "\nDiagTree:\n" * repr_tree(diagtree)
    return diagparam, diagtree
end

function integrate_polarization_with_ct(
    mcparam::UEG.ParaMC,
    diagparams::Vector{DiagParaF64},
    exprtrees::Vector{ExprTreeF64};
    kgrid=[0.0],
    alpha=3.0,
    neval=1e5,
    print=-1,
    solver=:vegasmc,
)
    @assert all(p.totalTauNum ≤ mcparam.order + 1 for p in diagparams)

    # We assume that each partition expression tree has a single root
    @assert all(length(et.root) == 1 for et in exprtrees)

    # List of expression tree roots, external times, and inner
    # loop numbers for each tree (to be passed to integrand)
    # roots = [et.root[1] for et in exprtrees]
    innerLoopNums = [p.innerLoopNum for p in diagparams]

    # Grid sizes
    n_kgrid = length(kgrid)

    # Temporary array for combined K-variables [ExtK, K].
    # We use the maximum necessary loop basis size for K pool.
    maxloops = maximum(p.totalLoopNum for p in diagparams)
    varK = zeros(3, maxloops)

    # Build adaptable MC integration variables
    (K, T, ExtKidx) = polarization_mc_variables(mcparam, n_kgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T), shape(ExtKidx)
    # We do not integrate the external times and Π is dynamic, hence n_τ = totalTauNum - 2
    dof = [[p.innerLoopNum, p.totalTauNum - 2, 1] for p in diagparams]
    println("Integration DOF: $dof")

    # UEG SOSEM diagram observables are a function of |k| only (equal-time)
    obs = repeat([zeros(n_kgrid)], length(dof))  # observable for each partition

    # External times are fixed for left/right measurement of the discontinuity at τ = 0
    T.data[1] = 0     # τin = 0 (COM coordinates)
    T.data[2] = 1e-8  # τout = 0⁺

    # Phase-space factors
    phase_factors = [1.0 / (2π)^(mcparam.dim * nl) for nl in innerLoopNums]

    # Total prefactors (including outer spin sum factor S=2)
    n0 = mcparam.kF^3 / 3π^2
    prefactors = -(mcparam.spin / n0) * phase_factors

    return integrate(
        integrand;
        solver=solver,
        measure=measure,
        neval=neval,
        print=print,
        # MC config kwargs
        userdata=(mcparam, exprtrees, innerLoopNums, prefactors, varK, kgrid),
        var=(K, T, ExtKidx),
        dof=dof,
        obs=obs,
    )
end

"""Build variable pools for the polarization."""
function polarization_mc_variables(mcparam::UEG.ParaMC, n_kgrid::Int, alpha::Float64)
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Offset T pool by 2 for fixed/binned external times τin/τout
    T = Continuous(0.0, mcparam.β; offset=2, alpha=alpha)
    # Bin in external momentum & time
    ExtKidx = Discrete(1, n_kgrid; alpha=alpha)
    return (K, T, ExtKidx)
end

"""Measurement for multiple diagram trees (counterterm partitions)."""
function measure(vars, obs, weights, config)
    ik = vars[3][1]  # ExtK bin index
    # Measure the weight of each partition
    for o in 1:(config.N)
        obs[o][ik] += weights[o]
    end
    return
end

"""Integrand for the polarization Π."""
function integrand(vars, config)
    # We sample internal momentum/times, and external momentum index
    K, varT, ExtKidx = vars
    R, Theta, Phi = K

    # Unpack userdata
    mcparam, exprtrees, innerLoopNums, prefactors, varK, kgrid = config.userdata

    # External momentum via random index into kgrid (wlog, we place it along the x-axis)
    ik = ExtKidx[1]
    varK[1, 1] = kgrid[ik]

    # Evaluate the integrand for each partition
    integrand = Vector(undef, config.N)
    for i in 1:(config.N)
        phifactor = 1.0
        for j in 1:innerLoopNums[i]  # config.dof[i][1]
            r = R[j] / (1 - R[j])
            θ = Theta[j]
            ϕ = Phi[j]
            varK[1, j + 1] = r * sin(θ) * cos(ϕ)
            varK[2, j + 1] = r * sin(θ) * sin(ϕ)
            varK[3, j + 1] = r * cos(θ)
            phifactor *= r^2 * sin(θ) / (1 - R[j])^2
        end

        @debug "K = $(varK)" maxlog = 3
        @debug "ik = $ik" maxlog = 3
        @debug "ExtK = $(kgrid[ik])" maxlog = 3

        # Evaluate the expression tree (additional = mcparam)
        ExprTree.evalKT!(exprtrees[i], varK, varT.data, mcparam)

        # Evaluate the polarization integrand Π for this partition
        root = exprtrees[i].root[1]  # there is only one root per partition
        weight = exprtrees[i].node.current
        integrand[i] = phifactor * prefactors[i] * weight[root]
    end
    return integrand
end

"""MC integration of the polarization."""
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
    orders = [1, 2, 3]
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

    # K-mesh for measurement
    minK = 0.2 * param.kF
    Nk, korder = 4, 7
    kgrid =
        CompositeGrid.LogDensedGrid(
            :uniform,
            [0.0, 3 * param.kF],
            [0.0, 2 * param.kF],
            Nk,
            minK,
            korder,
        ).grid

    # Build diagram/expression trees for the polarization to order
    # ξᴺ in the renormalized perturbation theory (includes CTs in μ and λ)
    partitions, diagparams, diagtrees, exprtrees = build_polarization_with_ct(
        orders;
        renorm_mu=renorm_mu,
        renorm_lambda=renorm_lambda,
        isFock=isFock,
    )

    println("Integrating partitions: $partitions")
    println("diagtrees: $diagtrees")
    println("exprtrees: $exprtrees")

    res = integrate_polarization_with_ct(
        param,
        diagparams,
        exprtrees;
        kgrid=kgrid,
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
            "results/data/static_structure_factor_n=$(param.order)_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver)$(ct_string)"
        jldopen("$savename.jld2", "a+"; compress=true) do f
            key = "$(UEG.short(param))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (orders, param, kgrid, partitions, res)
        end
    end
end

main()
