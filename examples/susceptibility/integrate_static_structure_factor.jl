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

"""DiagGen"""

function swap_extT(taus, extTout_old, extTout_new)
    new_taus = Vector{eltype(taus)}(undef, length(taus))
    for (i, tau) in enumerate(taus)
        if tau == extTout_old
            new_taus[i] = extTout_new
        elseif tau == extTout_new
            new_taus[i] = extTout_old
        else
            new_taus[i] = tau
        end
    end
    return Tuple(new_taus)
end

function update_id_taus(id::T, extTout_old, extTout_new) where {T<:DiagramId}
    # Update extT field of non-generic IDs
    @assert hasproperty(id, :extT)
    new_taus = swap_extT(id.extT, extTout_old, extTout_new)
    return reconstruct(id; extT=new_taus)
end

function update_id_taus(id::GenericId, extTout_old, extTout_new)
    # Reconstruct child ID with new/old extTout indices swapped
    @assert hasproperty(id, :extra)
    if hasproperty(id.extra, :t)
        # Update extra.t field of generic IDs
        new_taus = swap_extT(id.extra.t, extTout_old, extTout_new)
        return reconstruct(id; extra=(t=new_taus,))
    end
    return id
end

function swap_extTout(diag, extTout_new=diag.id.para.totalTauNum)
    # Get outgoing external time of the diagram
    extTout_old = diag.id isa GenericId ? diag.id.extra.t[2] : diag.id.extT[2]
    # Remap extTout_old ↦ exTout_new
    new_root = deepcopy(diag)
    for node in PreOrderDFS(new_root)
        # Update times in child IDs
        for (i, child) in enumerate(children(node))
            node.subdiagram[i].id = update_id_taus(child.id, extTout_old, extTout_new)
        end
    end
    # Update times in root ID
    new_root.id = update_id_taus(new_root.id, extTout_old, extTout_new)
    return new_root
end

function build_susceptibility_with_ct(
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
        @debug "Partition (n_inner_loop, n_ct_mu, n_ct_lambda): $p"
        diagparam, diagtree = build_diagtree(; n_inner_loop=p[1])
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

function build_diagtree(; n_inner_loop=1)
    @assert n_inner_loop > 0
    # χ has one external momentum loop
    n_outer_loop = 1
    # Interaction order for χ
    order = n_inner_loop - 1
    # External momentum and times
    extT = (1, n_inner_loop + 1)
    extK = DiagTree.getK(n_inner_loop + n_outer_loop, 1)

    # χ diagram parameters
    DiagTree.uidreset()
    diagparam = DiagParaF64(;
        type=PolarDiag,  # χ can be viewed diagram as an improper polarization diagram
        hasTau=true,
        innerLoopNum=n_inner_loop,
        totalLoopNum=n_inner_loop + n_outer_loop,
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa interaction
    )
    chi_id = DiagTree.PolarId(diagparam, ChargeCharge; k=extK, t=extT)

    # Get Dyson subterms of χ by generating all integer compositions `c` of n_inner_loop.
    # For example, c = [3, 2, 1] generates the term Π_2(14) V(44) Π_1(46) V(66) Π_0(67) in χ_5.
    chi_subtrees = []
    for (j, c) in enumerate(DiagGen.integer_compositions(n_inner_loop))
        first_tau_indices = accumulate(+, [1; c[1:(end - 1)]])
        first_loop_indices = accumulate(+, [1 + n_outer_loop; c[1:(end - 1)]])
        taus = [(first_tau_indices[i], first_tau_indices[i] + c[i]) for i in eachindex(c)]
        @debug "\nComposition #$j: $c"
        @debug "$(["Π_$(c[i] - 1)$(taus[i])" for i in eachindex(c)])"
        # Construct each Π[i] subdiagram
        pis = []
        for i in eachindex(c)
            pi_param = DiagParaF64(;
                type=PolarDiag,
                hasTau=true,
                innerLoopNum=c[i],
                firstTauIdx=first_tau_indices[i],
                firstLoopIdx=first_loop_indices[i],
                totalLoopNum=n_inner_loop + n_outer_loop,
                filter=[Proper, NoHartree],
                interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],
            )
            pi_c_df = Parquet.polarization(pi_param, extK; name=Symbol("Π_$(c[i] - 1)"))
            # Merge spins, Π_σ = Π↑↑ + Π↑↓
            pi_c = mergeby(pi_c_df, :extT).diagram[1]
            # If we had multiple spin channels to merge, we obtain a GenericId at root level. Move the 
            # extra field to an extra.t NamedTuple to match the convention in function update_id_taus.
            if size(pi_c_df)[1] > 1
                pi_c.id = reconstruct(pi_c.id; extra=(t=pi_c.id.extra,))
            end
            # To match the time ordering & extT conventions for χ, we need to swap the
            # outgoing external time of each Π[c] to the maximum contained time label.
            pi_c_swap = swap_extTout(pi_c)
            push!(pis, pi_c_swap)
        end
        # Construct diagram tree for χ[c]:
        # T = Π[1] ⨂ Π[2] ⨂ ⋯ ⨂ Π[length(c)] ⨂ (V_λ)^(length(c) - 1)
        v_param = reconstruct(diagparam; type=Ver4Diag, innerLoopNum=0, firstLoopIdx=1)
        v_taus = [(t, t) for t in first_tau_indices[2:end]]
        v_ids = [
            BareInteractionId(
                v_param,
                ChargeCharge,
                Instant,
                [0, 0, 0, 0];
                k=extK,
                t=tau_pair,
                permu=Di,
            ) for tau_pair in v_taus
        ]
        # NOTE: We have a Mahan-type Dyson equation in terms of -V because
        # in the N&O convention, χ = Π / (1 + V Π) = Π / (1 - (-V) Π)
        vs = [DiagramF64(v_id; name=:V, factor=-1.0) for v_id in v_ids]
        @debug "$vs"
        @debug "$pis"
        subtree = DiagramF64(chi_id, Prod(), [vs; pis]; name=Symbol("χ^{$c}_$order"))
        push!(chi_subtrees, subtree)
    end

    # Construct tree for full χ
    chi = DiagramF64(chi_id, Sum(), chi_subtrees; name=Symbol("χ_$order"))
    # For convenient access to the external times in the integration step, we swap the
    # outgoing time index from the maximum loop order to 2 (T[1] = τin, T[2] = τout)
    chi_swap = swap_extTout(chi, 2)
    @assert chi_swap.id.extT == (1, 2)
    @debug "\nDiagTree:\n" * repr_tree(diagtree)
    return diagparam, chi_swap
end

"""UEG_MC"""

function integrate_static_structure_with_ct(
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
    (K, T, ExtKidx) = static_structure_mc_variables(mcparam, n_kgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T), shape(ExtKidx)
    # We do not integrate the external times and χ is dynamic, hence n_τ = totalTauNum - 2.
    dof = [[p.innerLoopNum, p.totalTauNum - 2, 1] for p in diagparams]
    println("Integration DOF: $dof")

    # UEG SOSEM diagram observables are a function of |k| only (equal-time)
    obs = repeat([zeros(n_kgrid)], length(dof))  # observable for each partition

    # Fix external times to zero (COM coordinates and instantaneous observable)
    T.data[1] = 0     # τin  = 0
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

"""Build variable pools for the static structure factor S(q)."""
function static_structure_mc_variables(mcparam::UEG.ParaMC, n_kgrid::Int, alpha::Float64)
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Offset T pool by 2 for fixed external times τin & τout
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

"""Integrand for the static structure factor S(q)."""
function integrand(vars, config)
    # We sample internal momentum/times, and external momentum index
    K, varT, ExtKidx = vars
    R, Theta, Phi = K

    # Unpack userdata
    mcparam, exprtrees, innerLoopNums, prefactors, varK, kgrid = config.userdata

    # External momentum via random index into kgrid (wlog, we place it along the x-axis)
    ik = ExtKidx[1]
    varK[1, 1] = kgrid[ik]

    @assert varT.data[1] == 0
    @assert varT.data[2] == 1e-8

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

        # Evaluate the static structure factor S(q) for this partition
        root = exprtrees[i].root[1]  # there is only one root per partition
        weight = exprtrees[i].node.current
        integrand[i] = phifactor * prefactors[i] * weight[root]
    end
    return integrand
end

"""Generate and integrate expression trees for C⁽¹⁾ˡ."""
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

    # Total loop orders N (interaction order + 1). N = 1 ⟹ χ_0, etc.
    orders = [1, 2, 3, 4]
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
    Nk, korder = 4, 3
    kgrid =
        CompositeGrid.LogDensedGrid(
            :uniform,
            [0.0, 3 * param.kF],
            [0.0, 2 * param.kF],
            Nk,
            minK,
            korder,
        ).grid

    # Dimensionless k-grid
    # k_kf_grid = kgrid / param.kF

    # Build diagram/expression trees for the susceptibility to order
    # ξᴺ in the renormalized perturbation theory (includes CTs in μ and λ)
    partitions, diagparams, diagtrees, exprtrees = build_susceptibility_with_ct(
        orders;
        renorm_mu=renorm_mu,
        renorm_lambda=renorm_lambda,
        isFock=isFock,
    )

    println("Integrating partitions: $partitions")
    println("diagtrees: $diagtrees")
    println("exprtrees: $exprtrees")

    res = integrate_static_structure_with_ct(
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
