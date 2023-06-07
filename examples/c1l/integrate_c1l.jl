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

function build_c1l_with_ct(orders; renorm_mu=true, renorm_lambda=true, isFock=false)
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
    # Interaction order for V χ V
    order = n_inner_loop + 1
    # External momentum and times
    extT = (1, n_inner_loop + 1)
    extK = DiagTree.getK(n_inner_loop + n_outer_loop, 1)

    # Diagram parameters
    DiagTree.uidreset()
    diagparam = DiagParaF64(;
        type=PolarDiag,  # χ can be viewed diagram as an improper polarization diagram
        hasTau=true,
        innerLoopNum=n_inner_loop,
        totalLoopNum=n_inner_loop + n_outer_loop,
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa interaction
    )
    c1l_id = DiagTree.PolarId(diagparam, ChargeCharge; k=extK, t=extT)

    # Get Dyson subterms of χ by generating all integer compositions `c` of n_inner_loop.
    # For example, c = [3, 2, 1] generates the term Π_2(14) V(44) Π_1(46) V(66) Π_0(67) in χ_5.
    c1l_subtrees = []
    for (j, c) in enumerate(DiagGen.integer_compositions(n_inner_loop))
        first_tau_indices = accumulate(+, [1; c[1:(end - 1)]])
        first_loop_indices = accumulate(+, [1 + n_outer_loop; c[1:(end - 1)]])
        taus = [(first_tau_indices[i], first_tau_indices[i] + c[i]) for i in eachindex(c)]
        @debug "\nComposition #$j: $c"
        @debug "$(["Π_$(c[i] - 1)$(taus[i])" for i in eachindex(c)])"
        # Construct each Π[i] subdiagram
        pis = []
        for i in eachindex(c)
            # inner_tau_nums = c .- 1
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
        # Diagram parameters for interaction lines
        v_param = reconstruct(diagparam; type=Ver4Diag, innerLoopNum=0, firstLoopIdx=1)
        # Construct diagram tree for χ[c]:
        # T = Π[1] ⨂ Π[2] ⨂ ⋯ ⨂ Π[length(c)] ⨂ (V_λ)^(length(c) - 1)
        v_lambda_taus = [(t, t) for t in first_tau_indices[2:end]]
        v_lambda_ids = [
            BareInteractionId(
                v_param,
                ChargeCharge,
                Instant,
                [0, 0, 0, 0];
                k=extK,
                t=tau_pair,
                permu=Di,
            ) for tau_pair in v_lambda_taus
        ]
        # The local moment has two additional interaction lines.
        # We treat one as a fixed bare interaction V, and expand the other as V[V_λ].
        v_bare_in_id = BareInteractionId(
            v_param,
            ChargeCharge,
            Instant,
            [0, 0, 0, 1];  # Mark this interaction line as bare
            k=extK,
            t=(1, 1),
            permu=Di,
        )
        v_lambda_out_id = BareInteractionId(
            v_param,
            ChargeCharge,
            Instant,
            [0, 0, 0, 0];  # Mark this interaction line as expandable V[V_λ]
            k=extK,
            t=(n_inner_loop + 1, n_inner_loop + 1),
            permu=Di,
        )
        # NOTE: We have a Mahan-type Dyson equation in terms of -V because
        # in the N&O convention, χ = Π / (1 + V Π) = Π / (1 - (-V) Π)
        v_bare_in = DiagramF64(v_bare_in_id; name=:V, factor=-1.0)
        v_lambdas = [DiagramF64(v_id; name=:V_λ, factor=-1.0) for v_id in v_lambda_ids]
        v_lambda_out = DiagramF64(v_lambda_out_id; name=Symbol("V[V_λ]"), factor=-1.0)
        @debug "$v_lambdas"
        @debug "$pis"
        # C⁽¹⁾ˡ[c] = V χ[c] V[V_λ]
        subtree = DiagramF64(
            c1l_id,
            Prod(),
            [v_bare_in; v_lambdas; pis; v_lambda_out];
            name=Symbol("C⁽¹⁾ˡ_$order,$c"),
        )
        push!(c1l_subtrees, subtree)
    end

    # Construct tree for full C⁽¹⁾ˡ = V χ V
    c1l = DiagramF64(c1l_id, Sum(), c1l_subtrees; name=Symbol("C⁽¹⁾ˡ_$order"))
    # For convenient access to the external times in the integration step, we swap the
    # outgoing time index from the maximum loop order to 2 (T[1] = τin, T[2] = τout)
    c1l_swap = swap_extTout(c1l, 2)
    @assert c1l_swap.id.extT == (1, 2)
    @debug "\nDiagTree:\n" * repr_tree(diagtree)
    return diagparam, c1l_swap
end

"""UEG_MC"""

function integrate_c1l_with_ct(
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

    # Temporary array for K-variables
    # We use the maximum necessary loop basis size for K pool.
    maxloops = maximum(totalLoopNums)
    varK = zeros(3, maxloops)

    # Build adaptable MC integration variables
    (K, T) = c1l_mc_variables(mcparam, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T)
    # We do not integrate the external times and χ is dynamic, hence n_τ = totalTauNum - 2.
    dof = [[p.totalLoopNum, p.totalTauNum - 2] for p in diagparams]
    println("Integration DOF: $dof")

    # Local moment is a scalar
    obs = zeros(length(dof))  # observable for each partition

    # Fix external times to zero (COM coordinates and instantaneous observable)
    T.data[1] = 0     # τin  = 0
    T.data[2] = 1e-8  # τout = 0⁺

    # Thomas-Fermi energy squared
    eTF2 = mcparam.qTF^4 / (2 * mcparam.me)^2

    # Phase-space factors
    phase_factors = [1.0 / (2π)^(mcparam.dim * nl) for nl in totalLoopNums]

    # Total prefactors (including spin sum factor ∑_{σ'} = 2S+1 = 2)
    # The extra minus sign from the fermion loop (-1)^F = -1 is already
    # included in the N&O definition of the susceptibility, χ_N&O = -χ_Mahan.
    prefactors = mcparam.spin * phase_factors / eTF2

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

"""Build variable pools for the local moment C⁽¹⁾ˡ."""
function c1l_mc_variables(mcparam::UEG.ParaMC, alpha::Float64)
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Offset T pool by 2 for fixed external times τin & τout
    T = Continuous(0.0, mcparam.β; offset=2, alpha=alpha)
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

"""Integrand for the local moment C⁽¹⁾ˡ."""
function integrand(vars, config)
    # We sample internal momentum/times, and external momentum index
    K, varT = vars
    R, Theta, Phi = K

    # Unpack userdata
    mcparam, exprtrees, totalLoopNums, prefactors, varK = config.userdata

    @assert varT.data[1] == 0
    @assert varT.data[2] == 1e-8
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
        @debug "K = $(varK)" maxlog = 3
        @debug "T = $(varT.data)" maxlog = 3

        # Evaluate the expression tree (additional = mcparam)
        # NOTE: We use UEG_MC propagators to mark the outer interaction as bare
        ExprTree.evalKT!(
            exprtrees[i],
            varK,
            varT.data,
            mcparam;
            eval=UEG_MC.Propagators.eval,
        )

        # Evaluate the C⁽¹⁾ˡ integrand for this partition
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

    # Total loop orders N (interaction order + 1). N = 1 ⟹ V χ_0 V, etc.
    orders = [1, 2, 3, 4]
    # orders = [5]
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
        rs=2.0,
        beta=40.0,
        mass2=0.4,
        isDynamic=false,
        isFock=isFock,  # remove Fock insertions
    )
    @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)"

    # Build diagram/expression trees for C⁽¹⁾ˡ to order
    # ξᴺ in the renormalized perturbation theory (includes CTs in μ and λ)
    partitions, diagparams, diagtrees, exprtrees = build_c1l_with_ct(
        orders;
        renorm_mu=renorm_mu,
        renorm_lambda=renorm_lambda,
        isFock=isFock,
    )

    println("Integrating partitions: $partitions")
    println("diagtrees: $diagtrees")
    println("exprtrees: $exprtrees")

    res = integrate_c1l_with_ct(
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
            "results/data/c1l_n=$(param.order)_rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_neval=$(neval)_$(solver)$(ct_string)"
        jldopen("$savename.jld2", "a+"; compress=true) do f
            key = "$(UEG.short(param))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (orders, param, partitions, res)
        end
    end
    return
end

main()
