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
using LinearAlgebra
using Parameters
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np

"""
Exact expression for the Fock self-energy
in terms of the dimensionless Lindhard function.
"""
function fock_self_energy_exact(k, p::ParaMC)
    # The (dimensionful) value at k = 0 is minus the Thomas-Fermi energy
    eTF = p.qTF^2 / (2 * p.me)
    return -eTF * lindhard(k / p.kF)
end

"""Constructs diagram parameters for the exchange self-energy."""
function exchange_param(order=1)
    # Instantaneous interactions (interactionTauNum = 1) 
    # => innerLoopNum = totalTauNum = order
    return DiagParaF64(;
        type=SigmaDiag,
        hasTau=true,
        firstTauIdx=1,
        innerLoopNum=order,
        totalTauNum=order,  # includes outgoing external time
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],
    )
end

function build_sigma_x_with_ct(orders=[1]; renorm_mu=renorm_mu, renorm_lambda=renorm_lambda)
    DiagTree.uidreset()
    valid_partitions = Vector{PartitionType}()
    diagparams = Vector{DiagParaF64}()
    diagtrees = Vector{DiagramF64}()
    exprtrees = Vector{ExprTreeF64}()
    # Build all counterterm partitions at the given orders (lowest order (Fock) is N = 1)
    n_min, n_max = minimum(orders), maximum(orders)
    for p in DiagGen.counterterm_partitions(
        n_min,
        n_max;
        n_lowest=1,
        renorm_mu=renorm_mu,
        renorm_lambda=renorm_lambda,
    )
        # Build diagram tree for this partition
        @debug "Partition (n_loop, n_ct_mu, n_ct_lambda): $p"
        diagparam, diagtree = build_diagtree(; n_loop=p[1])

        # Build tree with counterterms (∂λ(∂μ(DT))) via automatic differentiation
        dμ_diagtree = DiagTree.derivative([diagtree], BareGreenId, p[2]; index=1)
        dλ_dμ_diagtree = DiagTree.derivative(dμ_diagtree, BareInteractionId, p[3]; index=2)
        if isempty(dλ_dμ_diagtree)
            @warn("Ignoring partition $p with no diagrams")
            continue
        end
        @debug "\nDiagTree:\n" * repr_tree(dλ_dμ_diagtree)
        # Optimize the tree
        # DiagTree.optimize!(dλ_dμ_diagtree)
        # Compile to expression tree and save results for this partition
        exprtree = ExprTree.build(dλ_dμ_diagtree)
        push!(valid_partitions, p)
        push!(diagparams, diagparam)
        push!(exprtrees, exprtree)
        append!(diagtrees, dλ_dμ_diagtree)
    end
    return valid_partitions, diagparams, diagtrees, exprtrees
end

function build_diagtree(; n_loop=1)
    DiagTree.uidreset()

    # Generate the diagram and expression trees
    diagparam = exchange_param(n_loop)

    # Momentum loop basis
    nk = diagparam.totalLoopNum
    k = DiagTree.getK(nk, 1)
    q = DiagTree.getK(nk, 2)

    # Σₓ is instantaneous
    extT = (1, 1)
    g_param = DiagParaF64(;
        type=GreenDiag,
        hasTau=true,
        firstLoopIdx=3,           # k and q already taken
        innerLoopNum=n_loop - 1,  # k and q already taken
        firstTauIdx=2,            # 1 external time (instantaneous Σ)
        totalTauNum=n_loop,       # includes outgoing external time
        totalLoopNum=n_loop + 1,  # Part of Sigma diagram
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa interaction
        filter=[NoHartree],
    )
    G = Parquet.green(g_param, k - q, extT; name=:G)

    # Add outer lbare Coulomb interaction line
    v_param = reconstruct(diagparam; type=Ver4Diag, innerLoopNum=0, firstLoopIdx=1)
    v_id = BareInteractionId(
        v_param,
        ChargeCharge,
        Instant,
        [0, 0, 0, 1];
        k=q,
        t=extT,
        permu=Di,
    )
    V = DiagramF64(v_id; name=:V, factor=-1.0)  # Factor of (-1) from Feynman rules

    # Build the full exchange self-energy diagram
    sigma_id = DiagTree.SigmaId(diagparam, Instant; k=k, t=extT)
    diagtree = DiagramF64(sigma_id, Prod(), [V, G]; name=:Σx)

    @debug "\nDiagTree:\n" * repr_tree(diagtree)
    return diagparam, diagtree
end

function integrate_sigma_x_with_ct(
    mcparam::UEG.ParaMC,
    diagparams::Vector{DiagParaF64},
    exprtrees::Vector{ExprTreeF64};
    kgrid=[0.0],
    alpha=3.0,
    neval=1e5,
    print=-1,
    solver=:vegasmc,
)
    # We assume that each partition expression tree has a single root
    @assert all(length(et.root) == 1 for et in exprtrees)

    # List of expression tree roots, external times, and inner
    # loop numbers for each tree (to be passed to integrand)
    # roots = [et.root[1] for et in exprtrees]
    innerLoopNums = [p.innerLoopNum for p in diagparams]

    # Grid size
    n_kgrid = length(kgrid)

    # Temporary array for combined K-variables [ExtK, K].
    # We use the maximum necessary loop basis size for K pool.
    maxloops = maximum(p.totalLoopNum for p in diagparams)
    varK = zeros(3, maxloops)

    # Build adaptable MC integration variables
    (K, T, ExtKidx) = exchange_mc_variables(mcparam, n_kgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T), shape(ExtKidx)
    # We do not integrate the incoming external time and Σₓ is instantaneous, hence n_τ = totalTauNum - 1
    dof = [[p.innerLoopNum, p.totalTauNum - 1, 1] for p in diagparams]
    println("Integration DOF: $dof")

    # UEG SOSEM diagram observables are a function of |k| only (equal-time)
    obs = repeat([zeros(n_kgrid)], length(dof))  # observable for each partition

    # External times are fixed for left/right measurement of the discontinuity at τ = 0
    T.data[1] = 0  # τin = 0 (= τout⁺)

    # We non-dimensionalize the result via division by the Thomas-Fermi energy
    eTF = mcparam.qTF^2 / (2 * mcparam.me)

    # Phase-space factors
    phase_factors = [1.0 / (2π)^(mcparam.dim * nl) for nl in innerLoopNums]

    # Total prefactors
    prefactors = -phase_factors / eTF

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

"""Build variable pools for the exchange self-energy integration."""
function exchange_mc_variables(mcparam::UEG.ParaMC, n_kgrid::Int, alpha::Float64)
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Offset T pool by 1 for fixed external times (instantaneous self-energy ⟹ τin = τout⁺)
    T = Continuous(0.0, mcparam.β; offset=1, alpha=alpha)
    # Bin in external momentum
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

"""Integrand for the exchange self-energy non-dimensionalized by E²_{TF} ~ q⁴_{TF}.."""
function integrand(vars, config)
    # We sample internal momentum/times, and external momentum index
    K, T, ExtKidx = vars
    R, Theta, Phi = K

    # Unpack userdata
    mcparam, exprtrees, innerLoopNums, prefactors, varK, kgrid = config.userdata

    # Evaluate the integrand for each partition
    integrand = Vector(undef, config.N)
    for i in 1:(config.N)
        # External momentum via random index into kgrid (wlog, we place it along the x-axis)
        ik = ExtKidx[1]
        varK[1, 1] = kgrid[ik]

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
        @assert T.data[1] == 0

        @debug "K = $(varK)" maxlog = 3
        @debug "ik = $ik" maxlog = 3
        @debug "ExtK = $(kgrid[ik])" maxlog = 3

        # Evaluate the expression tree (additional = mcparam)
        # NOTE: We use UEG_MC propagators to mark the outer interaction as bare
        ExprTree.evalKT!(exprtrees[i], varK, T.data, mcparam; eval=UEG_MC.Propagators.eval)

        # Evaluate the exchange integrand Σx(k) for this partition
        root = exprtrees[i].root[1]  # there is only one root per partition
        weight = exprtrees[i].node.current
        integrand[i] = phifactor * prefactors[i] * weight[root]
    end
    return integrand
end

"""MC integration of the exchange self-energy"""
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

    # Total loop order N (Fock self-energy is N = 1)
    # orders = [1, 2, 3, 4]
    orders = [5]
    max_order = maximum(orders)

    # Settings
    alpha = 3.0
    print = 0
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 5e10

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # UEG parameters for MC integration
    param = ParaMC(;
        order=max_order,
        rs=2.0,
        beta=40.0,
        mass2=0.4,
        isDynamic=false,
        isFock=false,
    )
    @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)"

    # # Dimensionless k-mesh for measurement
    # minK = 0.04
    # Nk, korder = 4, 4
    # k_kf_grid = CompositeGrid.LogDensedGrid(:cheb, [0.0, 3.0], [0.0, 1.0], Nk, minK, korder)

    # # Dimensionful k-grid
    # kgrid = k_kf_grid * param.kF

    # K-mesh for measurement
    minK = 0.225 * param.kF
    Nk, korder = 4, 5
    kgrid =
        CompositeGrid.LogDensedGrid(
            :uniform,
            [0.0, 3 * param.kF],
            [param.kF],
            Nk,
            minK,
            korder,
        ).grid

    # minK = 0.1 * param.kF
    # Nk, korder = 4, 7
    # kleft =
    #     CompositeGrid.LogDensedGrid(
    #         :uniform,
    #         [0.0, param.kF - 1e-8],
    #         [param.kF - 1e-8],
    #         Nk,
    #         minK,
    #         korder,
    #     ).grid
    # kright =
    #     CompositeGrid.LogDensedGrid(
    #         :uniform,
    #         [param.kF + 1e-8, 3 * param.kF],
    #         [param.kF + 1e-8],
    #         Nk,
    #         minK,
    #         korder,
    #     ).grid
    # kgrid = [kleft; kright]

    # # Dimensionless k-grid
    k_kf_grid = kgrid / param.kF

    # # Reduced number of kpoint
    # kgrid = kgrid[k_kf_grid .≤ 2.1]
    # k_kf_grid = k_kf_grid[k_kf_grid .≤ 2.1]

    # Build diagram/expression trees for the exchange self-energy to order
    # ξᴺ in the renormalized perturbation theory (includes CTs in μ and λ)
    partitions, diagparams, diagtrees, exprtrees =
        build_sigma_x_with_ct(orders; renorm_mu=renorm_mu, renorm_lambda=renorm_lambda)

    println("Integrating partitions: $partitions")
    println("diagtrees: $diagtrees")
    println("exprtrees: $exprtrees")

    # # Check the diagram trees
    # for (i, d) in enumerate(diagtrees)
    #     println("\nDiagram tree #$i, partition P = $(partitions[i]):")
    #     print_tree(d)
    #     plot_tree(d)
    #     # println(diagparams[i])
    # end
    # return

    res = integrate_sigma_x_with_ct(
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

    # Save to JLD2 on main thread
    if !isnothing(res)
        savename =
            "results/data/exchange/sigma_x_n=$(param.order)_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver)$(ct_string)_new"
        jldopen("$savename.jld2", "a+"; compress=true) do f
            key = "$(UEG.short(param))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            # Convert result to dictionary
            datadict = Dict{eltype(partitions),Any}()
            if length(diagparams) == 1
                avg, std = res.mean, res.stdev
                data = measurement.(avg, std)
                datadict[partitions[1]] = data
            else
                for o in eachindex(diagparams)
                    avg, std = res.mean[o], res.stdev[o]
                    data = measurement.(avg, std)
                    datadict[partitions[o]] = data
                end
            end
            return f[key] = (orders, kgrid, partitions, datadict)
        end
        # Test the Fock self-energy
        if 1 in orders
            # The nondimensionalized Fock self-energy is the negative Lindhard function
            exact = -UEG_MC.lindhard.(kgrid / param.kF)
            # Check the MC result at k = 0 against the exact (non-dimensionalized)
            # Fock (exhange) self-energy: Σx(0) / E_{TF} = -F(0) = -1
            if orders == [1]
                means_fock, stdevs_fock = res.mean, res.stdev
            else
                means_fock, stdevs_fock = res.mean[1], res.stdev[1]
            end
            meas = measurement.(means_fock, stdevs_fock)
            scores = stdscore.(meas, exact)
            score_k0 = scores[1]
            worst_score = argmax(abs, scores)
            println(meas)
            # Summarize results
            println("""
                  Σₓ(k) ($solver):
                   • Exact value    (k = 0): $(exact[1])
                   • Measured value (k = 0): $(meas[1])
                   • Standard score (k = 0): $score_k0
                   • Worst standard score: $worst_score
                  """)
        end
    end
end

main()
