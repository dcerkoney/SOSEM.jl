using AbstractTrees
using CodecZlib
using CompositeGrids
using ElectronGas
using ElectronLiquid
using ElectronLiquid.UEG: ParaMC, KOinstant
using FeynmanDiagram
using JLD2
using Measurements
using MCIntegration
using Lehmann
using LinearAlgebra
using PyCall
using SOSEM.UEG_MC: lindhard

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

"""Constructs diagram parameters for the polarization."""
function polarization_param(order=0)
    # Instantaneous bare interaction (interactionTauNum = 1) 
    # => innerLoopNum = order, totalTauNum = order + 2
    return DiagParaF64(;
        type=PolarDiag,
        hasTau=true,
        firstTauIdx=1,
        innerLoopNum=order,
        totalTauNum=order + 2,
        filter=[Proper, NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],
    )
end

function build_poln_with_ct(orders=[0])
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
        n_lowest=0,
        renorm_mu=true,
        renorm_lambda=true,
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
    # Get diagram parameters
    diagparam = polarization_param(n_loop)
    # Build diagram/expression trees for the polarization
    diagtree = Parquet.polarization(diagparam)
    # Check the diagram tree
    @debug "\nDiagTree:\n" * repr_tree(diagtree)
    return diagparam, diagtree
end

"""Build variable pools for the exchange self-energy integration."""
function polarization_mc_variables(
    mcparam::UEG.ParaMC,
    n_kgrid::Int,
    n_Tgrid::Int,
    alpha::Float64,
)
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Offset T pool by 2 for fixed external times (τin, τout)
    T = Continuous(0.0, mcparam.β; offset=2, alpha=alpha)
    # Bin in external momentum
    ExtKidx = Discrete(1, n_kgrid; alpha=alpha)
    # Bin in outgoing external imaginary time
    ExtTidx = Discrete(1, n_Tgrid; alpha=alpha)
    return (K, T, ExtKidx, ExtTidx)
end

"""Measurement for a single diagram tree (without CTs, fixed order in V)."""
function measure(vars, obs, weights, config)
    ik = vars[3][1]  # ExtK bin index
    iT = vars[4][1]  # ExtT bin index
    # Measure the weight of each partition
    for o in 1:(config.N)
        obs[o][ik] += weights[o]
    end
    return
end

function integrate_poln_with_ct(
    mcparam::UEG.ParaMC,
    diagparams::Vector{DiagParaF64},
    exprtrees::Vector{ExprTreeF64};
    kgrid=[0.0],
    Tgrid,
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

    # Temporary array for combined K-variables [ExtK, K].
    # We use the maximum necessary loop basis size for K pool.
    maxloops = maximum(p.totalLoopNum for p in diagparams)
    varK = zeros(3, maxloops)

    # Grid sizes
    n_kgrid = length(kgrid)
    n_Tgrid = length(Tgrid)

    # Build adaptable MC integration variables
    (K, T, ExtKidx, ExtTidx) = polarization_mc_variables(mcparam, n_kgrid, n_Tgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T), shape(ExtKidx)
    # The two external times are fixed & discretized, respectively, hence
    # n_τ = totalTauNum - 2 (number of continuous times).
    dof = [[p.innerLoopNum, p.totalTauNum - 2, 1, 1] for p in diagparams]

    # Π(q, τ) is a function of q and τ for each partition
    obs = repeat([zeros(length(n_kgrid), length(n_Tgrid))], length(dof))

    # The incoming external time is fixed at the origin (COM coordinates)
    T.data[1] = 0

    # We non-dimensionalize the result via division by the Thomas-Fermi energy
    eTF = mcparam.qTF^2 / (2 * mcparam.me)

    # Phase-space factors
    phase_factors = [1.0 / (2π)^(mcparam.dim * nl) for nl in innerLoopNums]

    # Total prefactors
    prefactors = phase_factors / eTF

    return integrate(
        integrand;
        solver=solver,
        measure=measure,
        neval=neval,
        print=print,
        # MC config kwargs
        userdata=(mcparam, exprtrees, innerLoopNums, prefactors, varK, kgrid, Tgrid),
        var=(K, T, ExtKidx, ExtTidx),
        dof=dof,
        obs=obs,
    )
end

"""Integrand for the polarization."""
function integrand(vars, config)
    # We sample internal momentum/times, and external momentum index
    K, T, ExtKidx, ExtTidx = vars
    R, Theta, Phi = K

    # Unpack userdata
    mcparam, exprtrees, innerLoopNums, prefactors, varK, kgrid, Tgrid = config.userdata

    # Evaluate the integrand for each partition
    integrand = Vector(undef, config.N)
    for i in 1:(config.N)
        # External momentum via random index into kgrid (wlog, we place it along the x-axis)
        ik = ExtKidx[1]
        varK[1, 1] = kgrid[ik]

        # Outgoing external time via random index into τ-grid
        iT = ExtTidx[1]
        T.data[2] = Tgrid[iT]

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
        # @assert T.data[1] == 0

        @debug "K = $(varK)" maxlog = 3
        @debug "ik = $ik" maxlog = 3
        @debug "ExtK = $(kgrid[ik])" maxlog = 3

        @debug "T = $(T.data)" maxlog = 3
        @debug "iT = $iT" maxlog = 3
        @debug "τin = $(Tgrid[1])" maxlog = 3
        @debug "τout = $(Tgrid[iT])" maxlog = 3

        # Evaluate the expression tree (additional = mcparam)
        ExprTree.evalKT!(exprtrees[i], varK, T.data, mcparam; eval=UEG_MC.Propagators.eval)

        # Evaluate the exchange integrand Σx(k) for this partition
        root = exprtrees[i].root[1]  # there is only one root per partition
        weight = exprtrees[i].node.current
        integrand[i] = phifactor * prefactors[i] * weight[root]
    end
    return integrand
end

"""MC integration of the charge polarization."""
function main()
    # Debug mode
    if isinteractive()
        ENV["JULIA_DEBUG"] = Main
    end

    # UEG parameters for MC integration
    mcparam = ParaMC(; order=1, rs=1.0, beta=40.0, mass2=1.0, isDynamic=false)
    @debug "β * EF = $(mcparam.beta), β = $(mcparam.β), EF = $(mcparam.EF)"

    # Settings
    alpha = 3.0
    print = 0
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 1e6

    # Inner loop order (skipping zeroth order, where the exact function is available)
    order = 0
    @assert order ≥ 0

    # K-mesh for measurement
    minK = 0.2 * mcparam.kF
    Nk, korder = 4, 4
    kgrid =
        CompositeGrid.LogDensedGrid(
            :uniform,
            [0.0, 3 * mcparam.kF],
            [0.0],
            # [mcparam.kF],
            Nk,
            minK,
            korder,
        ).grid

    # Dimensionless k-grid
    k_kf_grid = kgrid / mcparam.kF
    n_kgrid = length(kgrid)

    # We measure Π on a compact DLR grid, and will later
    # upsample to uniform grid with N_τ ≈ Nw_unif for FFTs.
    Euv = 1.0        # ultraviolet energy cutoff of the Green's function
    rtol = 1e-8      # accuracy of the representation
    isFermi = false  # Π is bosonic
    symmetry = :ph   # Π is particle-hole symmetric
    dlr = DLRGrid(Euv, mcparam.β, rtol, isFermi, symmetry)

    # τ-grid from DLR
    Tgrid = dlr.τ
    n_Tgrid = length(Tgrid)

    @assert n_kgrid * n_Tgrid ≤ 150 "Requested number of grid points is too high (N = $(n_kgrid * n_Tgrid))!"

    # Lowest-order result from exact expressions
    c1ls_vs_Kcut = []
    Kcuts = mcparam.kF * [3.0, 15.0, 50.0, 100.0, 500.0]
    for Kcut in Kcuts
        mcparam = ParaMC(; order=1, rs=2.0, beta=200.0, mass2=1.0, isDynamic=false)
        interp_kind = "cubic"
        Ncut = 10000              # Upper cutoff for Matsubara summation
        # Kcut = 10 * mcparam.kF    # Upper cutoff for the momentum integration
        minK = 0.20 * mcparam.kF  # Minimal spacing for the CompositeGrid
        # Build kgrid
        Nk, korder = 10, 10
        kgrid =
            CompositeGrid.LogDensedGrid(
                :uniform,
                [0.0, Kcut],
                [0.0, mcparam.kF],
                Nk,
                minK,
                korder,
            ).grid
        n_kgrid = length(kgrid)  # ~175 k-points
        ngrid = collect(1:Ncut)  # dense uniform Matsubara grid
        sum_q = [
            (
                2 * sum(Polarization.Polarization0_ZeroTemp(q, ngrid, mcparam)) +
                Polarization.Polarization0_ZeroTemp(q, 0, mcparam)  # static contribution
            ) / mcparam.β for q in kgrid
        ]
        @assert length(sum_q) == n_kgrid
        sum_q_interp = interp.interp1d(kgrid, sum_q; kind=interp_kind)
        push!(c1ls_vs_Kcut, measurement(integ.quad(sum_q_interp, 0, Kcut)...))
    end

    c1ls_vs_Ncut = []
    Ncuts = [1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]
    for Ncut in Ncuts
        mcparam = ParaMC(; order=1, rs=2.0, beta=200.0, mass2=1.0, isDynamic=false)
        interp_kind = "cubic"
        # Ncut = 10000              # Upper cutoff for Matsubara summation
        Kcut = 10 * mcparam.kF    # Upper cutoff for the momentum integration
        minK = 0.20 * mcparam.kF  # Minimal spacing for the CompositeGrid
        # Build kgrid
        Nk, korder = 10, 10
        kgrid =
            CompositeGrid.LogDensedGrid(
                :uniform,
                [0.0, Kcut],
                [0.0, mcparam.kF],
                Nk,
                minK,
                korder,
            ).grid
        n_kgrid = length(kgrid)  # ~175 k-points
        ngrid = collect(1:Ncut)  # dense uniform Matsubara grid
        sum_q = [
            (
                2 * sum(Polarization.Polarization0_ZeroTemp(q, ngrid, mcparam)) +
                Polarization.Polarization0_ZeroTemp(q, 0, mcparam)  # static contribution
            ) / mcparam.β for q in kgrid
        ]
        @assert length(sum_q) == n_kgrid
        sum_q_interp = interp.interp1d(kgrid, sum_q; kind=interp_kind)
        push!(c1ls_vs_Ncut, measurement(integ.quad(sum_q_interp, 0, Kcut)...))
    end

    # ... 

    # Build diagram/expression trees for the polarization to order
    # ξᴺ in the renormalized perturbation theory (includes CTs in μ and λ)
    partitions, diagparams, diagtrees, exprtrees = build_poln_with_ct(orders)

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

    res = integrate_poln_with_ct(
        param,
        diagparams,
        exprtrees;
        kgrid=kgrid,
        Tgrid=Tgrid,
        alpha=alpha,
        neval=neval,
        print=print,
        solver=solver,
    )

    # TODO: Post-process result, upsample to Nτ = Nω ~ 1000, FT, and use to obtain the local moment
    #   Qn: Can we use DLR for the FTs using the dense uniform grid instead of DLR grid, i.e.,
    #
    Nw_unif = 1000
    coeff = DLR.tau2dlr(dlr, ...)  # from unif grid Nω ~ 1000
    pi_kw = DLR.dlr2matfreq(dlr, coeff; ngrid=Nw_unif)

    # Save to JLD2 on main thread
    if !isnothing(res)
        savename =
            "results/data/poln_kt_rs=$(mcparam.rs)_" *
            "beta_ef=$(mcparam.beta)_neval=$(neval)_$(solver)"
        jldopen("$savename.jld2", "a+"; compress=true) do f
            key = "$(short(mcparam))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (settings, mcparam, kgrid, res)
        end
    end
end

main()
