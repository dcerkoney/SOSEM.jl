using AbstractTrees
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

"""Bare Coulomb interaction."""
@inline function CoulombBareinstant(q, p::ParaMC)
    return KOinstant(q, p.e0, p.dim, 0.0, 0.0, p.kF)
end

"""Evaluate a statically screened Coulomb interaction line."""
function eval(id::BareInteractionId, K, _, varT, p::ParaMC)
    # TODO: Implement check for bare interaction using: is_bare = (order[end] = 1)
    e0, ϵ0, mass2 = p.e0, p.ϵ0, p.mass2
    qd = sqrt(dot(K, K))
    if id.order[2] == 0
        # Screened Coulomb interaction
        return Coulombinstant(qd, p)
    else
        # Counterterms for screened interaction
        invK = 1.0 / (qd^2 + mass2)
        return e0^2 / ϵ0 * invK * (mass2 * invK)^id.order[2]
    end
end

"""Evaluate a bare Green's function line."""
function eval(id::BareGreenId, K, _, varT, p::ParaMC)
    β, me, μ, massratio = p.β, p.me, p.μ, p.massratio
    # External time difference
    τin, τout = varT[id.extT[1]], varT[id.extT[2]]
    τ = τout - τin
    # Get energy
    ϵ = norm(K)^2 / (2me * massratio) - μ
    # Normal-ordering for the equal-time case
    if τ ≈ 0
        return -Spectral.kernelFermiT(-1e-8, ϵ, β)
    end
    return -Spectral.kernelFermiT(τ, ϵ, β)
end

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
    obs[1][ik, iT] += weights[1]
    return
end

"""Integrand for the exchange self-energy non-dimensionalized by ϵₖ."""
function integrand(vars, config)
    # We sample internal momentum/times, and external momentum index
    K, T, ExtKidx, ExtTidx = vars
    R, Theta, Phi = K

    # Unpack userdata
    mcparam, exprtree, varK, kgrid, Tgrid = config.userdata

    # External momentum via random index into kgrid (wlog, we place it along the x-axis)
    ik = ExtKidx[1]
    varK[1, 1] = kgrid[ik]

    # Outgoing external time via random index into τ-grid
    iT = ExtTidx[1]
    T.data[2] = Tgrid[iT]

    phifactor = 1.0
    innerLoopNum = config.dof[1][1]
    for i in 1:innerLoopNum
        r = R[i] / (1 - R[i])
        θ = Theta[i]
        ϕ = Phi[i]
        varK[1, i + 1] = r * sin(θ) * cos(ϕ)
        varK[2, i + 1] = r * sin(θ) * sin(ϕ)
        varK[3, i + 1] = r * cos(θ)
        phifactor *= r^2 * sin(θ) / (1 - R[i])^2
    end
    # @assert (T.data[1] == 0) && (T.data[2] == 1e-6)

    @debug "K = $(varK)" maxlog = 3
    @debug "ik = $ik" maxlog = 3
    @debug "ExtK = $(kgrid[ik])" maxlog = 3

    @debug "T = $(T.data)" maxlog = 3
    @debug "iT = $iT" maxlog = 3
    @debug "τin = $(Tgrid[1])" maxlog = 3
    @debug "τout = $(Tgrid[iT])" maxlog = 3

    # Evaluate the expression tree (additional = mcparam)
    ExprTree.evalKT!(exprtree, varK, T.data, mcparam)

    # Phase-space and Jacobian factor
    # NOTE: extra minus sign on self-energy definition!
    factor = 1.0 / (2π)^(mcparam.dim * innerLoopNum) * phifactor

    # Return the non-dimensionalized exchange integrand, Σx(k) / ϵₖ
    weight = exprtree.node.current
    root = exprtree.root[1]  # only one root
    return factor * weight[root]
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
    order = 1
    @assert order > 0

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
    # upsample to uniform grid with N_τ ≈ 1000 for FFTs.
    Euv = 1.0        # ultraviolet energy cutoff of the Green's function
    rtol = 1e-8      # accuracy of the representation
    isFermi = false  # Π is bosonic
    symmetry = :ph   # Π is particle-hole symmetric
    dlr = DLRGrid(Euv, mcparam.β, rtol, isFermi, symmetry)

    # τ-grid from DLR
    Tgrid = dlr.τ
    n_Tgrid = length(Tgrid)

    @assert n_kgrid * n_Tgrid ≤ 150 "Requested number of grid points is too high (N = $(n_kgrid * n_Tgrid))!"

    # Get diagram parameters
    diagparam = polarization_param(order)

    # Build diagram/expression trees for the polarization
    diagtree = Parquet.polarization(diagparam)
    exprtree = ExprTree.build([diagtree])

    # Check the diagram tree
    print_tree(diagtree)

    # NOTE: We assume there is only a single root in the ExpressionTree
    @assert length(exprtree.root) == 1

    # Temporary array for combined K-variables [ExtK, K].
    # We use the maximum necessary loop basis size for K pool.
    varK = zeros(3, diagparam.totalLoopNum)

    # Build adaptable MC integration variables
    (K, T, ExtKidx, ExtTidx) = polarization_mc_variables(mcparam, n_kgrid, n_Tgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T), shape(ExtKidx)
    # The two external times are fixed & discretized, respectively, hence
    # n_τ = totalTauNum - 2 (number of continuous times).
    dof = [[diagparam.innerLoopNum, diagparam.totalTauNum - 2, 1, 1]]

    # Π(q, τ) is a function of q and τ
    obs = [zeros(length(n_kgrid), length(n_Tgrid))]

    # The incoming external time is fixed at the origin (COM coordinates)
    T.data[1] = 0

    res = integrate(
        integrand;
        solver=solver,
        measure=measure,
        neval=neval,
        print=print,
        # Config kwargs
        userdata=(mcparam, exprtree, varK, kgrid, Tgrid),
        var=(K, T, ExtKidx, ExtTidx),
        dof=dof,
        obs=obs,
    )
    isnothing(res) && return

    # Save to JLD2 on main thread
    if !isnothing(res)
        savename =
            "results/data/poln_kt_rs=$(mcparam.rs)_" *
            "beta_ef=$(mcparam.beta)_neval=$(neval)_$(solver)"
        jldopen("$savename.jld2", "a+") do f
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
