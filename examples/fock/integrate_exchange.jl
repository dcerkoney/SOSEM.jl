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
    if id.order[4] == 1
        # Bare Coulomb interaction (from EOM)
        # @debug "Bare V, T = $(id.extT)" maxlog = 5
        return CoulombBareinstant(qd, p)
    elseif id.order[2] == 0
        # Screened Coulomb interaction
        return Coulombinstant(qd, p)
    else
        # Counterterms for screened interaction
        invK = 1.0 / (qd^2 + mass2)
        return e0^2 / ϵ0 * invK * (mass2 * invK)^id.order[2]
    end
end

# """Evaluate an instantaneous bare Green's function."""
# function DiagTree.eval(id::BareGreenId, K, extT, varT, p::ParaMC)
#     @debug "Evaluating G: K = $K" maxlog = 3
#     β, me, μ, massratio = p.β, p.me, p.μ, p.massratio
#     ϵ = norm(K)^2 / (2me * massratio) - μ
#     # Overall sign difference relative to the Negle & Orland convention
#     return -Spectral.kernelFermiT(-1e-8, ϵ, β)
# end

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
    # Instantaneous bare interaction (interactionTauNum = 1) 
    # => innerLoopNum = innerTauNum = order
    return DiagParaF64(;
        type=SigmaDiag,
        hasTau=true,
        firstTauIdx=1,
        innerLoopNum=order,
        totalTauNum=order + 1,  # includes outgoing external time
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],
    )
end

"""Build variable pools for the exchange self-energy integration."""
function exchange_mc_variables(mcparam::UEG.ParaMC, n_kgrid::Int, alpha::Float64)
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Offset T pool by 3 for fixed external times (τout-, τout+, τin)
    T = Continuous(0.0, mcparam.β; offset=2, alpha=alpha)
    # Bin in external momentum
    ExtKidx = Discrete(1, n_kgrid; alpha=alpha)
    return (K, T, ExtKidx)
end

"""Measurement for a single diagram tree (without CTs, fixed order in V)."""
function measure(vars, obs, weights, config)
    # ExtK bin index
    ik = vars[2][1]
    obs[1][ik] += weights[1]
    return
end

"""Integrand for the exchange self-energy non-dimensionalized by ϵₖ."""
function integrand(vars, config)
    # We sample internal momentum/times, and external momentum index
    K, T, ExtKidx = vars
    R, Theta, Phi = K

    # Unpack userdata
    mcparam, exprtree, varK, kgrid = config.userdata

    # External momentum via random index into kgrid (wlog, we place it along the x-axis)
    ik = ExtKidx[1]
    varK[1, 1] = kgrid[ik]

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
    @assert (T.data[1] == 0) && (T.data[2] == 1e-6)

    @debug "K = $(varK)" maxlog = 3
    @debug "ik = $ik" maxlog = 3
    @debug "ExtK = $(kgrid[ik])" maxlog = 3

    # Evaluate the expression tree (additional = mcparam)
    ExprTree.evalKT!(exprtree, varK, T.data, mcparam)

    # Phase-space and Jacobian factor
    # NOTE: extra minus sign on self-energy definition!
    factor = 1.0 / (2π)^(mcparam.dim * innerLoopNum) * phifactor
    epsilon_k = norm(K)^2 / (2 * mcparam.me * mcparam.massratio)

    # Return the non-dimensionalized exchange integrand, Σx(k) / ϵₖ
    weight = exprtree.node.current
    root = exprtree.root[1]  # only one root
    return factor * weight[root] / epsilon_k
end

function build_sigma_x(order=1)
    # Generate the diagram and expression trees
    diagparam = exchange_param(order)

    # Momentum loop basis
    nk = diagparam.totalLoopNum
    k = DiagTree.getK(nk, 1)
    q = DiagTree.getK(nk, 2)

    g_param = DiagParaF64(type=GreenDiag, innerLoopNum=order-1, hasTau=true) 
    G = Parquet.green(g_param, k - q; name=:G)

    v_param = reconstruct(diagparam; type=Ver4Diag, innerLoopNum=0, firstLoopIdx=1)
    v_id = BareInteractionId(v_param, ChargeCharge, Instant, [0, 0, 0, 1]; k=q, permu=Di)
    V = DiagramF64(v_id; name=:V)

    sigma_id = SigmaID(diagparam, Instant; k=k)
    diagtree = DiagramF64(sigma_id, Prod(), [V, G]; name=:Σx)
    exprtree = ExprTree.build([diagtree])

    return diagparam, diagtree, exprtree
end

"""MC integration of the exchange self-energy"""
function main()
    # Debug mode
    if isinteractive()
        ENV["JULIA_DEBUG"] = Main
    end

    # UEG parameters for MC integration
    mcparam = ParaMC(; order=1, rs=1.0, beta=40.0, mass2=1.0, isDynamic=false)
    @debug "β * EF = $(mcparam.beta), β = $(mcparam.β), EF = $(mcparam.EF)"

    # Settings
    alpha = 2.0
    print = 0
    plot = true
    solver = :vegas

    # Number of evals below and above kF
    neval = 1e6

    # Inner loop order (first order ⟹ Fock)
    order = 1
    @assert order > 0

    # K-mesh for measurement
    minK = 0.2 * mcparam.kF
    Nk, korder = 4, 7
    kgrid =
        CompositeGrid.LogDensedGrid(
            :uniform,
            [0.0, 3 * mcparam.kF],
            [mcparam.kF],
            Nk,
            minK,
            korder,
        ).grid
    # Dimensionless k-grid
    k_kf_grid = kgrid / mcparam.kF
    # Grid size
    n_kgrid = length(kgrid)

    # Build diagram/expression trees for the exchange self-energy
    diagparam, sigma_x, exprtree = build_sigma_x(order)

    # Check the diagram tree
    print_tree(sigma_x)

    # NOTE: We assume there is only a single root in the ExpressionTree
    @assert length(exprtree.root) == 1

    # Temporary array for combined K-variables [ExtK, K].
    # We use the maximum necessary loop basis size for K pool.
    varK = zeros(3, diagparam.totalLoopNum)

    # Build adaptable MC integration variables
    (K, T, ExtKidx) = exchange_mc_variables(mcparam, n_kgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T), shape(ExtKidx)
    # We do not integrate the two external times, hence n_τ = totalTauNum - 2
    dof = [[diagparam.innerLoopNum, diagparam.totalTauNum - 2, 1]]

    # UEG SOSEM diagram observables are a function of |k| only (equal-time)
    obs = [zeros(n_kgrid)]

    # External times are fixed for left/right measurement of the discontinuity at τ = 0
    T.data[1] = 0      # τin  = 0
    T.data[2] = 1e-6   # τout = 0⁺

    res = integrate(
        integrand;
        solver=solver,
        measure=measure,
        neval=neval,
        print=print,
        # Config kwargs
        userdata=(mcparam, exprtree, varK, kgrid),
        var=(K, T, ExtKidx),
        dof=dof,
        obs=obs,
    )
    isnothing(res) && return

    # Save to JLD2 on main thread
    if !isnothing(res)
        means, stdevs = res.mean, res.stdev
        savename =
            "results/data/sigma_x_rs=$(mcparam.rs)_" *
            "beta_ef=$(mcparam.beta)_neval=$(neval)_$(solver)"
        jldopen("$savename.jld2", "a+") do f
            key = "$(short(mcparam))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (settings, mcparam, kgrid, res)
        end
        # Plot the result
        if plot
            fig, ax = plt.subplots()
            # Compare with exact non-dimensionalized Fock self-energy (ΣF(k) / ϵₖ)
            epsilon_k = kgrid^2 / (2 * mcparam.me * mcparam.massratio)
            ax.plot(
                k_kf_grid,
                fock_self_energy_exact.(kgrid, mcparam) / epsilon_k,
                "k";
                label="\$ \\sigma_{\\mathrm{F}}(k) / \\epsilon_k \$ (exact)",
            )
            sigma_x_label =
                order == 1 ? "\\sigma_{\\mathrm{F}}(k) / \\epsilon_k ($solver)" :
                "\\sigma^{($order)}_{\\mathrm{x}}(k) / \\epsilon_k ($solver)"
            ax.plot(k_kf_grid, means, "-"; color="C0", label=sigma_x_label)
            ax.fill_between(
                k_kf_grid,
                means - stdevs,
                means + stdevs;
                color="C0",
                alpha=0.4,
            )
            ax.legend(; loc="best")
            ax.set_xlabel("\$k / k_F\$")
            # ax.set_ylabel("\$\\Sigma_{x}(\\mathbf{k}) \\,/\\, \\epsilon_{\\mathbf{k}}\$")
            ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
            plt.tight_layout()
            fig.savefig(
                "results/fock/sigma_x_rs=$(mcparam.rs)_" *
                "beta_ef=$(mcparam.beta)_neval=$(neval)_$(solver).pdf",
            )
            plt.close("all")
        end
    end
end

main()
