using AbstractTrees
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
function CoulombBareinstant(q, p::ParaMC)
    return KOinstant(q, p.e0, p.dim, 0.0, 0.0, p.kF)
end

"""Evaluate a bare Coulomb interaction line."""
function DiagTree.eval(id::BareInteractionId, K, extT, varT, p::ParaMC)
    @debug "Evaluating V: K = $K, |K| = $(norm(K))" maxlog = 3
    return CoulombBareinstant(norm(K), p)
end

"""Evaluate an instantaneous bare Green's function."""
function DiagTree.eval(id::BareGreenId, K, extT, varT, p::ParaMC)
    @debug "Evaluating G: K = $K" maxlog = 3
    β, me, μ, massratio = p.β, p.me, p.μ, p.massratio
    ϵ = norm(K)^2 / (2me * massratio) - μ
    # Overall sign difference relative to the Negle & Orland convention
    return -Spectral.kernelFermiT(-1e-8, ϵ, β)
end

"""
Exact expression for the Fock (exchange) self-energy
in terms of the dimensionless Lindhard function.
"""
function fock_self_energy_exact(k, p::ParaMC)
    # The (dimensionful) value at k = 0 is minus the Thomas-Fermi energy
    eTF = p.qTF^2 / (2 * p.me)
    return -eTF * lindhard(k / p.kF)
end

"""Constructs diagram parameters for the bare Fock self-energy."""
function fock_param()
    # Instantaneous bare interaction (interactionTauNum = 1) 
    # => innerLoopNum = totalTauNum = 1
    return DiagParaF64(;
        type=SigmaDiag,
        hasTau=false,
        firstTauIdx=1,
        innerLoopNum=1,
        totalTauNum=1,
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],
    )
end

"""Build variable pools for the Fock MC integration."""
function fock_mc_variables(n_kgrid::Int, alpha::Float64)
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Bin in external momentum
    ExtKidx = Discrete(1, n_kgrid; alpha=alpha)
    return (K, ExtKidx)
end

"""Measurement for a single diagram tree (without CTs, fixed order in V)."""
function measure(vars, obs, weights, config)
    # ExtK bin index
    ik = vars[2][1]
    obs[1][ik] += weights[1]
    return
end

"""Integrand for the Fock self-energy non-dimensionalized by E²_{TF} ~ q⁴_{TF}."""
function integrand(vars, config)
    # Sample internal momentum (the Fock diagram is instantaneous)
    K, ExtKidx = vars
    R, Theta, Phi = K

    # Unpack userdata
    mcparam, exprtree, varK, kgrid = config.userdata

    # External momentum via random index into kgrid (wlog, we place it along the x-axis)
    ik = ExtKidx[1]
    varK[1, 1] = kgrid[ik]

    phifactor = 1.0
    innerLoopNum = config.dof[1][1]
    @assert innerLoopNum == 1
    for i in 1:innerLoopNum
        r = R[i] / (1 - R[i])
        θ = Theta[i]
        ϕ = Phi[i]
        varK[1, i + 1] = r * sin(θ) * cos(ϕ)
        varK[2, i + 1] = r * sin(θ) * sin(ϕ)
        varK[3, i + 1] = r * cos(θ)
        phifactor *= r^2 * sin(θ) / (1 - R[i])^2
    end

    @debug "K = $(varK)" maxlog = 3
    @debug "ik = $ik" maxlog = 3
    @debug "ExtK = $(kgrid[ik])" maxlog = 3

    # Evaluate the expression tree (additional = mcparam)
    weight = exprtree.node.current
    ExprTree.evalKT!(exprtree, varK, [], mcparam)
    r = exprtree.root[1]  # only one root

    # Phase-space and Jacobian factor
    # NOTE: extra minus sign on self-energy definition!
    factor = 1.0 / (2π)^(mcparam.dim * innerLoopNum) * phifactor

    # Return the non-dimensionalized Fock integrand
    eTF = mcparam.qTF^2 / (2 * mcparam.me)  # Thomas-Fermi energy
    return factor * weight[r] / eTF
end

"""Test MC integration on the Fock self-energy"""
function main()
    # Debug mode
    if isinteractive()
        ENV["JULIA_DEBUG"] = Main
    end

    # UEG parameters for MC integration
    param = ParaMC(; order=1, rs=2.0, beta=200.0, mass2=0.0, isDynamic=false)
    @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)"

    # K-mesh for measurement
    # k_kf_grid = [0.0]
    k_kf_grid = np.load("results/kgrids/kgrid_vegas_dimless_n=77_small.npy")
    kgrid = param.kF * k_kf_grid
    @assert 0 in kgrid

    # Settings
    alpha = 2.0
    print = 0
    plot = true
    solver = :vegas

    # Number of evals below and above kF
    neval = 1e8

    # Generate the diagram and expression trees
    diagparam = fock_param()
    fock_diagram = Parquet.sigma(diagparam; name=:Σx).diagram[1]
    exprtree = ExprTree.build([fock_diagram])

    # Check the diagram tree
    print_tree(fock_diagram)

    # NOTE: We assume there is only a single root in the ExpressionTree
    @assert length(exprtree.root) == 1

    # Grid size
    n_kgrid = length(kgrid)

    # Temporary array for combined K-variables [ExtK, K].
    # We use the maximum necessary loop basis size for K pool.
    varK = zeros(3, diagparam.totalLoopNum)

    # Build adaptable MC integration variables
    var = fock_mc_variables(n_kgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(ExtKidx)
    dof = [[diagparam.innerLoopNum, 1]]

    # UEG SOSEM diagram observables are a function of |k| only (equal-time)
    obs = [zeros(n_kgrid)]

    res = integrate(
        integrand;
        solver=solver,
        measure=measure,
        neval=neval,
        print=print,
        # Config kwargs
        userdata=(param, exprtree, varK, kgrid),
        var=var,
        dof=dof,
        obs=obs,
    )
    if isnothing(res)
        return
    end

    # The nondimensionalized Fock self-energy is the negative Lindhard function
    exact = -lindhard.(kgrid / param.kF)

    # Check the MC result at k = 0 against the exact (non-dimensionalized)
    # Fock (exhange) self-energy: Σx(0) / E²_{TF} = -F(0) = -1
    means, stdevs = res.mean, res.stdev
    meas = measurement.(means, stdevs)
    scores = stdscore.(meas, exact)
    score_k0 = scores[1]
    worst_score = argmax(abs, scores)

    # Summarize results
    print("""
          Σₓ(k) ($solver):
           • Exact value    (k = 0): $(exact[1])
           • Measured value (k = 0): $(meas[1])
           • Standard score (k = 0): $score_k0
           • Worst standard score: $worst_score
          """)

    # Save to JLD2 on main thread
    if !isnothing(res)
        savename =
            "results/data/sigma_fock_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_neval=$(neval)_$(solver)"
        jldopen("$savename.jld2", "a+") do f
            key = "$(short(param))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (settings, param, kgrid, res)
        end
    end

    # Plot the result
    if plot
        fig, ax = plt.subplots()
        # Compare with exact non-dimensionalized function (-F(k / kF))
        ax.plot(k_kf_grid, -lindhard.(k_kf_grid), "k"; label="(exact)")
        ax.plot(k_kf_grid, means, "-"; color="C0", label="($solver)")
        ax.fill_between(k_kf_grid, means - stdevs, means + stdevs; color="C0", alpha=0.4)
        ax.legend(; loc="best")
        ax.set_xlabel("\$k / k_F\$")
        ax.set_ylabel("\$\\Sigma_{F}(\\mathbf{k}) \\,/\\, E^{2}_{\\mathrm{TF}}\$")
        ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
        plt.tight_layout()
        fig.savefig(
            "results/fock/sigma_fock_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_neval=$(neval)_$(solver).pdf",
        )
        plt.close("all")
    end
end

main()
