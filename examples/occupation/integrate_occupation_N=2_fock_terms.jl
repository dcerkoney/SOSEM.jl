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

function integrate_occupation(
    mcparam::UEG.ParaMC,
    diagparam::DiagParaF64,
    exprtree::ExprTreeF64;
    kgrid=[0.0],
    alpha=3.0,
    neval=1e5,
    print=-1,
    solver=:vegasmc,
)
    # We assume that the expression tree has a single root
    @assert length(exprtree.root) == 1

    # List of expression tree roots, external times, and inner
    # loop numbers for each tree (to be passed to integrand)
    innerLoopNum = diagparam.innerLoopNum

    # Grid size
    n_kgrid = length(kgrid)

    # Temporary array for combined K-variables [ExtK, K].
    # We use the maximum necessary loop basis size for K pool.
    varK = zeros(3, diagparam.totalLoopNum)

    # Build adaptable MC integration variables
    (K, T, ExtKidx) = occupation_mc_variables(mcparam, n_kgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T), shape(ExtKidx)
    # We do not integrate the incoming external time and nₖ is instantaneous, hence n_τ = totalTauNum - 1
    dof = [[diagparam.innerLoopNum, diagparam.totalTauNum - 1, 1]]
    println("Integration DOF: $dof")

    # UEG SOSEM diagram observables are a function of |k| only (equal-time)
    obs = [zeros(n_kgrid)]

    # External times are fixed for left/right measurement of the discontinuity at τ = 0
    T.data[1] = 0  # τin = 0 (= τout⁺)

    # Phase-space factor
    phase_factor = 1.0 / (2π)^(mcparam.dim * innerLoopNum)

    # Total prefactor; extra minus sign relative to N&O convention for G₀
    prefactor = phase_factor

    return integrate(
        integrand;
        solver=solver,
        measure=measure,
        neval=neval,
        print=print,
        # MC config kwargs
        userdata=(mcparam, exprtree, innerLoopNum, prefactor, varK, kgrid),
        var=(K, T, ExtKidx),
        dof=dof,
        obs=obs,
    )
end

"""Build variable pools for the occupation."""
function occupation_mc_variables(mcparam::UEG.ParaMC, n_kgrid::Int, alpha::Float64)
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Offset T pool by 1 for fixed external times (instantaneous Green's function ⟹ τin = τout⁺)
    T = Continuous(0.0, mcparam.β; offset=1, alpha=alpha)
    # Bin in external momentum
    ExtKidx = Discrete(1, n_kgrid; alpha=alpha)
    return (K, T, ExtKidx)
end

"""Measurement for multiple diagram trees (counterterm partitions)."""
function measure(vars, obs, weights, config)
    ik = vars[3][1]  # ExtK bin index
    obs[1][ik] += weights[1]
    return
end

"""Integrand for the occupation number nₖ."""
function integrand(vars, config)
    # We sample internal momentum/times, and external momentum index
    K, T, ExtKidx = vars
    R, Theta, Phi = K

    # Unpack userdata
    mcparam, exprtree, innerLoopNum, prefactor, varK, kgrid = config.userdata

    # External momentum via random index into kgrid (wlog, we place it along the x-axis)
    ik = ExtKidx[1]
    varK[1, 1] = kgrid[ik]

    phifactor = 1.0
    for j in 1:innerLoopNum  # config.dof[i][1]
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

    # Evaluate the expression tree (additional = mcparam)
    ExprTree.evalKT!(exprtree, varK, T.data, mcparam)

    # Evaluate the occupation number integrand nₖ
    root = exprtree.root[1]
    weight = exprtree.node.current
    return phifactor * prefactor * weight[root]
end

"""
Constructs diagram parameters for the Fock self-energy Σ^λ_F(k).
Since the Fock self-energy is instantaneous, the tau labels do not matter.
"""
function fock_param(firstLoopIdx, totalLoopNum)
    # Instantaneous bare interaction (interactionTauNum = 1) 
    # => innerLoopNum = totalTauNum = 1
    return DiagParaF64(;
        type=SigmaDiag,
        hasTau=false,
        firstTauIdx=1,
        innerLoopNum=1,
        firstLoopIdx=firstLoopIdx,
        totalLoopNum=totalLoopNum,
        totalTauNum=1,
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],
    )
end

"""Builds a Fock self-energy diagram tree."""
function build_fock(extK; firstLoopIdx, totalLoopNum)
    return Parquet.sigma(fock_param(firstLoopIdx, totalLoopNum), extK; name=:Σx).diagram[1]
end

"""f''(ξₖ) (Σ^λ_F(k))^2 ((2,0,0) partition)"""
function build_term1(print=false, plot=false)
    n_loop = 2  # p[1]
    diagparam = DiagParaF64(;
        type=GreenDiag,
        innerLoopNum=n_loop,
        firstTauIdx=4,
        totalTauNum=n_loop + 1,
        hasTau=true,
    )
    g_param = DiagParaF64(;
        type=GreenDiag,
        innerLoopNum=0,
        firstTauIdx=4,
        totalLoopNum=diagparam.totalLoopNum,
        hasTau=true,
    )

    # Loop basis vector for external momentum
    k = DiagTree.getK(diagparam.totalLoopNum, 1)

    # Subdiagrams
    g1 = Parquet.green(g_param, k, (1, 2))
    g2 = Parquet.green(g_param, k, (2, 3))
    g3 = Parquet.green(g_param, k, (3, 1))
    sigmaF1 = build_fock(k; firstLoopIdx=2, totalLoopNum=3)
    sigmaF2 = build_fock(k; firstLoopIdx=3, totalLoopNum=3)

    # Build term 1
    extT = (1, 1)
    id = DiagTree.GreenId(diagparam, Instant; k=k, t=extT)
    diagtree = DiagramF64(id, Prod(), [g1, sigmaF1, g2, sigmaF2, g3])

    print && print_tree(diagtree)
    plot && plot_tree(diagtree)

    # Compile to expression tree
    exprtree = ExprTree.build([diagtree])
    print && println(exprtree)
    return diagparam, exprtree
end

"""-2 f''(ξₖ) Σ^λ_F(k) Σ^λ_F(k_F) ((1,1,0) partition without lambda derivative of Σ_F)"""
function build_term2(print=false, plot=false)
    n_loop = 1  # p[1]
    diagparam = DiagParaF64(;
        type=GreenDiag,
        innerLoopNum=n_loop,
        firstTauIdx=3,
        totalTauNum=n_loop + 1,
        hasTau=true,
    )
    g_param = DiagParaF64(;
        type=GreenDiag,
        innerLoopNum=0,
        totalLoopNum=diagparam.totalLoopNum,
        firstTauIdx=3,
        hasTau=true,
    )

    # Loop basis vector for external momentum
    k = DiagTree.getK(diagparam.totalLoopNum, 1)

    # Subdiagrams
    sigmaF = build_fock(k; firstLoopIdx=2, totalLoopNum=2)
    g1 = Parquet.green(g_param, k, (1, 2))
    g2 = Parquet.green(g_param, k, (2, 1))
    dμ_g2 = DiagTree.derivative([g2], BareGreenId, 1; index=1)[1]

    # Build term 2
    extT = (1, 1)
    id = DiagTree.GreenId(diagparam, Instant, [1, 0, 0, 0]; k=k, t=extT)
    diagtree = DiagramF64(id, Prod(), [g1, sigmaF, dμ_g2]; factor=2.0)

    print && print_tree(diagtree)
    plot && plot_tree(diagtree)

    # Compile to expression tree
    exprtree = ExprTree.build([diagtree])
    print && println(exprtree)
    return diagparam, exprtree
end

"""f''(ξₖ) (Σ^λ_F(k_F))^2 ((0,2,0) partition). Note that there is nothing to integrate here."""
function build_term3(print=false, plot=false)
    n_loop = 0  # p[1]
    diagparam = DiagParaF64(;
        type=GreenDiag,
        innerLoopNum=n_loop,
        firstTauIdx=2,
        totalTauNum=n_loop + 1,
        hasTau=true,
    )

    # Loop basis vector for external momentum
    k = DiagTree.getK(diagparam.totalLoopNum, 1)

    # Build term 3
    g = Parquet.green(diagparam, k, (1, 1))
    diagtree = DiagTree.derivative([g], BareGreenId, 2; index=1)[1]

    print && print_tree(diagtree)
    plot && plot_tree(diagtree)

    # Compile to expression tree
    exprtree = ExprTree.build([diagtree])
    print && println(exprtree)
    return diagparam, exprtree
end

"""MC integration of the occupation number."""
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

    # UEG parameters for MC integration
    param = ParaMC(; order=2, rs=1.0, beta=40.0, mass2=1.0, isDynamic=false)
    @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)"

    # Dimensionless k-mesh for measurement
    k_kf_grid = CompositeGrid.LogDensedGrid(:cheb, [0.0, 3.0], [1.0], 5, 0.02, 5)

    # Dimensionful k-grid
    kgrid = k_kf_grid * param.kF

    # # K-mesh for measurement
    # minK = 0.05 * param.kF
    # Nk, korder = 4, 5
    # kleft =
    #     CompositeGrid.LogDensedGrid(
    #         :uniform,
    #         [0.75 * param.kF, param.kF - 1e-8],
    #         [param.kF - 1e-8],
    #         Nk,
    #         minK,
    #         korder,
    #     ).grid
    # kright =
    #     CompositeGrid.LogDensedGrid(
    #         :uniform,
    #         [param.kF + 1e-8, 1.25 * param.kF],
    #         [param.kF + 1e-8],
    #         Nk,
    #         minK,
    #         korder,
    #     ).grid
    # kgrid = [kleft; kright]

    # Dimensionless k-grid
    # k_kf_grid = kgrid / param.kF

    # Settings
    alpha = 3.0
    print = 0
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 1e8

    # Integrate each term in the N=2 Fock series benchmark
    second_order_fock_terms = [build_term1(), build_term2(), build_term3()]
    for (i, (diagparam, exprtree)) in enumerate(second_order_fock_terms)
        println("Integrating term $i...")
        res = integrate_occupation(
            param,
            diagparam,
            exprtree;
            kgrid=kgrid,
            alpha=alpha,
            neval=neval,
            print=print,
            solver=solver,
        )
        println("Done!")
        # Save to JLD2 on main thread
        if !isnothing(res)
            savename = "results/data/occupation_N=2_fock_term_$(i)_neval=$(neval)"
            jldopen("$savename.jld2", "a+"; compress=true) do f
                key = "$(UEG.short(param))"
                if haskey(f, key)
                    @warn("replacing existing data for $key")
                    delete!(f, key)
                end
                return f[key] = (param, kgrid, res)
            end
        end
    end
end

main()
