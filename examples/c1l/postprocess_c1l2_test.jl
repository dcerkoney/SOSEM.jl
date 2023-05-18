using AbstractTrees
using CodecZlib
using CompositeGrids
using ElectronGas
using ElectronLiquid
using ElectronLiquid.UEG
using FeynmanDiagram
using Interpolations
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
@pyimport scipy.integrate as integ
@pyimport scipy.interpolate as interp

function c1l2_over_eTF2_vlambda_vlambda(l)
    m = sqrt(l)
    I1 = (l / (l + 4) - log(l / (l + 4)) - 1) / 4
    I2 = (l^2 / (l + 4) - (l + 4) - 2l * log(l / (l + 4))) / 64
    I3 = 2(2 / (l + 4) - atan(2 / m) / m) / 3
    return (I1 + I2 + I3)
end

function c1l2_over_eTF2_v_vlambda(l)
    m = sqrt(l)
    return (π / 3m - 1 / 12) + (l / 12 + 1) * log((4 + l) / l) / 4 - (2 / 3m) * atan(2 / m)
end

"""Returns the static structure factor S₀(q) of the UEG in the HF approximation."""
function static_structure_factor_hf_exact(q, param::ParaMC)
    x = q / param.kF
    if x < 2
        return 3x / 4.0 - x^3 / 16.0
    end
    return 1.0
end

"""Π₀(q, τ=0) = χ₀(q, τ=0) = -n₀ S₀(q)"""
function bare_polarization_exact_t0(q, param::ParaMC)
    n0 = param.kF^3 / 3π^2
    return -n0 * static_structure_factor_hf_exact(q, param)
end

# Post-process integrations for c1l given binned P(q, tau)
function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    rs = 1.0
    beta = 40.0
    mass2 = 1.0
    solver = :vegasmc

    # Number of evals
    neval = 1e10

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order = 1
    max_order = 3
    min_order_plot = 1
    max_order_plot = 3

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # UEG parameters for MC integration
    loadparam = ParaMC(;
        order=max_order,
        rs=rs,
        beta=beta,
        mass2=mass2,
        isDynamic=false,
        isFock=false,
    )

    # Load the raw data
    savename =
        "results/data/static_structure_factor_n=$(max_order)_rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(mass2)_neval=$(neval)_$(solver)$(ct_string)"
    # TODO: Rerun with new format,
    #   orders, param, kgrid, tgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
    orders, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(loadparam))"
        return f[key]
    end

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    n0 = param.kF^3 / 3π^2

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    for (k, v) in data
        data[k] = v / (factorial(k[2]) * factorial(k[3]))
    end

    println(typeof(data))
    for P in keys(data)
        # Extra overall sign due to N&O definition of Π
        data[P] *= -1
    end

    merged_data = UEG_MC.mergeInteraction(data)
    println(typeof(merged_data))

    # Get exact instantaneous bare polarization Π₀(q, τ=0)
    pi0_t0 = bare_polarization_exact_t0.(kgrid, [param])

    # Set bare result manually using exact function
    # if haskey(merged_data, (1, 0)) == false
    if min_order > 1
        # treat quadrature data as numerically exact
        merged_data[(1, 0)] = measurement.(-pi0_t0 / n0, 0.0)
    elseif min_order == 1
        stdscores = stdscore.(merged_data[(1, 0)], -pi0_t0 / n0)
        worst_score = argmax(abs, stdscores)
        println("Worst standard score for N=1 (measured): $worst_score")
        # @assert worst_score ≤ 10
    end

    # Reexpand merged data in powers of μ
    ct_filename = "examples/counterterms/data_Z$(ct_string).jld2"
    z, μ = UEG_MC.load_z_mu(param; ct_filename=ct_filename)
    # Add Taylor factors to CT data
    for (p, v) in z
        z[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    for (p, v) in μ
        μ[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    δz, δμ = CounterTerm.sigmaCT(max_order, μ, z; isfock=false, verbose=1)

    println("Computed δμ: ", δμ)
    static_structure = UEG_MC.chemicalpotential_renormalization_poln(
        merged_data,
        δμ;
        min_order=1,
        max_order=max_order,
    )
    δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
    static_structure_2_manual = merged_data[(2, 0)] + δμ1_exact * merged_data[(1, 1)]
    scores_2 = stdscore.(static_structure[2] - static_structure_2_manual, 0.0)
    worst_score_2 = argmax(abs, scores_2)
    println("2nd order renorm vs manual worst score: $worst_score_2")

    println(UEG.paraid(param))
    println(partitions)
    println(typeof(static_structure))

    # Aggregate the full results for Σₓ up to order N
    static_structure_total = UEG_MC.aggregate_orders(static_structure)
    println(kgrid)
    println(static_structure_total[1])

    static_structure_hf_means = Measurements.value.(static_structure_total[1])
    static_structure_hf_interp = interp.interp1d(kgrid, static_structure_hf_means)

    # Thomas-Fermi energy squared
    eTF2 = param.qTF^4 / (2 * param.me)^2

    exact = c1l2_over_eTF2_v_vlambda(param.mass2 / param.kF^2)
    println("\n Exact bare c1l over eTF^2 at T = 0:\n$exact")

    # Integrand for C⁽¹⁾ˡ₂ (measured S₀(q))
    k_lower, k_upper = minimum(kgrid), maximum(kgrid)
    function integrand(q)
        return 8 * n0 * param.e0^4 * static_structure_hf_interp(q) / (q^2 + param.mass2)
    end
    # Integrand for C⁽¹⁾ˡ₂ (exact S₀(q))
    function integrand_v2(q)
        return 8 * n0 * param.e0^4 * static_structure_factor_hf_exact(q, param) /
               (q^2 + param.mass2)
    end

    res_v1 = integ.quad(integrand, k_lower, k_upper)
    res_v2 = integ.quad(integrand_v2, k_lower, k_upper)
    # res_v3 = integ.quad(integrand_v2, k_lower, 10 * param.kF)
    res_v3 = integ.quad(integrand_v2, k_lower, 10 * k_upper)
    res_v4 = integ.quad(integrand_v2, 0.0, Inf)

    meas_v1 = measurement(res_v1...) / eTF2
    meas_v2 = measurement(res_v2...) / eTF2
    meas_v3 = measurement(res_v3...) / eTF2
    meas_v4 = measurement(res_v4...) / eTF2

    relerr1 = 100 * abs(exact - meas_v1.val) / exact
    relerr2 = 100 * abs(exact - meas_v2.val) / exact
    relerr3 = 100 * abs(exact - meas_v3.val) / exact
    relerr4 = 100 * abs(exact - meas_v4.val) / exact
    # z3 = stdscore(meas_v3, exact)

    println("\n c1l over eTF^2 via integration of linearly interpolated data on [0, 3kF]:")
    println("$res_v1\n$meas_v1 ($(round(relerr1, digits=2))% relative error)")

    println("\n c1l over eTF^2 via integration of exact bare integrand on [0, 3kF]:")
    println("$res_v2\n$meas_v2 ($(round(relerr2, digits=2))% relative error)")

    println("\n c1l over eTF^2 via integration of exact bare integrand on [0, 30kF]:")
    println("$res_v3\n$meas_v3 ($(round(relerr3, digits=2))% relative error)")

    println("\n c1l over eTF^2 via integration of exact bare integrand on [0, ∞):")
    println("$res_v4\n$meas_v4 ($(round(relerr4, digits=7))% relative error)")

    return
end

main()