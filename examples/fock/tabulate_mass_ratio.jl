using CodecZlib
using ElectronGas
using ElectronLiquid
using Interpolations
using JLD2
using LsqFit
using Measurements
using Polynomials
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np

# Dimensionless expansion parameter α for the UEG (powers of αrₛ)
const alpha = (4 / 9π)^(1 / 3)

"""
Exact expression for the Fock self-energy
in terms of the dimensionless Lindhard function.
"""
function fock_self_energy_exact(k, p::ParaMC)
    # The (dimensionful) value at k = 0 is minus the Thomas-Fermi energy
    eTF = p.qTF^2 / (2 * p.me)
    return -eTF * UEG_MC.lindhard(k / p.kF)
end
function fock_self_energy_exact(ks::Vector{Float64}, p::ParaMC)
    return [fock_self_energy_exact(k, p) for k in ks]
end

"""
Exact expression for the Fock quasiparticle energy
in terms of the dimensionless Lindhard function.
"""
function qp_fock_exact(k, p::ParaMC)
    return k^2 / (2 * p.me) + fock_self_energy_exact(k, p)
end
function qp_fock_exact(ks::Vector{Float64}, p::ParaMC)
    return [qp_fock_exact(k, p) for k in ks]
end

"""x ≡ k / kF (dimensionless wavenumber)."""
function fock_mass_ratio_exact(x::Float64, p::ParaMC)
    # return 1 +
    #        (param.e0^2 * param.me / (2pi * param.kF)) *
    #        ((1 + x^2) * log(abs((1 + x) / (1 - x))) / x - 2) / x^2
    return 1 + (alpha * p.rs / 2π) * ((1 + x^2) * log(abs((1 + x) / (1 - x))) / x - 2) / x^2
end
function fock_mass_ratio_exact(xs::Vector{Float64}, p::ParaMC)
    return [fock_mass_ratio_exact(x, p) for x in xs]
end
fock_mass_ratio_k0(p::ParaMC) = 1 + (4 / 3π) * alpha * p.rs

"""Compute the coefficient of determination (r-squared) for a dataset."""
function rsquared(xs, ys, yhats)
    ybar = sum(yhats) / length(yhats)
    ss_res = sum((ys .- yhats) .^ 2)
    ss_tot = sum((ys .- ybar) .^ 2)
    return 1 - ss_res / ss_tot
end
function rsquared(xs, ys, yhats, fit::LsqFit.LsqFitResult)
    ybar = sum(yhats) / length(yhats)
    ss_res = sum(fit.resid .^ 2)
    ss_tot = sum((ys .- ybar) .^ 2)
    return 1 - ss_res / ss_tot
end

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

    # Number of evals below and above kF
    neval123 = 1e10
    neval4 = 5e10
    neval = max(neval123, neval4)

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    n_min = 1  # True minimal loop order for this observable
    min_order = 1
    max_order = 4
    min_order_plot = 1
    max_order_plot = 4

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Distinguish results with different counterterm schemes used in the original run
    ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # UEG parameters for MC integration
    loadparam = ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    # Load raw data
    if max_order == 4
        max_together = 3
    else
        max_together = max_order
    end
    savename =
        "results/data/sigma_x_n=$(max_together)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval123)_$(solver)$(ct_string)"
    orders, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(loadparam))"
        return f[key]
    end
    if max_order == 4
        savename =
            "results/data/sigma_x_n=$(max_order)_rs=$(rs)_" *
            "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval4)_$(solver)$(ct_string)"
        orders4, param4, kgrid4, partitions4, res4 = jldopen("$savename.jld2", "a+") do f
            key = "$(UEG.short(loadparam))"
            return f[key]
        end
    end

    # Get dimensionless k-grid (k / kF) and index corresponding to the Fermi energy
    k_kf_grid = kgrid / param.kF
    if max_order == 4
        k_kf_grid4 = kgrid4 / param.kF
    end
    ikF = findfirst(x -> x == 1.0, k_kf_grid)

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    for (k, v) in data
        data[k] = v / (factorial(k[2]) * factorial(k[3]))
    end
    # Add 4th order results to data dict
    if max_order == 4
        data4 = UEG_MC.restodict(res4, partitions4)
        for (k, v) in data4
            data4[k] = v / (factorial(k[2]) * factorial(k[3]))
        end
        merge!(data, data4)
    end

    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])
    # println(merged_data)

    if min_order_plot == 1
        if 1 in orders
            # The nondimensionalized Fock self-energy is the negative Lindhard function
            exact = -UEG_MC.lindhard.(kgrid / param.kF)
            # Check the MC result at k = 0 against the exact (non-dimensionalized)
            # Fock (exhange) self-energy: Σx(0) / E_{TF} = -F(0) = -1
            meas = merged_data[(1, 0)]
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
        # Set bare result manually using exact Fock self-energy / eTF
        sigma_fock_over_eTF_exact = -UEG_MC.lindhard.(k_kf_grid)
        merged_data[(1, 0)] = measurement.(sigma_fock_over_eTF_exact, 0.0)  # treat quadrature data as numerically exact
    end

    # Reexpand merged data in powers of μ
    δμ = load_mu_counterterm(
        param;
        max_order=max_order - n_min,
        parafilename="examples/counterterms/data/para.csv",
        ct_filename="examples/counterterms/data/data_Z$(ct_string).jld2",
        verbose=1,
    )
    println("Computed δμ: ", δμ)
    sigma_x =
        UEG_MC.chemicalpotential_renormalization_sigma(merged_data, δμ; max_order=max_order)
    # Test manual renormalization with exact lowest-order chemical potential
    δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
    # Σₓ⁽²⁾ = Σₓ_{2,0} + δμ₁ Σₓ_{1,1}
    sigma_x_2_manual = merged_data[(2, 0)] + δμ1_exact * merged_data[(1, 1)]
    stdscores = stdscore.(sigma_x[2], sigma_x_2_manual)
    worst_score = argmax(abs, stdscores)
    println("Exact δμ₁: ", δμ1_exact)
    println("Computed δμ₁: ", δμ[1])
    println(
        "Worst standard score for total result to 3rd " *
        "order (auto vs exact+manual): $worst_score",
    )
    # Aggregate the full results for Σₓ up to order N
    sigma_x_over_eTF_total = UEG_MC.aggregate_orders(sigma_x)

    println(UEG.paraid(param))
    println(partitions)
    println(res)

    # Thomas-Fermi energy
    eTF = param.qTF^2 / (2 * param.me)

    # Bare dispersion in units of the Thomas-Fermi energy (for effective mass related plots)
    Ek = kgrid .^ 2 / (2 * param.me)
    Ek_over_eTF = Ek / eTF

    # Fock self-energy
    sigma_fock_exact = fock_self_energy_exact(kgrid, param)
    # Exact Fock energy at the Fermi surface
    EF_fock = qp_fock_exact(param.kF, param)
    println("ΣF(k = 0) (pred, exact):", sigma_fock_exact[1], " ", -eTF)
    println("EqpF(k = kF) (pred, exact):", EF_fock, " ", param.EF - eTF / 2)
    # @assert sigma_fock_exact[1] ≈ -eTF
    # @assert EF_fock ≈ param.EF - eTF / 2

    # Exact results on dense (quadrature) grids
    kgrid_quad = param.kF * np.linspace(0.0, 3.0; num=600)

    # Fock quasiparticle energy
    E_fock_quad = qp_fock_exact(kgrid_quad, param)

    # No fixed point (zpe is a free parameter)
    @. quasiparticle_model(k, p) = p[1] + k^2 / (2.0 * p[2])

    z1 = 0.87
    z2 = 0.80
    if rs == 1.0
        zexp = 0.87
    elseif rs == 2.0
        zexp = 0.80
    else
        zexp = 1.0
    end
    @. quasiparticle_model_with_zexp(k, p) = p[1] + zexp * k^2 / (2.0 * p[2])

    # Initial parameters for curve fitting procedure
    p0      = [-eTF, 1.0]  # E₀=0 and m=mₑ
    p0_fock = [1.0]        # m=mₑ

    # Gridded data for k ≤ kF
    k_data = kgrid_quad[kgrid_quad .< param.kF]
    E_fock_data = E_fock_quad[kgrid_quad .< param.kF]

    # Least-squares fit to (exact) Fock data
    fit_fock = curve_fit(
        (k, p) -> quasiparticle_model(k, [-eTF, p[1]]),
        k_data,
        E_fock_data,
        p0_fock,
    )

    # Least-squares quasiparticle fit for the Fock dispersion
    meff_fock = fit_fock.param[1]
    qp_fit_fock(k) = quasiparticle_model(k, [-eTF, meff_fock])
    @assert qp_fit_fock(0) ≈ -eTF

    # Coefficient of determination (r²)
    r2 = rsquared(k_data, E_fock_data, qp_fit_fock(k_data), fit_fock)

    # Low-energy effective mass ratio (mₑ/m⋆)(k≈0) from quasiparticle fit
    low_en_mass_ratio_fock = param.me / meff_fock
    println(
        "Fock low-energy effective mass info from quadratic fit: " *
        "m⋆_F ≈ $meff_fock, (mₑ/m⋆_F)(k=0) ≈ $low_en_mass_ratio_fock, r2=$r2",
    )
    mass_ratio_fit   = 1 / low_en_mass_ratio_fock
    mass_ratio_exact = 1 / fock_mass_ratio_k0(param)
    rel_error        = abs(mass_ratio_exact - mass_ratio_fit) / mass_ratio_exact
    println("Percent error vs exact low-energy limit: $(rel_error * 100)%")

    for (i, N) in enumerate(min_order:max_order_plot)
        # Eqp = ϵ(k) + Σₓ(k)
        Eqp_over_eTF = Ek_over_eTF .+ sigma_x_over_eTF_total[N]

        # Get means and error bars
        means_qp = Measurements.value.(Eqp_over_eTF)

        # Gridded data for k < kF
        k_data = kgrid[kgrid .< param.kF]
        Eqp_data = means_qp[kgrid .< param.kF] * eTF

        # Least-squares quasiparticle fit
        fit_N = curve_fit(quasiparticle_model, k_data, Eqp_data, p0)
        qp_fit_N(k) = quasiparticle_model(k, fit_N.param)

        # Coefficients of determination (r²)
        r2 = rsquared(k_data, Eqp_data, qp_fit_N(k_data), fit_N)

        # Low-energy effective mass ratio (mₑ/m⋆)(k≈0) from quasiparticle fit
        meff_N = fit_N.param[2]
        low_en_mass_ratio_N = param.me / meff_N
        println(
            "(N=$N) Low-energy effective mass info from quadratic fit: " *
            "m⋆(k=0) ≈ $meff_N, (mₑ/m⋆)(k=0) ≈ $low_en_mass_ratio_N, r2=$r2",
        )
    end

    return
end

main()
