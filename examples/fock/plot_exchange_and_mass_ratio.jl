using CodecZlib
using ElectronGas
using ElectronLiquid
using Interpolations
using JLD2
using LsqFit
using Measurements
using Parameters
using Polynomials
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

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
    neval = 5e10

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order = 2
    max_order = 4
    min_order_plot = 1
    max_order_plot = 4

    # Save total results
    save = true

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""

    # Full renormalization
    ct_string = "with_ct_mu_lambda"

    # UEG parameters for MC integration
    loadparam = ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    # NOTE: Taking N=3 data from N=4 run, renorm for N=4 not yet implemented!
    # loadparam = ParaMC(; order=4, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    savename =
        "results/data/sigma_x_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval)_$(solver)"

    # NOTE: Taking N=3 data from N=4 run, renorm for N=4 not yet implemented!
    # savename =
    #     "results/data/sigma_x_n=4_rs=$(rs)_" *
    #     "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval)_$(solver)"

    orders, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(loadparam))"
        return f[key]
    end

    # Get dimensionless k-grid (k / kF) and index corresponding to the Fermi energy
    k_kf_grid = kgrid / param.kF
    ikF = findfirst(x -> x == 1.0, k_kf_grid)

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])
    # println(merged_data)

    if min_order_plot == 1
        # Set bare result manually using exact Fock self-energy / eTF
        sigma_fock_over_eTF_exact = -UEG_MC.lindhard.(k_kf_grid)
        merged_data[(1, 0)] = measurement.(sigma_fock_over_eTF_exact, 0.0)  # treat quadrature data as numerically exact
    end

    println(param)
    ctparam = param
    # ctparam = reconstruct(param; order=3)

    # Reexpand merged data in powers of μ
    z, μ = UEG_MC.load_z_mu(ctparam)
    # z, μ = UEG_MC.load_z_mu(param)
    δz, δμ = CounterTerm.sigmaCT(max_order, μ, z; verbose=1)
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
    @assert worst_score ≤ 10

    if save
        savename =
            "results/data/rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_$(intn_str)$(solver)_$(ct_string)"
        f = jldopen("$savename.jld2", "a+")
        # NOTE: no bare result for c1b observable (accounted for in c1b0)
        for N in min_order_plot:max_order
            # Skip exact Fock (N = 1) result
            N == 1 && continue
            if haskey(f, "sigma_x") &&
               haskey(f["sigma_x"], "N=$N") &&
               haskey(f["sigma_x/N=$N"], "neval=$(neval)")
                @warn("replacing existing data for N=$N, neval=$(neval)")
                delete!(f["sigma_x/N=$N"], "neval=$(neval)")
            end
            f["sigma_x/N=$N/neval=$neval/meas"] = sigma_x_over_eTF_total[N]
            f["sigma_x/N=$N/neval=$neval/param"] = param
            f["sigma_x/N=$N/neval=$neval/kgrid"] = kgrid
        end
    end

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # Plot for each aggregate order

    # Σₓ(k) / eTF (dimensionless moment)
    fig1, ax1 = plt.subplots()
    # Compare result to exact non-dimensionalized Fock self-energy (-F(k / kF))
    ax1.plot(k_kf_grid, -UEG_MC.lindhard.(k_kf_grid), "k"; label="\$N=1\$ (exact, \$T=0\$)")
    for (i, N) in enumerate(min_order:max_order_plot)
        N == 1 && continue
        # Get means and error bars from the result up to this order
        means = Measurements.value.(sigma_x_over_eTF_total[N])
        stdevs = Measurements.uncertainty.(sigma_x_over_eTF_total[N])
        # Data gets noisy above 3rd loop order
        # marker = N > 2 ? "o-" : "-"
        marker = "o-"
        ax1.plot(
            k_kf_grid,
            means,
            marker;
            markersize=2,
            color="C$(i-1)",
            label="\$N=$N\$ ($solver)",
        )
        ax1.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            color="C$(i-1)",
            alpha=0.4,
        )
    end
    ax1.legend(; loc="lower right")
    ax1.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    ax1.set_xlabel("\$k / k_F\$")
    ax1.set_ylabel("\$\\Sigma_{x}(k) \\,/\\, \\epsilon_{\\mathrm{TF}}\$")
    xloc = 1.75
    yloc = -0.5
    ydiv = -0.095
    ax1.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax1.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    fig1.tight_layout()
    fig1.savefig(
        "results/fock/sigma_x_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver).pdf",
    )

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
    @assert sigma_fock_exact[1] ≈ -eTF
    @assert EF_fock ≈ param.EF - eTF / 2

    # Exact results on dense (quadrature) grids
    kgrid_quad = param.kF * np.linspace(0.0, 3.0; num=600)
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    Ek_quad = kgrid_quad .^ 2 / (2 * param.me)
    Ek_over_eTF_quad = Ek_quad / eTF
    ikF_quad = findall(x -> x == 1.0, k_kf_grid_quad)

    fig12, ax12 = plt.subplots()
    k_kf_dense = np.linspace(0.0, 2.0; num=600)
    ax12.axhline(
        1.0 ./ fock_mass_ratio_k0(param);
        label="\$\\left(1 + \\frac{4}{3\\pi}(\\alpha r_s)\\right)^{-1}\$",
        color="gray",
    )
    ax12.plot(k_kf_dense, 1.0 ./ fock_mass_ratio_exact(k_kf_dense, param))
    ax12.legend(; loc="best")
    ax12.set_xlim(0, 2)
    ax12.set_xlabel("\$k / k_F\$")
    ax12.set_ylabel("\$\\left(m^\\star_F \\left/ m\\right)\\right.(k)\$")
    xloc = 1.25
    yloc = 0.7
    ax12.text(
        xloc,
        yloc,
        "\$r_s = $rs,\\, \\beta \\hspace{0.1em} \\epsilon_F = $beta\$";
        fontsize=14,
    )
    fig12.tight_layout()
    fig12.savefig(
        "results/fock/fock_eff_mass_ratio_exact_rs=$(param.rs)_beta_ef=$(param.beta).pdf",
    )

    # Moment quasiparticle energy
    # Extract effective masses from quadratic fits to data for k ≤ kF
    fig2, ax2 = plt.subplots()

    # First order from exact expressions
    sigma_fock_exact_quad = fock_self_energy_exact(kgrid_quad, param)

    # Fock quasiparticle energy
    E_fock_quad = qp_fock_exact(kgrid_quad, param)
    E_fock_over_eTF_quad = E_fock_quad / eTF

    # No fixed point (zpe is a free parameter)
    @. model(k, p) = p[1] + k^2 / (2.0 * p[2])

    # Fixed point at Eqp(0) = -1 (exact Fock zpe)
    # @. model_fix_E0(k, p) = -eTF + k^2 / (2.0 * p[1])
    # Fixed point at Eqp(kF) (exact Fock energy at k = kF)
    # @. model_fix_EF(k, p) = EF_fock + (k^2 - param.kF^2) / (2.0 * p[1])
    # General quadratic fit
    # @. model_quadratic(k, p) = p[1] + p[2] * k + p[3] * k^2

    # Initial parameters for curve fitting procedure
    p0 = [1.0]        # m=mₑ

    # p0          = [-1.0, 1.0]  # E₀=0 and m=mₑ
    # p0_fixed_pt = [1.0]        # m=mₑ
    # p0_quadratic = [-1.0, 0.1, 1.0]

    # Gridded data for k ≤ kF
    k_data = kgrid_quad[kgrid_quad .≤ param.kF]
    E_fock_data = E_fock_quad[kgrid_quad .≤ param.kF]

    # Least-squares fit of models to data
    fit_fock = curve_fit((k, p) -> model(k, [-eTF, p[1]]), k_data, E_fock_data, p0)

    # fit_fock_fix_E0 = curve_fit(model_fix_E0, k_data, E_fock_data, p0_fixed_pt)
    # fit_fock_fix_EF = curve_fit(model_fix_EF, k_data, E_fock_data, p0_fixed_pt)
    # fit_quadratic = curve_fit(model_quadratic, k_data, E_fock_data, p0_quadratic)

    # QP params
    meff_fock = fit_fock.param[1]
    println(meff_fock)

    # zpe_fock         = fit_fock.param[1]
    # meff_fock        = fit_fock.param[2]
    # meff_fock_fix_E0 = fit_fock_fix_E0.param[1]
    # meff_fock_fix_EF = fit_fock_fix_EF.param[1]
    # meff_quadratic   = 1 / (2 * fit_quadratic.param[3])
    # println(meff_fock, " ", meff_fock_fix_E0, " ", meff_fock_fix_EF)
    # println(meff_fock, " ", meff_fock_fix_E0, " ", meff_fock_fix_EF, " ", meff_quadratic)

    # Fits to Eqp(k)
    qp_fit_fock(k) = model(k, [-eTF, meff_fock])
    @assert qp_fit_fock(0) ≈ -eTF

    # qp_fit_fock_fix_E0(k) = model_fix_E0(k, fit_fock_fix_E0.param)
    # qp_fit_fock_fix_EF(k) = model_fix_EF(k, fit_fock_fix_EF.param)
    # qp_fit_quadratic(k)   = model_quadratic(k, fit_quadratic.param)
    # @assert qp_fit_fock_fix_E0(0) ≈ -eTF
    # @assert qp_fit_fock_fix_EF(param.kF) ≈ param.EF - eTF / 2

    # Coefficients of determination (r²)
    r2 = rsquared(k_data, E_fock_data, qp_fit_fock(k_data), fit_fock)

    # r2_fix_E0 = rsquared(k_data, E_fock_data, qp_fit_fock_fix_E0(k_data), fit_fock_fix_E0)
    # r2_fix_EF = rsquared(k_data, E_fock_data, qp_fit_fock_fix_EF(k_data), fit_fock_fix_EF)
    # r2_quadratic = rsquared(k_data, E_fock_data, qp_fit_quadratic(k_data), fit_quadratic)

    # Low-energy effective mass ratios (mₑ/m⋆)(k≈0) from quasiparticle fits
    low_en_mass_ratio_fock = param.me / meff_fock

    # low_en_mass_ratio_fock_fix_E0 = param.me / meff_fock_fix_E0
    # low_en_mass_ratio_fock_fix_EF = param.me / meff_fock_fix_EF
    # low_en_mass_ratio_quadratic   = param.me / meff_quadratic

    # RESULT: Best fit is given by model 2, although this is not the fit with highest r².
    #         We hence use model 2, as it is most physical relevant with highest r².
    #         While the exact zero-point energy is not known beyond HF level, we will
    #         simply constrain it to the data at k = 0.

    # println(
    #     "Fock effective zero-point energy over eTF from quadratic fit: E₀ = $(zpe_fock / eTF)",
    # )

    # ZPEs
    println("Exact Fock ZPE over eTF: Eqp(0) = $(E_fock_quad[1] / eTF)")

    # Mass ratios
    println(
        "Fock low-energy effective mass ratio from quadratic fit: " *
        "(mₑ/m⋆)(k=0) ≈ $low_en_mass_ratio_fock, r2=$r2",
    )
    mass_ratio_fit   = 1 / low_en_mass_ratio_fock
    mass_ratio_exact = 1 / fock_mass_ratio_k0(param)
    rel_error        = abs(mass_ratio_exact - mass_ratio_fit) / mass_ratio_exact
    println("Percent error vs exact low-energy limit: $(rel_error * 100)%")

    ax2.axhline(0; linestyle="--", color="gray")
    # ax2.plot(
    #     k_kf_grid_quad,
    #     Ek_quad / eTF,
    #     "k";
    #     linestyle="--",
    #     label="\$\\epsilon_k / \\epsilon_{\\mathrm{TF}} = (k / q_{\\mathrm{TF}})^2\$",
    # )
    ax2.plot(
        k_kf_grid_quad,
        -1 .+ (π / 4alpha + 1 / 3) * k_kf_grid_quad .^ 2;
        color="k",
        label="\$\\epsilon_{\\mathrm{HF}}(k \\rightarrow 0) / \\epsilon_{\\mathrm{TF}} \\sim -1 + \\left(\\frac{\\pi}{4\\alpha} + \\frac{1}{3}\\right) \\left( \\frac{k}{k_F} \\right)^2\$",
    )
    ax2.plot(
        kgrid_quad / param.kF,
        qp_fit_fock(kgrid_quad) / eTF,
        "gray";
        label="\$\\left(\\epsilon_0 + \\frac{k^2}{2 m^\\star_{\\mathrm{HF}}} \\right) \\Big/ \\epsilon_{\\mathrm{TF}}\$ (quasiparticle fit)",
    )
    ax2.plot(
        kgrid_quad / param.kF,
        E_fock_quad / eTF,
        "C0";
        label="\$N=1\$ (exact, \$T=0\$)",
    )
    for (i, N) in enumerate(min_order:max_order_plot)
        N == 1 && continue
        # Eqp = ϵ(k) + Σₓ(k)
        Eqp_over_eTF = Ek_over_eTF .+ sigma_x_over_eTF_total[N]
        # Get means and error bars
        means_qp = Measurements.value.(Eqp_over_eTF)
        stdevs_qp = Measurements.uncertainty.(Eqp_over_eTF)

        # Gridded data for k ≤ kF
        k_data = kgrid[kgrid ≤ param.kF]
        Eqp_data = eTF * Eqp_over_eTF[kgrid ≤ param.kF]

        # Constrain ZPE to measured data at k = 0
        E0 = Eqp_data[1]

        # Least-squares fit of models to data
        fit_N = curve_fit((k, p) -> model(k, [E0, p[1]]), k_data, Eqp_data, p0)

        # QP params
        meff_N = fit_N.param[1]

        # Fits to Eqp(k)
        qp_fit(k) = model(k, [E0, meff_N])
        @assert qp_fit(0) ≈ E0

        # Coefficients of determination (r²)
        r2 = rsquared(k_data, Eqp_data, qp_fit(k_data), fit_N)

        # Low-energy effective mass ratios (mₑ/m⋆)(k≈0) from quasiparticle fits
        low_en_mass_ratio = param.me / meff_N

        # Mass ratio
        println(
            "N=$N low-energy effective mass ratio from quadratic fit: " *
            "(mₑ/m⋆)(k=0) ≈ $low_en_mass_ratio, r2=$r2",
        )

        # Data gets noisy above 3rd loop order
        # marker = N > 2 ? "o-" : "-"
        marker = "o-"
        ax2.plot(
            k_kf_grid,
            means_qp,
            marker;
            markersize=2,
            color="C$i",
            label="\$N=$N\$ ($solver)",
        )
        ax2.fill_between(
            k_kf_grid,
            means_qp - stdevs_qp,
            means_qp + stdevs_qp;
            color="C$i",
            alpha=0.4,
        )
    end
    ax2.legend(; loc="lower right")
    # ax2.set_xlim(minimum(k_kf_grid), 1)
    ax2.set_xlim(minimum(k_kf_grid), 1.5)
    ax2.set_ylim(-2.0, 3.0)
    ax2.set_xlabel("\$k / k_F\$")
    ax2.set_ylabel(
        "\$\\epsilon_{\\mathrm{momt.}}(k) \\,/\\, \\epsilon_{\\mathrm{TF}} =  \\left(\\epsilon_{k} + \\Sigma_{x}(k)\\right) \\,/\\, \\epsilon_{\\mathrm{TF}} \$",
    )
    xloc = 0.1
    yloc = 2.5
    ydiv = -0.5
    ax2.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax2.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    ax2.text(
        xloc,
        yloc + 2 * ydiv,
        "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=12,
    )
    fig2.tight_layout()
    fig2.savefig(
        "results/fock/moment_qp_energy_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver).pdf",
    )

    # Mass ratio
    fig3, ax3 = plt.subplots()
    ax3.axhline(1; linestyle="--", color="gray")
    # ax3.plot(
    #     k_kf_grid,
    #     1.0 .+ sigma_fock_exact ./ Ek,
    #     "k";
    #     label="\$N=1\$ (exact, \$T=0\$)",
    # )
    # First order from exact expressions
    # Σₓ(k) / ϵ(k)
    sigma_fock_over_Ek = sigma_fock_exact_quad ./ Ek_quad
    # m_e / m_{momt}(k) ≈ 1 + Σₓ(k) / ϵ(k)
    effmass_ratio = 1.0 .+ sigma_fock_over_Ek
    effmass_ratio_kF = (1.0 .+ sigma_fock_exact ./ Ek)[ikF]
    # println("Naive effective mass at k = kF (N=1): ", effmass_ratio_kF)
    ax3.plot(k_kf_grid_quad, effmass_ratio, "C0"; label="\$N=1\$ (exact, \$T=0\$)")
    for (i, N) in enumerate(min_order:max_order_plot)
        N == 1 && continue
        # Σₓ(k) / ϵ(k)
        sigma_x_over_Ek = eTF * sigma_x_over_eTF_total[N] ./ Ek
        # m_e / m_{momt}(k) ≈ 1 + Σₓ(k) / ϵ(k)
        mass_ratio = 1.0 .+ sigma_x_over_Ek
        # Get means and error bars from the mass ratio up to this order
        means_mass_ratio = Measurements.value.(mass_ratio)
        stdevs_mass_ratio = Measurements.uncertainty.(mass_ratio)

        # println("Naive effective mass at k = kF (N=$N): ", means_mass_ratio[ikF])
        # Data gets noisy above 3rd loop order
        # marker = N > 2 ? "o-" : "-"
        marker = "o-"
        ax3.plot(
            k_kf_grid,
            means_mass_ratio,
            marker;
            markersize=2,
            color="C$i",
            label="\$N=$N\$ ($solver)",
        )
        ax3.fill_between(
            k_kf_grid,
            means_mass_ratio - stdevs_mass_ratio,
            means_mass_ratio + stdevs_mass_ratio;
            color="C$i",
            alpha=0.4,
        )
    end
    ax3.legend(; loc="lower right")
    ax3.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    ax3.set_ylim(0, 1.1)
    ax3.set_xlabel("\$k / k_F\$")
    ax3.set_ylabel("\$\\epsilon_{\\mathrm{momt.}}(k) / \\epsilon_k\$")
    # ax3.set_ylabel("\$m_e \\,/\\, m_{\\mathrm{momt.}}\$")
    xloc = 1.75
    yloc = 0.75
    ydiv = -0.095
    ax3.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax3.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax3.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    fig3.tight_layout()
    fig3.savefig(
        "results/fock/moment_energy_ratio_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver).pdf",
    )

    plt.close("all")
    return
end

main()
