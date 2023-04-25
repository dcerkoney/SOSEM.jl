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
    neval123 = 1e10
    neval4 = 5e10
    neval5 = 5e10
    neval = max(neval123, neval4, neval5)

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    n_min = 1  # True minimal loop order for this observable
    min_order = 1
    max_order = 5
    min_order_plot = 1
    max_order_plot = 5

    # Save total results
    save = true

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
    # if max_order >= 5
    #     max_together = 4
    if max_order >= 4
        max_together = 3
    else
        max_together = max_order
    end
    savename =
        "results/data/sigma_x_n=$(max_together)_rs=$(rs)_" *
        # "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval4)_$(solver)$(ct_string)"
        "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval123)_$(solver)$(ct_string)"
    orders, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(loadparam))"
        return f[key]
    end
    if max_order >= 4
        savename =
            "results/data/sigma_x_n=4_rs=$(rs)_" *
            "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval4)_$(solver)$(ct_string)"
        orders4, param4, kgrid4, partitions4, res4 = jldopen("$savename.jld2", "a+") do f
            key = "$(UEG.short(loadparam))"
            return f[key]
        end
    end
    if max_order >= 5
        savename =
            "results/data/sigma_x_n=5_rs=$(rs)_" *
            "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval5)_$(solver)$(ct_string)"
        orders5, param5, kgrid5, partitions5, res5 = jldopen("$savename.jld2", "a+") do f
            key = "$(UEG.short(loadparam))"
            return f[key]
        end
    end

    # Get dimensionless k-grid (k / kF) and index corresponding to the Fermi energy
    k_kf_grid = kgrid / param.kF
    if max_order >= 4
        k_kf_grid4 = kgrid4 / param.kF
    end
    if max_order >= 5
        k_kf_grid5 = kgrid5 / param.kF
    end
    ikF = findfirst(x -> x == 1.0, k_kf_grid)

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    for (k, v) in data
        data[k] = v / (factorial(k[2]) * factorial(k[3]))
    end
    # Add 4th order results to data dict
    if max_order >= 4
        data4 = UEG_MC.restodict(res4, partitions4)
        for (k, v) in data4
            data4[k] = v / (factorial(k[2]) * factorial(k[3]))
        end
        merge!(data, data4)
    end
    # Add 5th order results to data dict
    if max_order >= 5
        data5 = UEG_MC.restodict(res5, partitions5)
        for (k, v) in data5
            data5[k] = v / (factorial(k[2]) * factorial(k[3]))
        end
        merge!(data, data5)
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
    ct_filename = "examples/counterterms/data_Z$(ct_string).jld2"
    z, μ = UEG_MC.load_z_mu(param; ct_filename=ct_filename)
    # Add Taylor factors to CT data
    for (p, v) in z
        z[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    for (p, v) in μ
        μ[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    # δz, δμ = CounterTerm.sigmaCT(2, μ, z; verbose=1)  # TODO: Fix order 3
    δz, δμ = CounterTerm.sigmaCT(max_order - 1, μ, z; verbose=1)
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

    if save
        savename =
            "results/data/rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_$(intn_str)$(solver)$(ct_string)"
        f = jldopen("$savename.jld2", "a+"; compress=true)
        # NOTE: no bare result for c1b observable (accounted for in c1b0)
        for N in min_order_plot:max_order
            if N < 4
                num_eval = neval123
            elseif N == 4
                num_eval = neval4
            elseif N == 5
                num_eval = neval5
            else
                error("An unexpected error occurred!")
            end
            # num_eval = N == 4 ? neval4 : neval123
            # num_eval = N == 5 ? neval5 : neval4
            # Skip exact Fock (N = 1) result
            N == 1 && continue
            if haskey(f, "sigma_x") &&
               haskey(f["sigma_x"], "N=$N") &&
               haskey(f["sigma_x/N=$N"], "neval=$num_eval")
                @warn("replacing existing data for N=$N, neval=$num_eval")
                delete!(f["sigma_x/N=$N"], "neval=$num_eval")
            end
            f["sigma_x/N=$N/neval=$num_eval/meas"] = sigma_x_over_eTF_total[N]
            f["sigma_x/N=$N/neval=$num_eval/param"] = param
            f["sigma_x/N=$N/neval=$num_eval/kgrid"] = kgrid
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
        # N == 1 && continue
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
            label="\$N=$(N)\$ ($solver)",
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
    ax1.set_ylabel(
        "\$C^{(0)}_\\sigma(k) \\,/\\, \\epsilon_{\\mathrm{TF}} = \\Sigma_{x}(k) \\,/\\, \\epsilon_{\\mathrm{TF}}\$",
    )
    # ax1.set_ylabel("\$\\Sigma_{x}(k) \\,/\\, \\epsilon_{\\mathrm{TF}}\$")
    xloc = 1.5
    yloc = -0.4
    ydiv = -0.095
    # xloc = 1.5
    # yloc = -0.6
    # ydiv = -0.175
    ax1.text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax1.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    ax1.text(
        xloc,
        yloc + 2 * ydiv,
        "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=12,
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
    # @assert sigma_fock_exact[1] ≈ -eTF
    # @assert EF_fock ≈ param.EF - eTF / 2

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
    @. quasiparticle_model(k, p) = p[1] + k^2 / (2.0 * p[2])

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
    println("meff_fock = $meff_fock")
    qp_fit_fock(k) = quasiparticle_model(k, [-eTF, meff_fock])
    @assert qp_fit_fock(0) ≈ -eTF

    # Coefficient of determination (r²)
    r2 = rsquared(k_data, E_fock_data, qp_fit_fock(k_data), fit_fock)

    # Low-energy effective mass ratio (mₑ/m⋆)(k≈0) from quasiparticle fit
    low_en_mass_ratio_fock = param.me / meff_fock
    println(
        "Fock low-energy effective mass ratio from quadratic fit: " *
        "(mₑ/m⋆)(k=0) ≈ $low_en_mass_ratio_fock, r2=$r2",
    )
    mass_ratio_fit   = 1 / low_en_mass_ratio_fock
    mass_ratio_exact = 1 / fock_mass_ratio_k0(param)
    rel_error        = abs(mass_ratio_exact - mass_ratio_fit) / mass_ratio_exact
    println("Percent error vs exact low-energy limit: $(rel_error * 100)%")

    ax2.axhline(0; linestyle="--", color="gray", linewidth=1)
    # ax2.plot(
    #     k_kf_grid_quad,
    #     Ek_quad / eTF,
    #     "k";
    #     linestyle="--",
    #     label="\$\\epsilon_k / \\epsilon_{\\mathrm{TF}} = (k / q_{\\mathrm{TF}})^2\$",
    # )
    ax2.plot(
        k_kf_grid_quad,
        -1 .+ (π / (4alpha * param.rs) + 1 / 3) * k_kf_grid_quad .^ 2;
        color="k",
        # linestyle="--",
        label="\$\\epsilon_{\\mathrm{HF}}(k \\rightarrow 0) / \\epsilon_{\\mathrm{TF}} \\sim -1 + \\left(\\frac{\\pi}{4\\alpha r_s} + \\frac{1}{3}\\right) \\left( \\frac{k}{k_F} \\right)^2\$",
    )
    # ax2.plot(
    #     kgrid_quad / param.kF,
    #     E_fock_quad / eTF,
    #     "k";
    #     label="\$N=1\$ (exact, \$T=0\$)",
    # )
    # ax2.plot(
    #     kgrid_quad / param.kF,
    #     qp_fit_fock(kgrid_quad) / eTF,
    #     "C0";
    #     linestyle="--",
    #     label="\$N=1\$ (quasiparticle fit)",
    #     # label="\$\\left(\\epsilon_0 + \\frac{k^2}{2 m^\\star_{\\mathrm{HF}}} \\right) \\Big/ \\epsilon_{\\mathrm{TF}}\$ (quasiparticle fit)",
    # )
    ic = 0

    m_test = [0.2, 0.3, 0.4, 0.5]
    e_test = [-1.0, -1.25, -1.5, -1.75]
    for (i, N) in enumerate(min_order:max_order_plot)
        # N == 1 && continue

        # Eqp = ϵ(k) + Σₓ(k)
        Eqp_over_eTF = Ek_over_eTF .+ sigma_x_over_eTF_total[N]

        # Get means and error bars
        means_qp = Measurements.value.(Eqp_over_eTF)
        stdevs_qp = Measurements.uncertainty.(Eqp_over_eTF)

        # Gridded data for k < kF
        k_data = kgrid[kgrid .< param.kF]
        Eqp_data = means_qp[kgrid .< param.kF] * eTF
        # Eqp_data = e_test[i] .+ k_data .^ 2 / (2 * m_test[i])

        # Least-squares quasiparticle fit
        fit_N = curve_fit(quasiparticle_model, k_data, Eqp_data, p0)
        qp_fit_N(k) = quasiparticle_model(k, fit_N.param)

        # Coefficients of determination (r²)
        r2 = rsquared(k_data, Eqp_data, qp_fit_N(k_data), fit_N)

        # Low-energy effective mass ratio (mₑ/m⋆)(k≈0) from quasiparticle fit
        meff_N = fit_N.param[2]
        low_en_mass_ratio_N = param.me / meff_N
        println(
            "(N=$N) Low-energy effective mass ratio from quadratic fit: " *
            "(mₑ/m⋆)(k=0) ≈ $low_en_mass_ratio_N, r2=$r2",
        )
        if N == max_order_plot
            ax2.text(
                0.175,
                0.5,
                "\$(N=$N) \\; m^\\star / m \\approx $(round(meff_N / param.me; digits=5))\$";
                fontsize=12,
            )
        end

        # Data gets noisy above 3rd loop order
        # marker = N > 2 ? "o-" : "-"
        marker = "o-"
        ax2.plot(
            k_kf_grid,
            means_qp,
            marker;
            markersize=2,
            color="C$ic",
            label="\$N=$(N)\$ ($solver)",
        )
        ax2.fill_between(
            k_kf_grid,
            means_qp - stdevs_qp,
            means_qp + stdevs_qp;
            color="C$ic",
            alpha=0.4,
        )
        ax2.plot(
            kgrid_quad / param.kF,
            qp_fit_N(kgrid_quad) / eTF;
            color="C$ic",
            linestyle="--",
            # label="\$N=$N\$ (quasiparticle fit)",
        )
        ic += 1
    end
    ax2.legend(; loc="upper left")
    # ax2.set_xlim(minimum(k_kf_grid), 1)
    ax2.set_xlim(minimum(k_kf_grid), 1.5)
    ax2.set_ylim(-2.0, 3.0)
    # ax2.set_ylim(-2.0, 1.0)
    ax2.set_xlabel("\$k / k_F\$")
    ax2.set_ylabel(
        "\$M^{(1)}_\\sigma(k) \\,/\\, \\epsilon_{\\mathrm{TF}} =  \\left(\\epsilon_{k} + \\Sigma_{x}(k)\\right) \\,/\\, \\epsilon_{\\mathrm{TF}} \$",
    )
    # ax2.set_ylabel(
    #     "\$\\epsilon_{\\mathrm{momt.}}(k) \\,/\\, \\epsilon_{\\mathrm{TF}} =  \\left(\\epsilon_{k} + \\Sigma_{x}(k)\\right) \\,/\\, \\epsilon_{\\mathrm{TF}} \$",
    # )
    xloc = 0.8
    yloc = -0.5
    ydiv = -0.5
    # xloc = 0.8
    # yloc = -1.25
    # ydiv = -0.25
    ax2.text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
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

    plt.close("all")
    return
end

main()
