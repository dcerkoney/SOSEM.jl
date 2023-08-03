using CodecZlib
using ElectronGas
using ElectronLiquid
using Interpolations
using JLD2
using LsqFit
using Measurements
using Polynomials
using PyCall
using PyPlot
using SOSEM

# For style "science"
@pyimport scienceplots

# For saving/loading numpy data
@pyimport numpy as np
@pyimport scipy.interpolate as interp

# @pyimport matplotlib.pyplot as plt
# @pyimport mpl_toolkits.axes_grid1.inset_locator as il

# Dimensionless expansion parameter α for the UEG (powers of αrₛ)
const alpha = (4 / 9π)^(1 / 3)

# Vibrant qualitative colour scheme from https://personal.sron.nl/~pault/
const cdict = Dict([
    "orange" => "#EE7733",
    "blue" => "#0077BB",
    "cyan" => "#33BBEE",
    "magenta" => "#EE3377",
    "red" => "#CC3311",
    "teal" => "#009988",
    "grey" => "#BBBBBB",
]);

function spline(x, y, e)
    # generate knots with spline without constraints
    w = 1.0 ./ e
    spl = interp.UnivariateSpline(x, y; w=w, k=3)
    __x = collect(LinRange(x[1], x[end], 1000))
    # __x = collect(LinRange(x[1], x[end], 100))
    yfit = spl(__x)
    return __x, yfit
end

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
    #        (para.e0^2 * para.me / (2pi * para.kF)) *
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

    # Setup plot styles
    style = PyPlot.matplotlib."style"
    style.use(["science", "std-colors"])
    color = [
        "k",
        cdict["orange"],
        cdict["blue"],
        cdict["cyan"],
        cdict["magenta"],
        cdict["red"],
        # cdict["teal"],
    ]
    # color = [cdict["blue"], cdict["orange"], "green", cdict["red"], "black"]
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")

    # Use LaTex fonts for plots
    rcParams["font.size"] = 16
    rcParams["mathtext.fontset"] = "cm"
    # rcParams["font.family"] = "Times New Roman"

    beta = 40.0
    solver = :vegasmc

    # rs = 1 runs: N = [[1, 2, 3], [4], [5]]
    rs = 1.0
    mass2 = 1.0

    # Number of evals below and above kF
    neval123 = rs == 1 ? 1e10 : 5e10
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
    save = false
    # save = true

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

    if max_order >= 4
        max_together = 3
    else
        max_together = max_order
    end

    # UEG parameters for MC integration
    para = ParaMC(; order=3, rs=rs, beta=beta, mass2=mass2, isDynamic=false)
    para4 = ParaMC(; order=4, rs=rs, beta=beta, mass2=mass2, isDynamic=false)
    para5 = ParaMC(; order=5, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    # Load raw data
    local htf3, htf4, htf5
    savename =
        "results/data/exchange/sigma_x_n=$(max_together)_rs=$(rs)_" *
        # "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval4)_$(solver)$(ct_string)"
        "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval123)_$(solver)$(ct_string)"
    orders, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        htf3 = f["has_taylor_factors"]
        key = "$(UEG.short(para))"
        return f[key]
    end
    if max_order >= 4
        savename =
            "results/data/exchange/sigma_x_n=4_rs=$(rs)_" *
            "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval4)_$(solver)$(ct_string)"
        orders4, kgrid4, partitions4, res4 = jldopen("$savename.jld2", "a+") do f
            htf4 = f["has_taylor_factors"]
            key = "$(UEG.short(para4))"
            return f[key]
        end
    end
    if max_order >= 5
        savename =
            "results/data/exchange/sigma_x_n=5_rs=$(rs)_" *
            "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval5)_$(solver)$(ct_string)"
        orders5, kgrid5, partitions5, res5 = jldopen("$savename.jld2", "a+") do f
            htf5 = f["has_taylor_factors"]
            key = "$(UEG.short(para5))"
            return f[key]
        end
    end

    # Get dimensionless k-grid (k / kF) and index corresponding to the Fermi energy
    k_kf_grid = kgrid / para.kF
    if max_order >= 4
        k_kf_grid4 = kgrid4 / para.kF
    end
    if max_order >= 5
        k_kf_grid5 = kgrid5 / para.kF
    end
    ikF = findfirst(x -> x == 1.0, k_kf_grid)

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    if htf3 == false
        for (k, v) in data
            data[k] = v / (factorial(k[2]) * factorial(k[3]))
        end
    end
    # Add 4th order results to data dict
    if max_order >= 4
        data4 = UEG_MC.restodict(res4, partitions4)
        if htf4 == false
            for (k, v) in data4
                data4[k] = v / (factorial(k[2]) * factorial(k[3]))
            end
        end
        merge!(data, data4)
    end
    # Add 5th order results to data dict
    if max_order >= 5
        data5 = UEG_MC.restodict(res5, partitions5)
        if htf5 == false
            for (k, v) in data5
                data5[k] = v / (factorial(k[2]) * factorial(k[3]))
            end
        end
        merge!(data, data5)
    end

    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])
    # println(merged_data)

    if min_order_plot == 1
        if 1 in orders
            # The nondimensionalized Fock self-energy is the negative Lindhard function
            exact = -UEG_MC.lindhard.(kgrid / para.kF)
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
    ct_filename = "examples/counterterms/data/data_Z.jld2"
    z, μ, has_taylor_factors = UEG_MC.load_z_mu_old(para4; ct_filename=ct_filename)
    # Add Taylor factors to CT data
    if has_taylor_factors == false
        for (p, v) in z
            z[p] = v / (factorial(p[2]) * factorial(p[3]))
        end
        for (p, v) in μ
            μ[p] = v / (factorial(p[2]) * factorial(p[3]))
        end
    end
    _, δμ, _ = CounterTerm.sigmaCT(max_order - n_min, μ, z; verbose=1)
    println("Computed δμ: ", δμ)
    sigma_x =
        UEG_MC.chemicalpotential_renormalization_sigma(merged_data, δμ; max_order=max_order)

    # Test manual renormalization with exact lowest-order chemical potential
    δμ1_exact = UEG_MC.delta_mu1(para)  # = ReΣ₁[λ](kF, 0)
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

    println(UEG.paraid(para))
    println(partitions)
    println(res)

    if save
        savename =
            "results/data/processed/rs=$(para.rs)/rs=$(para.rs)_beta_ef=$(para.beta)_" *
            "lambda=$(para.mass2)_$(intn_str)$(solver)$(ct_string)_archive1"
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
            f["sigma_x/N=$N/neval=$num_eval/para"] = para
            f["sigma_x/N=$N/neval=$num_eval/kgrid"] = kgrid
        end
    end

    # Plot for each aggregate order
    fig1 = figure(; figsize=(6, 4))

    # Σₓ(k) / eTF (dimensionless moment)
    # Compare result to exact non-dimensionalized Fock self-energy (-F(k / kF))
    plot(k_kf_grid, -UEG_MC.lindhard.(k_kf_grid), "k"; label="\$N=1\$")
    # plot(k_kf_grid, -UEG_MC.lindhard.(k_kf_grid), "k"; label="\$N=1\$ (exact, \$T=0\$)")
    for (i, N) in enumerate(min_order:max_order_plot)
        N == 1 && continue
        # Get means and error bars from the result up to this order
        means = Measurements.value.(sigma_x_over_eTF_total[N])
        stdevs = Measurements.uncertainty.(sigma_x_over_eTF_total[N])
        _x, _y = spline(k_kf_grid, means, stdevs)
        plot(_x, _y; color=color[i], linestyle="--", zorder=10 * i + 3)
        errorbar(
            k_kf_grid,
            means;
            yerr=stdevs,
            color=color[i],
            capsize=2,
            markersize=2,
            fmt="o",
            markerfacecolor="none",
            label="\$N = $N\$",
            zorder=10 * i + 3,
        )
        # plot(
        #     k_kf_grid,
        #     means,
        #     marker;
        #     markersize=2,
        #     color="C$(i-1)",
        #     label="\$N=$(N)\$ ($solver)",
        # )
        # fill_between(
        #     k_kf_grid,
        #     means - stdevs,
        #     means + stdevs;
        #     color="C$(i-1)",
        #     alpha=0.4,
        # )
    end
    legend(; loc="lower right")
    # xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    xlim(0, 2)
    ylim(nothing, 0.0)
    xlabel("\$k / k_F\$")
    ylabel(
        "\$C^{(0)}_\\sigma(k) \\,/\\, \\epsilon_{\\mathrm{TF}}\$",
        # "\$C^{(0)}_\\sigma(k) \\,/\\, \\epsilon_{\\mathrm{TF}} = \\Sigma_{x}(k) \\,/\\, \\epsilon_{\\mathrm{TF}}\$",
    )
    # ylabel("\$\\Sigma_{x}(k) \\,/\\, \\epsilon_{\\mathrm{TF}}\$")
    xloc = 0.15
    yloc = -0.15
    ydiv = -0.125
    # xloc = 1.5
    # yloc = -0.6
    # ydiv = -0.175
    text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=12,
    )
    text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=12,
    )
    # text(
    #     xloc,
    #     yloc + 2 * ydiv,
    #     "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
    #     fontsize=12,
    # )
    plt.tight_layout()
    savefig(
        "results/fock/sigma_x_N=$(max_order_plot)_rs=$(para.rs)_" *
        "beta_ef=$(para.beta)_lambda=$(para.mass2)_neval=$(neval)_$(solver).pdf",
    )
    plt.close("all")

    # Thomas-Fermi energy
    eTF = para.qTF^2 / (2 * para.me)

    # Bare dispersion in units of the Thomas-Fermi energy (for effective mass related plots)
    Ek = kgrid .^ 2 / (2 * para.me)
    Ek_over_eTF = Ek / eTF

    # Fock self-energy
    sigma_fock_exact = fock_self_energy_exact(kgrid, para)
    # Exact Fock energy at the Fermi surface
    EF_fock = qp_fock_exact(para.kF, para)
    println("ΣF(k = 0) (pred, exact):", sigma_fock_exact[1], " ", -eTF)
    println("EqpF(k = kF) (pred, exact):", EF_fock, " ", para.EF - eTF / 2)
    # @assert sigma_fock_exact[1] ≈ -eTF
    # @assert EF_fock ≈ para.EF - eTF / 2

    # Exact results on dense (quadrature) grids
    kgrid_quad = para.kF * np.linspace(0.0, 3.0; num=600)
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    Ek_quad = kgrid_quad .^ 2 / (2 * para.me)
    Ek_over_eTF_quad = Ek_quad / eTF
    ikF_quad = findall(x -> x == 1.0, k_kf_grid_quad)

    fig12 = figure(; figsize=(6, 4))

    k_kf_dense = np.linspace(0.0, 2.0; num=600)
    axhline(
        1.0 ./ fock_mass_ratio_k0(para);
        label="\$\\left(1 + \\frac{4}{3\\pi}(\\alpha r_s)\\right)^{-1}\$",
        color="gray",
    )
    plot(k_kf_dense, 1.0 ./ fock_mass_ratio_exact(k_kf_dense, para))
    legend(; loc="best")
    xlim(0, 2)
    xlabel("\$k / k_F\$")
    ylabel("\$\\left(m^\\star_F \\left/ m\\right)\\right.(k)\$")
    xloc = 1.25
    yloc = 0.7
    text(
        xloc,
        yloc,
        "\$r_s = $rs,\\, \\beta \\hspace{0.1em} \\epsilon_F = $beta\$";
        fontsize=14,
    )
    plt.tight_layout()
    savefig("results/fock/fock_eff_mass_ratio_exact_rs=$(para.rs)_beta_ef=$(para.beta).pdf")
    plt.close("all")

    # Moment quasiparticle energy
    # Extract effective masses from quadratic fits to data for k ≤ kF
    fig2 = figure(; figsize=(6, 4))

    # First order from exact expressions
    sigma_fock_exact_quad = fock_self_energy_exact(kgrid_quad, para)

    # Fock quasiparticle energy
    E_fock_quad = qp_fock_exact(kgrid_quad, para)
    E_fock_over_eTF_quad = E_fock_quad / eTF

    # No fixed point (zpe is a free parameter)
    @. quasiparticle_model(k, p) = p[1] + k^2 / (2.0 * p[2])

    # Initial parameters for curve fitting procedure
    p0      = [-eTF, 1.0]  # E₀=0 and m=mₑ
    p0_fock = [1.0]        # m=mₑ

    # Gridded data for k ≤ kF
    k_data = kgrid_quad[kgrid_quad .< para.kF]
    E_fock_data = E_fock_quad[kgrid_quad .< para.kF]

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
    low_en_mass_ratio_fock = para.me / meff_fock
    println(
        "Fock low-energy effective mass ratio from quadratic fit: " *
        "(mₑ/m⋆)(k=0) ≈ $low_en_mass_ratio_fock, r2=$r2",
    )
    mass_ratio_fit   = 1 / low_en_mass_ratio_fock
    mass_ratio_exact = 1 / fock_mass_ratio_k0(para)
    rel_error        = abs(mass_ratio_exact - mass_ratio_fit) / mass_ratio_exact
    println("Percent error vs exact low-energy limit: $(rel_error * 100)%")

    axhline(0; linestyle="--", color="gray", linewidth=1)
    # plot(
    #     k_kf_grid_quad,
    #     Ek_quad / eTF,
    #     "k";
    #     linestyle="--",
    #     label="\$\\epsilon_k / \\epsilon_{\\mathrm{TF}} = (k / q_{\\mathrm{TF}})^2\$",
    # )
    plot(
        k_kf_grid_quad,
        -1 .+ (π / (4alpha * para.rs) + 1 / 3) * k_kf_grid_quad .^ 2;
        color="k",
        linestyle="--",
        # label="\$\\epsilon_{\\mathrm{HF}}(k \\rightarrow 0) / \\epsilon_{\\mathrm{TF}} \\sim -1 + \\left(\\frac{\\pi}{4\\alpha r_s} + \\frac{1}{3}\\right) \\left( \\frac{k}{k_F} \\right)^2\$",
    )
    plot(
        kgrid_quad / para.kF,
        E_fock_quad / eTF,
        "k";
        label="\$N=1\$",
        # label="\$N=1\$ (exact, \$T=0\$)",
    )
    # plot(
    #     kgrid_quad / para.kF,
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
        N == 1 && continue

        # Eqp = ϵ(k) + Σₓ(k)
        Eqp_over_eTF = Ek_over_eTF .+ sigma_x_over_eTF_total[N]

        # Get means and error bars
        means_qp = Measurements.value.(Eqp_over_eTF)
        stdevs_qp = Measurements.uncertainty.(Eqp_over_eTF)

        # Gridded data for k < kF
        k_data = kgrid[kgrid .< para.kF]
        Eqp_data = means_qp[kgrid .< para.kF] * eTF
        # Eqp_data = e_test[i] .+ k_data .^ 2 / (2 * m_test[i])

        # Least-squares quasiparticle fit
        fit_N = curve_fit(quasiparticle_model, k_data, Eqp_data, p0)
        qp_fit_N(k) = quasiparticle_model(k, fit_N.param)

        # Coefficients of determination (r²)
        r2 = rsquared(k_data, Eqp_data, qp_fit_N(k_data), fit_N)

        # Low-energy effective mass ratio (mₑ/m⋆)(k≈0) from quasiparticle fit
        meff_N = fit_N.param[2]
        low_en_mass_ratio_N = para.me / meff_N
        println(
            "(N=$N) Low-energy effective mass ratio from quadratic fit: " *
            "(mₑ/m⋆)(k=0) ≈ $low_en_mass_ratio_N, r2=$r2",
        )
        if N == max_order_plot
            text(
                0.8,
                -1.7,
                "\$m^\\star / m \\approx $(round(meff_N / para.me; digits=3))\$";
                fontsize=12,
            )
        end
        # _x, _y = spline(k_kf_grid, means, stdevs)
        # plot(_x, _y; color=color[i], linestyle="--", zorder=10 * i + 3)
        errorbar(
            k_kf_grid,
            means_qp;
            yerr=stdevs_qp,
            color=color[i],
            capsize=2,
            markersize=2,
            fmt="o",
            markerfacecolor="none",
            label="\$N = $N\$",
            zorder=10 * i + 3,
        )
        # plot(
        #     k_kf_grid,
        #     means_qp,
        #     marker;
        #     markersize=2,
        #     color=color[i],
        #     label="\$N=$N\$",
        #     # label="\$N=$(N)\$ ($solver)",
        # )
        # fill_between(
        #     k_kf_grid,
        #     means_qp - stdevs_qp,
        #     means_qp + stdevs_qp;
        #     color=color[i],
        #     alpha=0.4,
        # )
        plot(
            kgrid_quad / para.kF,
            qp_fit_N(kgrid_quad) / eTF;
            color=color[i],
            linestyle="--",
            # label="\$N=$N\$ (quasiparticle fit)",
        )
        ic += 1
    end
    legend(; loc="upper left")
    # xlim(minimum(k_kf_grid), 1)
    xlim(minimum(k_kf_grid), 1.5)
    ylim(-2.2, 3.2)
    # ylim(-2.0, 1.0)
    xlabel("\$k / k_F\$")
    ylabel(
        "\$M^{(1)}_\\sigma(k) \\,/\\, \\epsilon_{\\mathrm{TF}}\$",
        # "\$M^{(1)}_\\sigma(k) \\,/\\, \\epsilon_{\\mathrm{TF}} =  \\left(\\epsilon_{k} + \\Sigma_{x}(k)\\right) \\,/\\, \\epsilon_{\\mathrm{TF}} \$",
    )
    # ylabel(
    #     "\$\\epsilon_{\\mathrm{momt.}}(k) \\,/\\, \\epsilon_{\\mathrm{TF}} =  \\left(\\epsilon_{k} + \\Sigma_{x}(k)\\right) \\,/\\, \\epsilon_{\\mathrm{TF}} \$",
    # )
    xloc = 0.8
    yloc = -0.7
    ydiv = -0.5
    # xloc = 0.8
    # yloc = -1.25
    # ydiv = -0.25
    text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=12,
    )
    text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=12,
    )
    # text(
    #     xloc,
    #     yloc + 2 * ydiv,
    #     "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
    #     fontsize=12,
    # )
    plt.tight_layout()
    savefig(
        "results/fock/moment_qp_energy_N=$(max_order_plot)_rs=$(para.rs)_" *
        "beta_ef=$(para.beta)_lambda=$(para.mass2)_neval=$(neval)_$(solver).pdf",
    )
    plt.close("all")

    return
end

main()
