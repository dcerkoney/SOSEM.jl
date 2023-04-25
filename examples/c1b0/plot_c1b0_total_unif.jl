using CodecZlib
using ElectronLiquid
using ElectronGas
using FeynmanDiagram
using Interpolations
using JLD2
using MCIntegration
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1b0/plot_c1b0_total.jl

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    rs = 1.0
    beta = 40.0
    mass2 = 2.0
    solver = :vegasmc
    expand_bare_interactions = false

    neval = 1e10
    n_min = 2  # True minimal loop order for this observable
    min_order = 3
    max_order = 4
    min_order_plot = 2
    max_order_plot = 4
    @assert max_order ≥ 3

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Manually perform chemical potential renormalization
    renorm_mu_lo_ex = false  # at lowest order
    # renorm_mu_nlo_ex = false  # at next-lowest order

    # Save total results
    save = true

    plotparam =
        UEG.ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions
        intn_str = "no_bare_"
    end

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "with_ct" : ""
    renorm_string = (renorm_mu && renorm_mu_lo_ex) ? "_lo_mu_manual" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # colors = ["orchid", "cornflowerblue", "turquoise", "chartreuse", "greenyellow"]
    # markers = ["-", "-", "-", "-", "-"]

    # Load the results from JLD2 (and μ data from csv, if applicable)
    savename =
        "results/data/c1bL0_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)"
    settings, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(plotparam))"
        return f[key]
    end

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])

    # Non-dimensionalize bare and RPA+FL non-local moments
    rs_lo = rs
    sosem_lo = np.load("results/data/soms_rs=$(rs_lo)_beta_ef=40.0.npz")
    # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    param_lo = Parameter.atomicUnit(0, rs_lo)    # (dimensionless T, rs)
    eTF_lo = param_lo.qTF^2 / (2 * param_lo.me)

    # Get vegas k-mesh for RPA(+FL)
    k_kf_grid_vegas = np.load("results/kgrids/kgrid_vegas_dimless_n=77_small.npy")
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    # Bare result (stored in Hartree a.u.)
    c1b_bare_quad = sosem_lo.get("bare_b") / eTF_lo^2

    # Interpolate bare results and downsample to coarse k_kf_grid_vegas
    c1b_bare_interp  = linear_interpolation(k_kf_grid_quad, c1b_bare_quad; extrapolation_bc=Line())
    c1b2_exact_vegas = c1b_bare_interp(k_kf_grid_vegas)  # Bare result on vegas grid
    c1b2_exact       = c1b2_exact_vegas        # Bare result on composite grid
    # c1b2_exact       = c1b_bare_interp(k_kf_grid)        # Bare result on composite grid

    # RPA(+FL) corrections to LO for class (b) moment
    delta_c1b_rpa = sosem_lo.get("delta_rpa_b_vegas_N=1e+07.npy") / eTF_lo^2
    delta_c1b_rpa_err = sosem_lo.get("delta_rpa_b_err_vegas_N=1e+07.npy") / eTF_lo^2
    delta_c1b_rpa_fl = sosem_lo.get("delta_rpa+fl_b_vegas_N=1e+07.npy") / eTF_lo^2
    delta_c1b_rpa_fl_err = sosem_lo.get("delta_rpa+fl_b_err_vegas_N=1e+07.npy") / eTF_lo^2

    # Total RPA(+FL) results for class (b) moment
    c1b_rpa = delta_c1b_rpa + c1b2_exact_vegas
    c1b_rpa_err = delta_c1b_rpa_err
    c1b_rpa_fl = delta_c1b_rpa_fl + c1b2_exact_vegas
    c1b_rpa_fl_err = delta_c1b_rpa_fl_err

    if min_order_plot == 2
        # Set bare result manually using exact data to avoid systematic error in (2,0,0) calculation
        # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
        #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
        c1b2L_exact = c1b2_exact / 2
        merged_data[(2, 0)] = measurement.([c1b2L_exact[1]], 0.0)  # treat quadrature data as numerically exact
    end

    # Get total data
    if renorm_mu
        if renorm_mu_lo_ex && max_order_plot == 4
            δμ1 = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
            # C⁽¹ᵇ⁾₄ = C⁽¹ᵇ⁾_{4,0} + δμ₁ C⁽¹ᵇ⁾_{3,1}
            c1b3L0 = merged_data[(3, 0)]
            c1b4L0 = merged_data[(4, 0)] + δμ1 * merged_data[(3, 1)]
            c1bL0 = SortedDict(3 => c1b3L0, 4 => c1b4L0)
            if min_order_plot == 2
                c1bL0[2] = c1b2_exact
            end
        else
            # Reexpand merged data in powers of μ
            z, μ = UEG_MC.load_z_mu(param)
            δz, δμ = CounterTerm.sigmaCT(max_order - n_min, μ, z; verbose=1)
            println("Computed δμ: ", δμ)
            c1bL0 = UEG_MC.chemicalpotential_renormalization_sosem(
                merged_data,
                δμ;
                min_order=min(min_order, min_order_plot),
                max_order=max(max_order, max_order_plot),
            )
            # Test manual renormalization with exact lowest-order chemical potential
            if !renorm_mu_lo_ex && max_order >= 4
                # NOTE: For this observable, there is no second-order
                δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
                # C⁽¹ᵇ⁾₄ = 2(C⁽¹ᵇ⁾ᴸ_{4,0} + δμ₁ C⁽¹ᵇ⁾ᴸ_{3,1})
                c1b04_manual =
                    2 * (
                        merged_data[(3, 0)] +
                        merged_data[(4, 0)] +
                        δμ1_exact * merged_data[(3, 1)]
                    )
                c1b4L0 = 2 * (c1bL0[3] + c1bL0[4])
                stdscores = stdscore.(c1b4L0, c1b04_manual)
                worst_score = argmax(abs, stdscores)
                println("Exact δμ₁: ", δμ1_exact)
                println("Computed δμ₁: ", δμ[1])
                println(
                    "Worst standard score for total result to 4th " *
                    "order (auto vs exact+manual): $worst_score",
                )
            end
        end
    else
        c1bL0 = merged_data
    end

    # Aggregate the full results for C⁽¹ᶜ⁾ up to order N
    if renorm_mu_lo_ex
        c1bL0_total = Dict(3 => c1b3L0, 4 => c1b3L0 + c1b4L0)
        if min_order_plot == 2
            c1bL0_total[2] = c1b02_exact / 2
            c1bL0_total[3] += c1b02_exact / 2
            c1bL0_total[4] += c1b02_exact / 2
        end
    else
        c1bL0_total = UEG_MC.aggregate_orders(c1bL0)
    end

    # partitions = collect(Iterators.flatten(partitions_list))

    println(settings)
    println(UEG.paraid(param))
    # println(res_list)
    # println(partitions_list)
    println(res)
    println(partitions)

    # Plot the results
    fig, ax = plt.subplots()

    if min_order_plot == 2
        ax.plot(k_kf_grid_vegas, c1b_rpa, "k"; linestyle="--", label="RPA (vegas)")
        ax.fill_between(
            k_kf_grid_vegas,
            (c1b_rpa - c1b_rpa_err),
            (c1b_rpa + c1b_rpa_err);
            color="k",
            alpha=0.3,
        )
        ax.plot(k_kf_grid_vegas, c1b_rpa_fl, "k"; label="RPA\$+\$FL (vegas)")
        ax.fill_between(
            k_kf_grid_vegas,
            (c1b_rpa_fl - c1b_rpa_fl_err),
            (c1b_rpa_fl + c1b_rpa_fl_err);
            color="k",
            alpha=0.3,
        )
        ax.plot(
            k_kf_grid_vegas,
            c1b2_exact_vegas,
            "C0";
            linestyle="-",
            label="\$N=2\$ (quad)",
        )
    end

    if save
        savename =
            "results/data/rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_$(intn_str)$(solver)_$(ct_string)"
        f = jldopen("$savename.jld2", "a+"; compress=true)
        # NOTE: no bare result for c1b observable (accounted for in c1b0)
        for N in min_order_plot:max_order
            if haskey(f, "c1b0") &&
               haskey(f["c1b0"], "N=$N") &&
               haskey(f["c1b0/N=$N"], "neval=$(neval)")
                @warn("replacing existing data for N=$N, neval=$(neval)")
                delete!(f["c1b0/N=$N"], "neval=$(neval)")
            end
            # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
            #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
            f["c1b0/N=$N/neval=$neval/meas"] = 2 * c1bL0_total[N]
            f["c1b0/N=$N/neval=$neval/settings"] = settings
            f["c1b0/N=$N/neval=$neval/param"] = param
            f["c1b0/N=$N/neval=$neval/kgrid"] = kgrid
        end
    end

    # Plot for each aggregate order
    for (i, N) in enumerate(min_order:max_order_plot)
        # if N == 2
        #     continue
        # end
        # Get means and error bars from the result up to this order
        # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
        #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
        means = 2 * Measurements.value.(c1bL0_total[N])
        stdevs = 2 * Measurements.uncertainty.(c1bL0_total[N])
        # Data gets noisy above 3rd loop order
        marker = "o-"
        # marker = "-"
        # marker = N > 3 ? "o-" : "-"
        ax.plot(
            k_kf_grid,
            means,
            marker;
            markersize=2,
            color="C$i",
            label="\$N=$N\$ ($solver)",
        )
        ax.fill_between(k_kf_grid, means - stdevs, means + stdevs; color="C$i", alpha=0.4)
        if !renorm_mu_lo_ex && max_order <= 3 && N == 3
            ax.plot(
                k_kf_grid,
                Measurements.value.(c1b03_manual);
                color="r",
                linestyle="--",
                label="\$N=3\$ (manual, vegasmc)",
            )
        end
    end
    ax.legend(; loc="lower right")
    # ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    ax.set_xlim(minimum(k_kf_grid) - 0.1, 2.0)
    ax.set_ylim(-1.2, -0.675)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$C^{(1b0)}_{N}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    # xloc = 1.75
    # yloc = -0.5
    # ydiv = -0.095
    xloc = 0.125
    yloc = -1.05
    ydiv = -0.05
    ax.text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + 2 * ydiv,
        "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=12,
    )
    plt.title("Using fixed bare Coulomb interactions \$V_1\$, \$V_2\$")
    # plt.title(
    #     "Using re-expanded Coulomb interactions \$V_1[V_\\lambda]\$, \$V_2[V_\\lambda]\$",
    # )
    plt.tight_layout()
    fig.savefig(
        "results/c1b0/c1b0_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)" *
        "$(renorm_string)_total.pdf",
    )
    plt.close("all")
    return
end

main()
