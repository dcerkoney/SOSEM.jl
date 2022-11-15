using ElectronLiquid
using ElectronGas
using Interpolations
using JLD2
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1b/plot_c1b.jl

function main()
    rs = 1.0
    beta = 200.0
    mass2 = 2.0
    # mass2 = 0.1
    solver = :vegasmc
    expand_bare_interactions = false

    neval = 5e8
    max_order = 4
    max_order_plot = 4

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Include unscreened bare result
    plot_bare = true

    plotparams =
        UEG.ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions
        intn_str = "no_bare_"
    end

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "with_ct" : ""
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

    # Load the results from JLD2
    savename =
        "results/data/c1bL_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)"
    settings, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(plotparams))"
        return f[key]
    end

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    # println(settings)
    # println(UEG.paraid(param))
    # println(res_list)
    # println(partitions_list)

    println(settings)
    println(UEG.paraid(param))
    println(partitions)
    println(res)

    # Plot the results
    fig, ax = plt.subplots()

    # Non-dimensionalize bare and RPA+FL non-local moments
    rs_lo = 1.0
    sosem_lo = np.load("results/data/soms_rs=$(rs_lo)_beta_ef=200.0.npz")
    # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    param_lo = Parameter.atomicUnit(0, rs_lo)    # (dimensionless T, rs)
    eTF_lo = param_lo.qTF^2 / (2 * param_lo.me)
    c1b_lo_quad = sosem_lo.get("bare_b") / eTF_lo^2
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    # delta RPA results for class (b) moment
    delta_c1b_rpa = sosem_lo.get("delta_rpa_b_vegas_N=1e+07.npy") / eTF_lo^2
    delta_c1b_rpa_err = sosem_lo.get("delta_rpa_b_err_vegas_N=1e+07.npy") / eTF_lo^2
    # delta RPA+FL results for class (b) moment
    delta_c1b_rpa_fl = sosem_lo.get("delta_rpa+fl_b_vegas_N=1e+07.npy") / eTF_lo^2
    delta_c1b_rpa_fl_err = sosem_lo.get("delta_rpa+fl_b_err_vegas_N=1e+07.npy") / eTF_lo^2
    if plot_bare
        ax.plot(k_kf_grid_quad, c1b_lo_quad, "C0"; linestyle="-", label="\$N=2\$ (quad)")
        ax.plot(k_kf_grid, delta_c1b_rpa, "k"; linestyle="--", label="RPA (vegas)")
        ax.fill_between(
            k_kf_grid,
            (delta_c1b_rpa - delta_c1b_rpa_err),
            (delta_c1b_rpa + delta_c1b_rpa_err);
            color="k",
            alpha=0.3,
        )
        ax.plot(k_kf_grid, delta_c1b_rpa_fl, "k"; label="RPA\$+\$FL (vegas)")
        ax.fill_between(
            k_kf_grid,
            (delta_c1b_rpa_fl - delta_c1b_rpa_fl_err),
            (delta_c1b_rpa_fl + delta_c1b_rpa_fl_err);
            color="k",
            alpha=0.3,
        )
    end

    for o in eachindex(partitions)
        if sum(partitions[o]) > max_order_plot
            continue
        end
        # Get means and error bars from the result for this partition
        local means, stdevs
        if res.config.N == 1
            # res gets automatically flattened for a single-partition measurement
            means, stdevs = res.mean, res.stdev
        else
            means, stdevs = res.mean[o], res.stdev[o]
        end
        # Get means and error bars from the result up to this order
        # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
        #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
        means = 2 * means
        stdevs = 2 * stdevs
        # Data gets noisy above 1st Green's function counterterm order
        # marker =
        #     (partitions[o][2] > 1 || (partitions[o][1] > 3 && partitions[o][2] > 0)) ?
        #     "o-" : "-"
        marker = "-"
        ax.plot(
            k_kf_grid,
            means,
            marker;
            markersize=2,
            color="C$(o - 1)",
            label="\$C^{(1)nl}_{n=3} = \\delta C^{(1b)}_{3}\$ ($solver)",
            # label="\$\\mathcal{P}=$(partitions[o])\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            color="C$(o - 1)",
            alpha=0.4,
        )
    end
    ax.legend(; loc="lower right")
    ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    # ax.set_ylim(-0.1, 0.0025)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        # "\$C^{(1b)}_{\\mathcal{P}}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
        "\$C^{(1)nl}_{n}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    xloc = 1.75
    yloc = -0.1
    ydiv = -0.02
    ax.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = 200,\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = 2\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
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
        "results/c1b/c1b_n=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_" *
        "$(ct_string).pdf",
    )
    plt.close("all")
    return
end

main()
