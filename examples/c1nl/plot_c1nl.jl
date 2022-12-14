using ElectronLiquid
using ElectronGas
using Interpolations
using JLD2
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1d/plot_c1d_total.jl

function main()
    rs = 1.0
    beta = 20.0
    mass2 = 2.0
    solver = :vegasmc
    expand_bare_interactions = false

    neval_c1b0 = 3e10
    neval_c1b = 1e10
    neval_c1c = 1e10
    neval_c1d = 1e10
    neval = max(neval_c1b0, neval_c1b, neval_c1c, neval_c1d)

    # Plot total results to order ξ 
    plot_order = 4

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions
        intn_str = "no_bare_"
    end

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # colors = ["orchid", "cornflowerblue", "turquoise", "chartreuse", "greenyellow"]
    # markers = ["-", "-", "-", "-", "-"]

    # Filename for new JLD2 format
    filename =
        "results/data/rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(mass2)_$(intn_str)$(solver)_with_ct_mu_lambda"

    # Plot each observable at order N = plot_order
    fig, ax = plt.subplots()
    nevals = [nothing, neval_c1c, neval_c1d]
    obsnames = ["c1b_total", "c1c", "c1d"]
    # Prefixes for split \delta C^{(1b0)} run
    # obsprefixes = [["\\delta C^{(1b0)}_{$plot_order}", "C^{(1b0)}_2 + C^{(1b)}_{$plot_order}"], "C^{(1c)}_{$plot_order}", "C^{(1d)}_{$plot_order}"]
    obsprefixes = [
        ["C^{(1b0)}_{$plot_order}", "C^{(1b)}_{$plot_order}"],
        "C^{(1c)}_{$plot_order}",
        "C^{(1d)}_{$plot_order}",
    ]
    next_color = 0
    for (i, obsname) in enumerate(obsnames)
        # Load the data for this observable
        f = jldopen("$filename.jld2", "r")
        if obsname == "c1b_total"
            meas0 = f["c1b0/N=$plot_order/neval=$(neval_c1b0)/meas"]
            if plot_order == 2
                meas = zero(meas0)
            else
                meas = f["c1b/N=$plot_order/neval=$(neval_c1b)/meas"]
            end
            meas_rpa = f["c1b/RPA/neval=$(1e7)/meas"]
            meas_rpa_fl = f["c1b/RPA+FL/neval=$(1e7)/meas"]
            param = f["c1b0/N=$plot_order/neval=$(neval_c1b0)/param"]
            kgrid = f["c1b0/N=$plot_order/neval=$(neval_c1b0)/kgrid"]
        else
            meas = f["$obsname/N=$plot_order/neval=$(nevals[i])/meas"]
            param = f["$obsname/N=$plot_order/neval=$(nevals[i])/param"]
            kgrid = f["$obsname/N=$plot_order/neval=$(nevals[i])/kgrid"]
        end
        close(f)  # close file

        # Get dimensionless k-grid (k / kF)
        k_kf_grid = kgrid / param.kF

        # Get means and error bars from the result up to this order
        means, stdevs = Measurements.value.(meas), Measurements.uncertainty.(meas)
        @assert length(k_kf_grid) == length(means) == length(stdevs)

        # Plot RPA(+FL) corrections to the class (b) observable
        if obsname == "c1b_total"
            c1b_rpa = Measurements.value.(meas_rpa)
            c1b_rpa_err = Measurements.uncertainty.(meas_rpa)
            c1b_rpa_fl = Measurements.value.(meas_rpa_fl)
            c1b_rpa_fl_err = Measurements.uncertainty.(meas_rpa_fl)
            ax.plot(
                k_kf_grid,
                c1b_rpa,
                "k";
                linestyle="--",
                label="\$C^{(1b)}_{\\mathrm{RPA}}\$ (vegas)",
            )
            ax.fill_between(
                k_kf_grid,
                (c1b_rpa - c1b_rpa_err),
                (c1b_rpa + c1b_rpa_err);
                color="k",
                alpha=0.3,
            )
            ax.plot(
                k_kf_grid,
                c1b_rpa_fl,
                "k";
                label="\$C^{(1b)}_{\\mathrm{RPA}+\\mathrm{FL}}\$ (vegas)",
            )
            ax.fill_between(
                k_kf_grid,
                (c1b_rpa_fl - c1b_rpa_fl_err),
                (c1b_rpa_fl + c1b_rpa_fl_err);
                color="k",
                alpha=0.3,
            )
        end

        # Data gets noisy above 3rd loop order
        # marker = N > 3 ? "o-" : "-"
        marker = "-"
        # marker = "o-"
        if obsname == "c1b_total"
            means0, stdevs0 = Measurements.value.(meas0), Measurements.uncertainty.(meas0)
            means_list = [means0, means]
            stdevs_list = [stdevs0, stdevs]
            for (j, obsprefix) in enumerate(obsprefixes[i])
                if plot_order == 2 && j == 2
                    continue
                end
                ax.plot(
                    k_kf_grid,
                    means_list[j],
                    marker;
                    markersize=2,
                    color="C$next_color",
                    label="\$$obsprefix\$ ($solver)",
                )
                ax.fill_between(
                    k_kf_grid,
                    means_list[j] - stdevs_list[j],
                    means_list[j] + stdevs_list[j];
                    color="C$next_color",
                    alpha=0.4,
                )
                next_color += 1
            end
        else
            ax.plot(
                k_kf_grid,
                means,
                marker;
                markersize=2,
                color="C$next_color",
                label="\$$(obsprefixes[i])\$ ($solver)",
            )
            ax.fill_between(
                k_kf_grid,
                means - stdevs,
                means + stdevs;
                color="C$next_color",
                alpha=0.4,
            )
            next_color += 1
        end
        ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
        # ax.set_xlim(minimum(k_kf_grid), 2.0)
    end
    ax.set_ylim(-2.1, 2.1)
    ax.legend(; loc="best")
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$C^{(1\\cdot)}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    # xloc = 0.5
    # yloc = -0.075
    # ydiv = -0.009
    xloc = 1.7
    yloc = -1.175
    ydiv = -0.3
    ax.text(
        xloc,
        yloc,
        "\$N = $plot_order,\\, r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
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
    # plt.title("Using fixed bare Coulomb interactions \$V_1\$, \$V_2\$")
    # plt.title(
    #     "Using re-expanded Coulomb interactions \$V_1[V_\\lambda]\$, \$V_2[V_\\lambda]\$",
    # )
    plt.tight_layout()
    fig.savefig(
        "results/c1nl/c1nl_N=$(plot_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver).pdf",
    )
    plt.close("all")
    return
end

main()
