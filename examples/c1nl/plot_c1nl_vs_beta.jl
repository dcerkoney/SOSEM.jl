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
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    rs = 1.0
    betas = [20.0, 40.0]
    mass2 = 2.0
    solver = :vegasmc
    expand_bare_interactions = false

    # neval_c1b0 = [3e10, 5e10]
    # neval_c1b = [1e10, 5e10]
    # neval_c1c = [1e10, 5e10]
    # neval_c1d = [1e10, 5e10]
    nevals = [[3e10, 1e10, 1e10, 1e10], [5e10, 5e10, 5e10, 5e10]]
    neval = 5e10
    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order_plot = 3
    # max_order_plot = 3
    max_order_plot = 4

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

    # Non-dimensionalize bare and RPA+FL non-local moments
    rs_lo = 1.0
    sosem_lo = np.load("results/data/soms_rs=$(rs_lo)_beta_ef=40.0.npz")
    # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    param_lo = Parameter.atomicUnit(0, rs_lo)    # (dimensionless T, rs)
    eTF_lo = param_lo.qTF^2 / (2 * param_lo.me)

    # Bare and RPA(+FL) results (stored in Hartree a.u.)
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    c1nl_lo =
        (sosem_lo.get("bare_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d")) /
        eTF_lo^2
    c1nl_rpa =
        (sosem_lo.get("rpa_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d")) / eTF_lo^2
    c1nl_rpa_fl =
        (sosem_lo.get("rpa+fl_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d")) /
        eTF_lo^2
    # RPA(+FL) means are error bars
    c1nl_rpa_means, c1nl_rpa_stdevs =
        Measurements.value.(c1nl_rpa), Measurements.uncertainty.(c1nl_rpa)
    c1nl_rpa_fl_means, c1nl_rpa_fl_stdevs =
        Measurements.value.(c1nl_rpa_fl), Measurements.uncertainty.(c1nl_rpa_fl)

    # Filename for new JLD2 format
    filenames = [
        "results/data/rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(mass2)_$(intn_str)$(solver)_with_ct_mu_lambda" for beta in betas
    ]
    linestyles = ["--", "-"]

    # Plot the results for each order ξ and compare to RPA(+FL)
    fig, ax = plt.subplots()
    for (i, N) in enumerate(min_order_plot:max_order_plot)
        for (j, filename) in enumerate(filenames)
            # Load the data for each observable
            f = jldopen("$filename.jld2", "r")
            neval_c1b0, neval_c1b, neval_c1c, neval_c1d = nevals[j]
            param = f["c1d/N=$N/neval=$neval_c1d/param"]
            kgrid = f["c1d/N=$N/neval=$neval_c1d/kgrid"]
            c1nl_N_total =
                f["c1b0/N=$N/neval=$neval_c1b0/meas"] +
                f["c1b/N=$N/neval=$neval_c1b/meas"] +
                f["c1c/N=$N/neval=$neval_c1c/meas"] +
                f["c1d/N=$N/neval=$neval_c1d/meas"]
            close(f)  # close file

            # Get dimensionless k-grid (k / kF)
            k_kf_grid = kgrid / param.kF

            # Get means and error bars from the result up to this order
            c1nl_N_means, c1nl_N_stdevs =
                Measurements.value.(c1nl_N_total), Measurements.uncertainty.(c1nl_N_total)
            @assert length(k_kf_grid) == length(c1nl_N_means) == length(c1nl_N_stdevs)

            if i == j == 1
                ax.plot(
                    k_kf_grid_quad,
                    c1nl_rpa_means,
                    "k";
                    linestyle="--",
                    label="RPA (vegas)",
                )
                ax.fill_between(
                    k_kf_grid_quad,
                    (c1nl_rpa_means - c1nl_rpa_stdevs),
                    (c1nl_rpa_means + c1nl_rpa_stdevs);
                    color="k",
                    alpha=0.3,
                )
                ax.plot(k_kf_grid_quad, c1nl_rpa_fl_means, "k"; label="RPA\$+\$FL (vegas)")
                ax.fill_between(
                    k_kf_grid_quad,
                    (c1nl_rpa_fl_means - c1nl_rpa_fl_stdevs),
                    (c1nl_rpa_fl_means + c1nl_rpa_fl_stdevs);
                    color="r",
                    alpha=0.3,
                )
                ax.plot(
                    k_kf_grid_quad,
                    c1nl_lo,
                    "C0";
                    linestyle="-",
                    label="\$N=2,\\, T = 0\$ (quad)",
                )
            end
            # ax.plot(k_kf_grid, c1nl_N_means, "o-"; color="C$i", markersize=2, label="\$N=$N\$ ($solver)")
            ax.plot(
                k_kf_grid,
                c1nl_N_means,
                "C$i";
                linestyle=linestyles[j],
                label="\$N=$N,\\, \\beta = $(betas[j])\$ ($solver)",
            )
            ax.fill_between(
                k_kf_grid,
                (c1nl_N_means - c1nl_N_stdevs),
                (c1nl_N_means + c1nl_N_stdevs);
                color="C$i",
                alpha=0.3,
            )
            ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
        end
    end

    # # Plot speculative results
    # f = jldopen("$filename.jld2", "r")
    # param = f["c1d/N=3/neval=$neval_c1d/param"]
    # kgrid = f["c1d/N=3/neval=$neval_c1d/kgrid"]
    # # Get dimensionless k-grid (k / kF)
    # k_kf_grid = kgrid / param.kF
    # # N=3
    # c1nl_3_conj =
    #     f["c1b0/N=2/neval=$neval_c1b0/meas"] +
    #     f["c1b/N=3/neval=$neval_c1b/meas"] +
    #     f["c1c/N=2/neval=$neval_c1c/meas"] +
    #     f["c1d/N=2/neval=$neval_c1d/meas"]
    # c1nl_3_means, c1nl_3_stdevs =
    #     Measurements.value.(c1nl_3_conj), Measurements.uncertainty.(c1nl_3_conj)
    # ax.plot(k_kf_grid, c1nl_3_means, "C1"; label="\$N=3\$ \$(\\mathrm{$solver},\\, T \\approx 0)\$")
    # ax.fill_between(
    #     k_kf_grid,
    #     (c1nl_3_means - c1nl_3_stdevs),
    #     (c1nl_3_means + c1nl_3_stdevs);
    #     color="C1",
    #     alpha=0.3,
    # )
    # # N=4
    # c1nl_4_conj =
    #     f["c1b0/N=2/neval=$neval_c1b0/meas"] +
    #     f["c1b/N=4/neval=$neval_c1b/meas"] +
    #     f["c1c/N=2/neval=$neval_c1c/meas"] +
    #     f["c1d/N=2/neval=$neval_c1d/meas"]
    # c1nl_4_means, c1nl_4_stdevs =
    #     Measurements.value.(c1nl_4_conj), Measurements.uncertainty.(c1nl_4_conj)
    # ax.plot(k_kf_grid, c1nl_4_means, "C2"; label="\$N=4\$ \$(\\mathrm{$solver},\\, T \\approx 0)\$")
    # ax.fill_between(
    #     k_kf_grid,
    #     (c1nl_4_means - c1nl_4_stdevs),
    #     (c1nl_4_means + c1nl_4_stdevs);
    #     color="C2",
    #     alpha=0.3,
    # )
    # close(f)  # close file

    # ax.set_xlim(minimum(k_kf_grid), 2.0)
    ax.set_ylim(; top=-0.195)
    ax.legend(; loc="best")
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$C^{(1)nl}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$")
    xloc = 0.25
    yloc = -0.25
    ydiv = -0.05
    ax.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + ydiv,
        "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=12,
    )
    # plt.title("Using fixed bare Coulomb interactions \$V_1\$, \$V_2\$")
    # plt.title(
    #     "Using re-expanded Coulomb interactions \$V_1[V_\\lambda]\$, \$V_2[V_\\lambda]\$",
    # )
    plt.tight_layout()
    fig.savefig(
        # "results/c1nl/c1nl_N=4_rs=$(rs)_" *
        "results/c1nl/c1nl_N=$(max_order_plot)_rs=$(rs)_" *
        "beta_ef=$(betas)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_total.pdf",
        # "neval=$(neval)_$(intn_str)$(solver)_total_conjectured.pdf",
    )
    plt.close("all")
    return
end

main()
