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
    beta = 40.0
    neval = 1e10
    lambdas = [0.5, 1.0, 1.5, 2.0, 3.0]
    # lambdas = [1.0, 3.0]
    solver = :vegasmc
    expand_bare_interactions = false

    plot_rpa = false

    min_order = 3
    max_order = 4

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order_plot = 2
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

    # Bare and RPA(+FL) uniform results (stored in Hartree a.u.)
    c1nl_lo =
        (sosem_lo.get("bare_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d"))[1] /
        eTF_lo^2
    c1nl_rpa =
        (sosem_lo.get("rpa_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d"))[1] /
        eTF_lo^2
    c1nl_rpa_fl =
        (sosem_lo.get("rpa+fl_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d"))[1] /
        eTF_lo^2

    # RPA(+FL) mean are error bar
    c1nl_rpa_mean, c1nl_rpa_stdev =
        Measurements.value(c1nl_rpa), Measurements.uncertainty(c1nl_rpa)
    c1nl_rpa_fl_mean, c1nl_rpa_fl_stdev =
        Measurements.value(c1nl_rpa_fl), Measurements.uncertainty(c1nl_rpa_fl)

    # Reshape to lambda plot
    c1nl_los = c1nl_lo * one.(lambdas)
    c1nl_rpa_means = c1nl_rpa_mean * one.(lambdas)
    c1nl_rpa_stdevs = c1nl_rpa_stdev * one.(lambdas)
    c1nl_rpa_fl_means = c1nl_rpa_fl_mean * one.(lambdas)
    c1nl_rpa_fl_stdevs = c1nl_rpa_fl_stdev * one.(lambdas)

    # Filename for new JLD2 format
    filenames = [
        "results/data/rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(lambda)_$(intn_str)$(solver)_with_ct_mu_lambda" for lambda in lambdas
    ]

    # Plot the results for each order ξ vs lambda and compare to RPA(+FL)
    fig, ax = plt.subplots()
    if min_order_plot == 2
        if plot_rpa
            ax.plot(
                lambdas,
                c1nl_rpa_means,
                "o--";
                color="k",
                markersize=3,
                label="RPA (vegas)",
            )
            ax.fill_between(
                lambdas,
                (c1nl_rpa_means - c1nl_rpa_stdevs),
                (c1nl_rpa_means + c1nl_rpa_stdevs);
                color="k",
                alpha=0.3,
            )
            ax.plot(
                lambdas,
                c1nl_rpa_fl_means,
                "o-";
                color="k",
                markersize=3,
                label="RPA\$+\$FL (vegas)",
            )
            ax.fill_between(
                lambdas,
                (c1nl_rpa_fl_means - c1nl_rpa_fl_stdevs),
                (c1nl_rpa_fl_means + c1nl_rpa_fl_stdevs);
                color="r",
                alpha=0.3,
            )
        end
        # ax.plot(
        #     lambdas,
        #     c1nl_los,
        #     "o-";
        #     color="C0",
        #     markersize=3,
        #     label="\$N=2\$ (quad, \$T = 0\$)",
        # )
        ax.plot(
            lambdas,
            -0.5 * one.(lambdas),
            "-";
            color="C0",
            markersize=3,
            label="\$N=2\$ (exact, \$T = 0\$)",
        )
    end
    for (i, N) in enumerate(min_order:max_order_plot)
        c1nl_N_means = []
        c1nl_N_stdevs = []
        for (j, filename) in enumerate(filenames)
            if j == 4
                println("\nN = $N, lambda = $(lambdas[j]):")
            end
            # Load the data for each observable
            f = jldopen("$filename.jld2", "r")
            this_kgrid = f["c1d/N=$(N)_unif/neval=$neval/kgrid"]
            @assert this_kgrid == [0.0]

            r1 = f["c1b0/N=$(N)_unif/neval=$neval/meas"]
            r2 = f["c1c/N=$(N)_unif/neval=$neval/meas"]
            r3 = f["c1d/N=$(N)_unif/neval=$neval/meas"]
            c1nl_N_total =
                f["c1b0/N=$(N)_unif/neval=$neval/meas"] +
                f["c1c/N=$(N)_unif/neval=$neval/meas"] +
                f["c1d/N=$(N)_unif/neval=$neval/meas"]
            # The c1b observable has no data for N = 2
            if N > 2
                r4 = f["c1b/N=$(N)_unif/neval=$neval/meas"]
                c1nl_N_total += f["c1b/N=$(N)_unif/neval=$neval/meas"]
                if j == 4
                    println(
                        "c1b0_unif = $r1\nc1b_unif = $r4\nc1c_unif = $r2\nc1d_unif = $r3",
                    )
                end
            else
                if j == 4
                    println("c1b0_unif = $r1\nc1c_unif = $r2\nc1d_unif = $r3")
                end
            end
            if j == 4
                println("c1nl_N_total = $c1nl_N_total")
            end
            close(f)  # close file
            @assert length(c1nl_N_total) == 1

            # Get means and error bars from the result up to this order
            push!(c1nl_N_means, Measurements.value(c1nl_N_total[1]))
            push!(c1nl_N_stdevs, Measurements.uncertainty(c1nl_N_total[1]))
        end
        ax.plot(
            lambdas,
            c1nl_N_means,
            "o-";
            color="C$i",
            markersize=3,
            label="\$N=$N\$ ($solver)",
        )
        ax.fill_between(
            lambdas,
            (c1nl_N_means - c1nl_N_stdevs),
            (c1nl_N_means + c1nl_N_stdevs);
            color="C$i",
            alpha=0.3,
        )
    end
    # ax.set_ylim(; top=-0.195)
    ax.legend(; loc="best")
    ax.set_xlabel("\$\\lambda\$ (Ry)")
    ax.set_ylabel(
        "\$C^{(1)nl}(k=0,\\, \\lambda) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    xloc = 1.07
    yloc = -0.54
    ydiv = -0.025
    # xloc = 1.7
    # yloc = -0.5
    # ydiv = -0.05
    ax.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta), N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + ydiv,
        "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=12,
    )
    # plt.title("")
    plt.tight_layout()
    fig.savefig(
        "results/c1nl/c1nl_k=0_rs=$(rs)_" *
        "beta_ef=$(beta)_neval=$(neval)_" *
        "$(intn_str)$(solver)_vs_lambda.pdf",
    )
    plt.close("all")
    return
end

main()
