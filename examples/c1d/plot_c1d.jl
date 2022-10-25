using ElectronGas
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1d/plot_c1d.jl

function main()
    rs = 2.0
    beta = 200.0
    mass2 = 0.1
    solver = :vegasmc
    expand_bare_interactions = true

    orders = [2]
    nevals = [1e7]
    maxeval = maximum(nevals)
    max_order = maximum(orders)

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions
        intn_str = "no_bare_"
    end

    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    colors = ["orchid", "cornflowerblue", "turquoise", "chartreuse", "greenyellow"]
    markers = ["-", "-", "o-", "o-", "o-"]

    fig, ax = plt.subplots()
    for (i, order) in enumerate(orders)
        # Load the vegas results
        data_path =
            "results/data/c1d_n=$(order)_rs=$(Float64(rs))_" *
            "beta_ef=$(beta)_lambda=$(mass2)_" *
            "neval=$(nevals[i])_$(intn_str)$(solver).npz"
        print("Loading data for n = $order at '$data_path'...")
        sosem_vegas = np.load(data_path)
        println("done!")

        paramdict = sosem_vegas.get("param")
        if isnothing(paramdict)
            paramdict = sosem_vegas.get("params")
        end
        println(paramdict)
        param = UEG_MC.PlotParams(paramdict...)
        # TODO: kwargs implementation (kgrid_<solver>...)
        # solver = param.solver
        kgrid = sosem_vegas.get("kgrid")
        means = sosem_vegas.get("means")
        stdevs = sosem_vegas.get("stdevs")

        # k / kf
        k_kf_grid = kgrid / param.kF

        # Plot numerically exact result for n = 2
        if i == 1
            # Compare with bare quadrature results (stored in Hartree a.u.)
            rs_quad = 2.0
            sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=200.0.npz")
            # np.load("results/data/soms_rs=$(Float64(param.rs))_beta_ef=$(param.beta).npz")
            k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
            # Get Thomas-Fermi screening factor to non-dimensionalize rs = 2 quadrature results
            param_quad = Parameter.atomicUnit(0, rs_quad)    # (dimensionless T, rs)
            eTF_quad = param_quad.qTF^2 / (2 * param_quad.me)
            c1d_quad_dimless = sosem_quad.get("bare_c") / eTF_quad^2
            # qTF_quad = Parameter.atomicUnit(0, rs_quad).qTF    # (dimensionless T, rs)
            # c1d_quad_dimless = 4 * sosem_quad.get("bare_c") / qTF_quad^4
            ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
            ax.plot(k_kf_grid_quad, c1d_quad_dimless, "k"; label="\$n=2\$ (bare, quad)")
        end

        # Plot Monte-Carlo result at this order
        ax.plot(
            k_kf_grid,
            means,
            markers[i];
            markersize=2,
            color=colors[i],
            label="\$n=$(order)\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            color=colors[i],
            alpha=0.3,
        )
    end
    # Setup legend and axes
    ax.legend(; loc="lower right")
    ax.set_xlim(0.0, 3.0)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$C^{(1d)}_n(\\mathbf{k}) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    ax.text(1.75, -0.4, "\$r_s = 2,\\, \\beta = 200 \\epsilon_F,\$"; fontsize=14)
    ax.text(
        1.75,
        -0.5,
        "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{1e8}\$";
        fontsize=14,
    )
    plt.title("Using fixed bare Coulomb interactions \$V_1\$, \$V_2\$")
    # plt.title(
    #     "Using re-expanded Coulomb interactions \$V_1[V_\\lambda]\$, \$V_2[V_\\lambda]\$",
    # )
    plt.tight_layout()
    # Save the plot
    fig.savefig(
        "results/c1d/c1d_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(maxeval)_$(solver).pdf",
        # "neval=$(maxeval)_$(intn_str)$(solver).pdf",
    )
    plt.close("all")
    return
end

main()