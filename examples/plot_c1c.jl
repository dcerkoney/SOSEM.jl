using ElectronGas
using IterTools: flagfirst
using Plots
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

function main()
    rs = 2.0
    beta = 40.0
    maxeval = 5e7
    solver = :vegasmc
    orders = [2]
    max_order = maximum(orders)

    fig, ax = plt.subplots()
    for (i, order) in enumerate(orders)
        # Load the vegas results
        data_path =
            "results/data/c1c_n=$(order)_rs=$(Float64(rs))_" *
            "beta_ef=$(beta)_neval=$(maxeval)_$(solver).npz"
        print("Loading data for n = $order at '$data_path'...")
        sosem_vegas = np.load(data_path)
        println("done!")
        params = UEG_MC.PlotParams(sosem_vegas.get("params")...)
        println(params)
        # TODO: kwargs implementation (kgrid_<solver>...)
        # solver = params.solver
        kgrid = sosem_vegas.get("kgrid")
        means = sosem_vegas.get("means")
        stdevs = sosem_vegas.get("stdevs")

        # k / kf
        k_kf_grid = kgrid / params.kF

        # Plot numerically exact result for n = 2
        if i == 1
            # Compare with bare quadrature results (stored in Hartree a.u.)
            rs_quad = 2.0
            sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=40.0.npz")
            # np.load("results/data/soms_rs=$(Float64(params.rs))_beta_ef=$(params.beta).npz")
            k_kf_grid_quad = np.linspace(0.0, 6.0; num=600)
            # Get Thomas-Fermi screening factor to non-dimensionalize rs = 2 quadrature results
            qTF_quad = Parameter.rydbergUnit(0, rs_quad).qTF    # (dimensionless T, rs)
            c1c_quad_dimless = 4 * sosem_quad.get("bare_c") / qTF_quad^4
            ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
            ax.plot(k_kf_grid_quad, c1c_quad_dimless, "k"; label="\$n=2\$ (quad)")
        end

        # Plot Monte-Carlo result at this order
        ax.plot(
            k_kf_grid,
            means,
            "o-";
            markersize=2,
            color="C$(i-1)",
            label="\$n=$(order)\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            color="C$(i-1)",
            alpha=0.4,
        )
    end
    # Setup legend and axes
    ax.legend(; loc="best")
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$C^{(1c)}(\\mathbf{k}) \\,/\\, q^{4}_{\\mathrm{TF}}\$")
    plt.tight_layout()
    # Save the plot
    fig.savefig(
        "results/c1c/c1c_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_neval=$(maxeval)_$(solver).pdf",
    )
    plt.close("all")
    return
end

main()