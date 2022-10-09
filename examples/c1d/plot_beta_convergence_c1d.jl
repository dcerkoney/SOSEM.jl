using ElectronGas
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1d/plot_c1d.jl

@enum MeshType begin
    linear
    logarithmic
end

function main()
    rs = 2.0
    mass2 = 0.1
    solver = :vegas
    observable = DiagGen.c1d::DiagGen.Observables

    order = 2
    neval = 1e8

    # Either a linear or logarithmic mesh was used
    # mesh_type = linear::MeshType
    mesh_type = logarithmic::MeshType
    meshtypestr = (mesh_type == linear) ? "linear_" : "log2_"

    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # Load the vegas results
    data_path =
        "results/data/converge_beta_$(meshtypestr)c1d_" *
        "n=$(order)_rs=$(rs)_lambda=$(mass2)_" *
        "neval=$(neval)_$(solver).npz"
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
    beta_grid = sosem_vegas.get("beta_grid")
    means = sosem_vegas.get("means")
    stdevs = sosem_vegas.get("stdevs")

    # Plot numerically exact result for n = 2
    # Compare with zero-temperature quadrature result for the uniform value.
    # Since the bare result is independent of rs after non-dimensionalization, we
    # are free to mix rs of the current MC calculation with this result at rs = 2.
    # Similarly, the bare results were calculated at zero temperature (beta is arb.)
    rs_quad = 2.0
    sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=40.0.npz")
    # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    param_quad = Parameter.atomicUnit(0, rs_quad)    # (dimensionless T, rs)
    eTF_quad = param_quad.qTF^2 / (2 * param_quad.me)
    c1d_quad_unif_dimless = sosem_quad.get("bare_d")[1] / eTF_quad^2

    # Plot the convergence wrt beta
    fig, ax = plt.subplots()
    # Either linear or logarithmic grid
    beta_plot = (mesh_type == linear) ? beta_grid : log2.(beta_grid)
    ax.axhline(DiagGen.get_exact_k0(observable); color="k", label="\$T=0\$ (exact)")
    ax.axhline(c1d_quad_unif_dimless; linestyle="--", color="gray", label="\$T=0\$ (quad)")
    ax.plot(
        beta_plot,
        means,
        "o-";
        markersize=2,
        color="C0",
        label="\$n=$(param.order)\$ ($solver)",
    )
    ax.fill_between(beta_plot, means - stdevs, means + stdevs; color="C0", alpha=0.4)
    ax.legend(; loc="center right")
    local coords
    if mesh_type == linear
        ax.set_xlabel("\$\\beta / \\epsilon_F\$")
        coords = (25, 1.27)
    else
        ax.set_xlabel("\$\\log_2(\\beta / \\epsilon_F)\$")
        coords = (5, 1.5)
    end
    ax.text(coords..., "\$r_s = 2,\\, N_{\\mathrm{eval}} = \\mathrm{1e8}\$"; fontsize=14)
    ax.set_ylabel(
        "\$C^{(1d)}(\\mathbf{k}) \\,/\\, \\epsilon^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    ax.set_xlim(minimum(beta_plot), maximum(beta_plot))
    # ax.set_xticks(collect(range(1, stop=14, step=2)), minor=true)
    plt.tight_layout()
    fig.savefig(
        "results/c1d/n=$(param.order)/converge_beta_$(meshtypestr)c1d_n=$(param.order)_" *
        "rs=$(param.rs)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(solver).pdf",
    )
    plt.close("all")
    return
end

main()