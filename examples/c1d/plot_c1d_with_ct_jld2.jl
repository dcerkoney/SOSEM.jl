using ElectronLiquid
using ElectronGas
using JLD2
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1c/plot_c1c.jl

function main()
    rs = 2.0
    beta = 200.0
    mass2 = 0.1
    solver = :vegasmc
    expand_bare_interactions = true

    neval = 5e8
    max_order = 4

    plotparam =
        UEG.ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

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

    # Load the results from JLD2
    savename =
        "results/data/c1c_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_with_ct"
    settings, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(plotparam))"
        return f[key]
    end
    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    println(settings)
    println(UEG.paraid(param))
    println(partitions)
    println(res)

    # Plot the results
    fig, ax = plt.subplots()
    # Compare with the bare quadrature results (stored in Hartree a.u.)
    # Since the bare result is independent of rs after non-dimensionalization, we
    # are free to mix rs of the current MC calculation with this result at rs = 2.
    # Similarly, the bare results were calculated at zero temperature (beta is arb.)
    rs_quad = 2.0
    sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=200.0.npz")
    # np.load("results/data/soms_rs=$(Float64(param.rs))_beta_ef=$(param.beta).npz")
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    param_quad = Parameter.atomicUnit(0, rs_quad)    # (dimensionless T, rs)
    eTF_quad = param_quad.qTF^2 / (2 * param_quad.me)
    c1c_quad_dimless = sosem_quad.get("bare_c") / eTF_quad^2
    ax.plot(
        k_kf_grid_quad,
        c1c_quad_dimless,
        "k";
        label="\$\\mathcal{P}=$((2,0,0))\$ (quad)",
    )
    for o in eachindex(partitions)
        # Get means and error bars from the result for this partition
        local means, stdevs
        if res.config.N == 1
            # res gets automatically flattened for a single-partition measurement
            means, stdevs = res.mean, res.stdev
        else
            means, stdevs = res.mean[o], res.stdev[o]
        end
        # Data gets noisy above 3rd loop order
        marker = partitions[o][1] > 3 ? "o-" : "-"
        ax.plot(
            k_kf_grid,
            means,
            marker;
            markersize=2,
            color="C$(o - 1)",
            label="\$\\mathcal{P}=$(partitions[o])\$ ($solver)",
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
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$C^{(1c)}_{\\mathcal{P}}(\\mathbf{k}) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    yoffset = 0.1
    ax.text(
        1.75,
        -0.425 + yoffset,
        "\$r_s = 2,\\, \\beta \\hspace{0.1em} \\epsilon_F = 200,\$";
        fontsize=14,
    )
    ax.text(
        1.75,
        -0.525 + yoffset,
        "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{5e8},\$";
        fontsize=14,
    )
    ax.text(
        1.75,
        -0.625 + yoffset,
        "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=12,
    )
    # plt.title("Using fixed bare Coulomb interactions \$V_1\$, \$V_2\$")
    plt.title(
        "Using re-expanded Coulomb interactions \$V_1[V_\\lambda]\$, \$V_2[V_\\lambda]\$",
    )
    plt.tight_layout()
    fig.savefig(
        "results/c1c/c1c_n=$(param.order)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_with_ct_jld2.pdf",
    )
    plt.close("all")
    return
end

main()