using ElectronLiquid
using ElectronGas
using JLD2
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1b0/plot_c1b0_total_jld2.jl

function main()
    rs = 2.0
    beta = 200.0
    mass2 = 0.1
    solver = :vegasmc
    expand_bare_interactions = true

    neval = 1e7
    max_order = 4
    max_order_plot = 3

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    plotparam =
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
        "results/data/c1bL0_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$ct_string"
    settings, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(plotparam))"
        return f[key]
    end
    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    println(k_kf_grid)
    println(settings)
    println(UEG.paraid(param))
    println(partitions)
    println(res)

    # Plot the results
    fig, ax = plt.subplots()

    # Non-dimensionalize bare and RPA+FL non-local moments
    rs_quad = 2.0
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    # Get Thomas-Fermi screening factor to non-dimensionalize rs = 2 quadrature results
    param_quad = Parameter.atomicUnit(0, rs_quad)    # (dimensionless T, rs)
    eTF_quad = param_quad.qTF^2 / (2 * param_quad.me)

    data = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=200.0.npz")

    # Bare results (stored in Hartree a.u.)
    c1b0_bare_quad = data.get("bare_b") / eTF_quad^2
    ax.plot(k_kf_grid_quad, c1b0_bare_quad, "k"; linestyle="--", label="\$LO\$ (quad)")

    # RPA+FL results for class (b) moment
    c1b_rpa_fl = data.get("rpa+fl_b") / eTF_quad^2
    c1b_rpa_fl_err = data.get("rpa+fl_b_err") / eTF_quad^2
    ax.plot(k_kf_grid_quad, c1b_rpa_fl, "k"; label="RPA\$+\$FL (vegas)")
    ax.fill_between(
        k_kf_grid_quad,
        c1b_rpa_fl - c1b_rpa_fl_err,
        c1b_rpa_fl + c1b_rpa_fl_err;
        color="k",
        alpha=0.3,
    )

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
        # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
        #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
        means *= 2
        stdevs *= 2
        # Data gets noisy above 3rd loop order and 1st Green CT order
        marker = (partitions[o][1] > 3 || partitions[o][2] > 1) ? "o-" : "-"
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
    # ax.set_ylim(-0.25, 0)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$C^{(1b)0}_{\\mathcal{P}}(\\mathbf{k}) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    yoffset = 0.1
    # ax.text(
    #     1.75,
    #     -0.425 + yoffset,
    #     "\$r_s = 2,\\, \\beta \\hspace{0.1em} \\epsilon_F = 200,\$";
    #     fontsize=14,
    # )
    # ax.text(
    #     1.75,
    #     -0.525 + yoffset,
    #     "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
    #     fontsize=14,
    # )
    # ax.text(
    #     1.75,
    #     -0.625 + yoffset,
    #     "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
    #     fontsize=12,
    # )
    # plt.title("Using fixed bare Coulomb interactions \$V_1\$, \$V_2\$")
    plt.title(
        "Using re-expanded Coulomb interactions \$V_1[V_\\lambda]\$, \$V_2[V_\\lambda]\$",
    )
    plt.tight_layout()
    fig.savefig(
        "results/c1b0/c1b0_n=$(param.order)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)_jld2.pdf",
    )
    plt.close("all")
    return
end

main()