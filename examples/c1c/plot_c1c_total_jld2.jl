using ElectronLiquid
using ElectronGas
using JLD2
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1c/plot_c1c.jl

"""Convert a list of MCIntegration results for partitions {P} to a Dict of measurements."""
function restodict(res, partitions)
    data = Dict()
    for (i, p) in enumerate(partitions)
        data[p] = measurement.(res.mean[i], res.stdev[i])
    end
    return data
end

"""
Aggregate the measurements for C⁽¹ᶜ⁾ up to order N for nmin ≤ N ≤ nmax.
Assumes the input data has already been merged by interaction order and, 
if applicable, reexpanded in μ.
"""
function aggregate_results_c1cN(merged_data; nmax, nmin=2)
    c1cN = Dict()
    for n in nmin:nmax
        c1cN[n] = zero(merged_data[(n, 0)])
        println(n)
        for (p, meas) in merged_data
            if p[1] <= n
                c1cN[n] += meas
            end
        end
    end
    return c1cN
end

function main()
    rs = 2.0
    beta = 200.0
    mass2 = 2.0
    solver = :vegasmc
    expand_bare_interactions = true

    neval = 5e8
    min_order = 2
    max_order = 3

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
        "results/data/c1c_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)"
    settings, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(plotparam))"
        return f[key]
    end
    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = restodict(res, partitions)
    merged_data = CounterTerm.mergeInteraction(data)

    # Aggregate the full results for C⁽¹ᶜ⁾
    c1cN = aggregate_results_c1cN(merged_data; nmax=max_order, nmin=min_order)

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
    ax.plot(k_kf_grid_quad, c1c_quad_dimless, "k"; label="\$N = 2\$ (bare, quad)")
    # Plot for each aggregate order
    for N in min_order:max_order
        # Get means and error bars from the result up to this order
        means = Measurements.value.(c1cN[N])
        stdevs = Measurements.uncertainty.(c1cN[N])
        # Data gets noisy above 3rd loop order
        marker = N > 3 ? "o-" : "-"
        ax.plot(
            k_kf_grid,
            means,
            marker;
            markersize=2,
            color="C$(N - 2)",
            label="\$N=$N\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            color="C$(N - 2)",
            alpha=0.4,
        )
    end
    ax.legend(; loc="lower right")
    ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$C^{(1c)}_{N}(\\mathbf{k}) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    # xloc = 0.5
    # yloc = -0.075
    # ydiv = -0.009
    xloc = 1.75
    yloc = -0.3
    ydiv = -0.085
    ax.text(
        xloc,
        yloc,
        "\$r_s = 2,\\, \\beta \\hspace{0.1em} \\epsilon_F = 200,\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = 2\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{5e8},\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + 2 * ydiv,
        "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=12,
    )
    # plt.title("Using fixed bare Coulomb interactions \$V_1\$, \$V_2\$")
    plt.title(
        "Using re-expanded Coulomb interactions \$V_1[V_\\lambda]\$, \$V_2[V_\\lambda]\$",
    )
    plt.tight_layout()
    fig.savefig(
        "results/c1c/c1c_N=$(param.order)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)_total.pdf",
    )
    plt.close("all")
    return
end

main()
