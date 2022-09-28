using Plots
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

function main()
    rs = 2.0
    beta = 40.0
    maxeval = 1.0e6

    # Load the vegas results
    sosem_vegas = np.load("data/c1c_rs=$(Float64(rs))_beta_ef=$(beta)_neval=$(maxeval).npz")
    params = UEG_MC.PlotParams(sosem_vegas.get("params")...)
    kgrid = sosem_vegas.get("kgrid")
    means = sosem_vegas.get("means")
    stdevs = sosem_vegas.get("stdevs")

    # k / kf
    k_kf_grid = kgrid / params.kF
    println(k_kf_grid)

    # Compare with quadrature results (stored in Hartree a.u.)
    sosem_quad = np.load("data/soms_rs=$(Float64(rs))_beta_ef=$(beta).npz")
    k_kf_grid_quad = np.linspace(0.0, 6.0; num=600)
    # NOTE: (q_TF a₀) is dimensionless, hence q_TF  is the same in Rydberg
    #       and Hartree a.u., and no additional conversion factor is needed
    c1c_quad_dimless = sosem_quad.get("bare_c") / params.qTF^4

    # Plot the result
    fig, ax = plt.subplots()
    ax.plot(k_kf_grid_quad, c1c_quad_dimless, "k"; label="\$n=$(params.order)\$ (quad)")
    ax.plot(k_kf_grid, means, "o-"; color="C0", label="\$n=$(params.order)\$ (vegas)")
    ax.fill_between(k_kf_grid, means - stdevs, means + stdevs; color="C0", alpha=0.4)
    ax.legend(; loc="best")
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$C^{(1c)}(\\mathbf{k}) / q^{4}_{\\mathrm{TF}}\$")
    ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    plt.tight_layout()
    fig.savefig(
        "c1c_n=$(params.order)_rs=$(rs)_" *
        "beta_ef=$(beta)_neval=$(maxeval)_dimless.pdf",
    )
    plt.close("all")
    return
end

main()