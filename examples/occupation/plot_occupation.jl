using CodecZlib
using ElectronGas
using ElectronLiquid
using Interpolations
using JLD2
using Lehmann
using LsqFit
using Measurements
using Parameters
using Polynomials
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt
@pyimport mpl_toolkits.axes_grid1.inset_locator as il

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    rs = 1.0
    beta = 40.0
    mass2 = 1.0
    solver = :vegasmc

    # Number of evals
    neval = 1e6

    # Plot part itions for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order = 1
    max_order = 3
    min_order_plot = 0
    max_order_plot = 2
    plot_orders = collect(min_order_plot:max_order_plot)

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = false

    # Remove Fock insertions?
    isFock = true

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end
    if isFock
        ct_string *= "_noFock"
    end

    # UEG parameters for MC integration
    loadparam = ParaMC(;
        order=max_order,
        rs=rs,
        beta=beta,
        mass2=mass2,
        isDynamic=false,
        isFock=isFock,
    )

    # Load the raw data
    savename =
        "results/data/occupation_n=$(max_order)_rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(mass2)_neval=$(neval)_$(solver)_$(ct_string)"
    orders, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(loadparam))"
        return f[key]
    end

    # Get dimensionless k-grid (k / kF) and index corresponding to the Fermi energy
    k_kf_grid = kgrid / param.kF

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    # Zero out double-counting (Fock renormalized) partitions
    if isFock && min_order ≤ 1
        data[(1, 0, 0)] = zero(data[(max_order, 0, 0)])
        # data[(0, 1, 0)] = zero(data[(max_order, 0, 0)])  # Combines with dMu2, nonzero!
    end
    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])
    # println(merged_data)

    # List of interaction-merged partitions
    partitions_merged = sort(collect(keys(merged_data)))

    # Get exact bare/Fock occupation
    if param.isFock
        fock =
            SelfEnergy.Fock0_ZeroTemp.(kgrid, [param.basic]) .-
            SelfEnergy.Fock0_ZeroTemp(param.kF, param.basic)
        ϵk = kgrid .^ 2 / (2 * param.me) .- param.μ + fock  # ϵ_HF = ϵ_0 + (Σ_F(k) - δμ₁)
    else
        ϵk = kgrid .^ 2 / (2 * param.me) .- param.μ         # ϵ_0
    end
    bare_occupation_exact = -Spectral.kernelFermiT.(-1e-8, ϵk, param.β)

    # Set bare result manually using exact Fermi function
    if min_order_plot == 0 && min_order > 0
        # treat quadrature data as numerically exact
        data[(0, 0, 0)] = measurement.(bare_occupation_exact, 0.0)
        merged_data[(0, 0)] = measurement.(bare_occupation_exact, 0.0)
    elseif min_order_plot == 0 && min_order == 0
        stdscores = stdscore.(merged_data[(0, 0)], bare_occupation_exact)
        worst_score = argmax(abs, stdscores)
        println("Worst standard score for Fock occupation: $worst_score")
        @assert worst_score ≤ 10
    end

    println(param)
    println(UEG.paraid(param))
    println(partitions)
    println(partitions_merged)
    println(res)
    println(k_kf_grid)

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # Plot the occupation number for each partition
    fig1, ax1 = plt.subplots()
    ax1.axvline(1.0; linestyle="--", linewidth=1, color="gray")
    for (i, P) in enumerate(partitions)
        isFock && P == (1, 0, 0) && continue  # Don't plot double-counted partition
        (min_order_plot ≤ sum(P) ≤ max_order_plot) == false && continue
        # Get means and error bars from the result up to this order
        means = Measurements.value.(data[P])
        stdevs = Measurements.uncertainty.(data[P])
        marker = "o-"
        ax1.plot(k_kf_grid, means, marker; markersize=2, color="C$(i-1)", label="\$P=$P\$")
        ax1.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            color="C$(i-1)",
            alpha=0.4,
        )
    end
    ax1.legend(; loc="best")
    ax1.set_xlim(0.8, 1.2)
    # ax1.set_ylim(nothing, 2)
    ax1.set_xlabel("\$k / k_F\$")
    ax1.set_ylabel("\$n_{\\mathcal{P}}(k\\sigma)\$")
    xloc = 1.025
    yloc = -10
    ydiv = -5
    ax1.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax1.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
        fontsize=14,
    )
    fig1.tight_layout()
    fig1.savefig(
        "results/occupation/benchmark/occupation_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string)_partns.pdf",
    )

    # Plot the occupation number for each interaction-merged partition
    fig2, ax2 = plt.subplots()
    ax2.axvline(1.0; linestyle="--", linewidth=1, color="gray")
    for (i, Pm) in enumerate(partitions_merged)
        isFock && Pm == (1, 0) && continue  # Don't plot double-counted partition
        (min_order_plot ≤ sum(Pm) ≤ max_order_plot) == false && continue
        # Get means and error bars from the result up to this order
        means = Measurements.value.(merged_data[Pm])
        stdevs = Measurements.uncertainty.(merged_data[Pm])
        marker = "o-"
        ax2.plot(k_kf_grid, means, marker; markersize=2, color="C$(i-1)", label="\$P=$Pm\$")
        ax2.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            color="C$(i-1)",
            alpha=0.4,
        )
    end
    ax2.legend(; loc="best")
    ax2.set_xlim(0.8, 1.2)
    # ax2.set_ylim(nothing, 2)
    ax2.set_xlabel("\$k / k_F\$")
    ax2.set_ylabel("\$n_{\\tilde{\\mathcal{P}}}(k\\sigma)\$")
    xloc = 1.025
    yloc = -10
    ydiv = -5
    ax2.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax2.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
        fontsize=14,
    )
    fig2.tight_layout()
    fig2.savefig(
        "results/occupation/benchmark/occupation_N=$(plot_orders)_rs=$(param.rs)_beta_ef=$(param.beta)_" *
        "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string)_merged_partns.pdf",
    )

    plt.close("all")
    return
end

main()
