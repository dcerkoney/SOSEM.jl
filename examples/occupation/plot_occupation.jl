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
    # betas = [25.0, 40.0, 80.0]
    betas = [25.0]
    mass2 = 1.0
    solver = :vegasmc

    # Number of evals
    neval = 1e9

    # Plot part itions for orders min_order_plot ≤ ξ ≤ max_order_plot
    max_order = 3
    min_order_plot = 2
    max_order_plot = 3
    plot_orders = collect(min_order_plot:max_order_plot)

    for beta in betas

        # UEG parameters for MC integration
        loadparam =
            ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

        # Load the raw data
        savename =
            "results/data/occupation_n=$(max_order)_rs=$(rs)_" *
            "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval)_$(solver)"
        orders, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
            key = "$(UEG.short(loadparam))"
            return f[key]
        end

        # Get dimensionless k-grid (k / kF) and index corresponding to the Fermi energy
        k_kf_grid = kgrid / param.kF

        # Convert results to a Dict of measurements at each order with interaction counterterms merged
        data = UEG_MC.restodict(res, partitions)
        merged_data = CounterTerm.mergeInteraction(data)
        println([k for (k, _) in merged_data])
        # println(merged_data)

        # List of interaction-merged partitions
        partitions_merged = sort(collect(keys(merged_data)))

        if min_order_plot == 0
            # Set bare result manually using exact Fermi function
            ϵk = kgrid .^ 2 / (2 * param.me) .- param.μ
            bare_occupation_exact = -Spectral.kernelFermiT.(-1e-8, ϵk, param.β)
            # treat quadrature data as numerically exact
            data[(0, 0, 0)] = measurement.(bare_occupation_exact, 0.0)
            merged_data[(0, 0)] = measurement.(bare_occupation_exact, 0.0)
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
            (min_order_plot ≤ sum(P) ≤ max_order_plot) == false && continue
            # Get means and error bars from the result up to this order
            means = Measurements.value.(data[P])
            stdevs = Measurements.uncertainty.(data[P])
            marker = "o-"
            ax1.plot(
                k_kf_grid,
                means,
                marker;
                markersize=2,
                color="C$(i-1)",
                label="\$P=$P\$",
            )
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
            "results/occupation/occupation_N=$(max_order_plot)_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver)_partns.pdf",
        )

        # Plot the occupation number for each interaction-merged partition
        fig2, ax2 = plt.subplots()
        ax2.axvline(1.0; linestyle="--", linewidth=1, color="gray")
        for (i, Pm) in enumerate(partitions_merged)
            (min_order_plot ≤ sum(Pm) ≤ max_order_plot) == false && continue
            # Get means and error bars from the result up to this order
            means = Measurements.value.(merged_data[Pm])
            stdevs = Measurements.uncertainty.(merged_data[Pm])
            marker = "o-"
            ax2.plot(
                k_kf_grid,
                means,
                marker;
                markersize=2,
                color="C$(i-1)",
                label="\$P=$Pm\$",
            )
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
            "results/occupation/occupation_N=$(plot_orders)_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver)_merged_partns.pdf",
        )

        plt.close("all")
    end

    return
end

main()
