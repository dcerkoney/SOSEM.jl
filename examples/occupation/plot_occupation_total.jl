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
    betas = [40.0]
    mass2 = 1.0
    solver = :vegasmc

    # Number of evals
    neval = 1e7

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order = 1
    max_order = 3
    min_order_plot = 0
    max_order_plot = 3

    # Save total results
    save = true

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""

    # Full renormalization
    ct_string = "with_ct_mu_lambda"

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
        println(k_kf_grid)

        # Convert results to a Dict of measurements at each order with interaction counterterms merged
        data = UEG_MC.restodict(res, partitions)
        merged_data = CounterTerm.mergeInteraction(data)
        println([k for (k, _) in merged_data])
        # println(merged_data)

        # TODO: Fix factor of β by removing extra τ integration
        for (k, v) in merged_data
            merged_data[k] = v / param.β
        end

        if min_order_plot == 0
            # Set bare result manually using exact Fermi function
            ϵk = kgrid .^ 2 / (2 * param.me) .- param.μ
            bare_occupation_exact = -Spectral.kernelFermiT.(-1e-8, ϵk, param.β)
            merged_data[(0, 0)] = measurement.(bare_occupation_exact, 0.0)  # treat quadrature data as numerically exact
        end


        # Reexpand merged data in powers of μ
        z, μ = UEG_MC.load_z_mu(param)
        δz, δμ = CounterTerm.sigmaCT(max_order, μ, z; verbose=1)
        println("Computed δμ: ", δμ)
        occupation = UEG_MC.chemicalpotential_renormalization_green(
            merged_data,
            δμ;
            min_order=min_order_plot,
            max_order=max_order,
        )
        if max_order ≥ 1
            # Test manual renormalization with exact lowest-order chemical potential
            δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
            # nₖ⁽¹⁾ = nₖ_{1,0} + δμ₁ nₖ_{0,1}
            occupation_1_manual = merged_data[(1, 0)] + δμ1_exact * merged_data[(0, 1)]
            stdscores = stdscore.(occupation[1], occupation_1_manual)
            worst_score = argmax(abs, stdscores)
            println("Exact δμ₁: ", δμ1_exact)
            println("Computed δμ₁: ", δμ[1])
            println(
                "Worst standard score for total result to 1st " *
                "order (auto vs exact+manual): $worst_score",
            )
            @assert worst_score ≤ 10
        end
        # Aggregate the full results for Σₓ up to order N
        occupation_total = UEG_MC.aggregate_orders(occupation)

        println(UEG.paraid(param))
        println(partitions)
        println(res)

        if save
            savename =
                "results/data/rs=$(param.rs)_beta_ef=$(param.beta)_" *
                "lambda=$(param.mass2)_$(intn_str)$(solver)_$(ct_string)"
            f = jldopen("$savename.jld2", "a+")
            # NOTE: no bare result for c1b observable (accounted for in c1b0)
            for N in min_order_plot:max_order
                # Skip exact (N = 0) result
                N == 0 && continue
                if haskey(f, "occupation") &&
                   haskey(f["occupation"], "N=$N") &&
                   haskey(f["occupation/N=$N"], "neval=$(neval)")
                    @warn("replacing existing data for N=$N, neval=$(neval)")
                    delete!(f["occupation/N=$N"], "neval=$(neval)")
                end
                f["occupation/N=$N/neval=$neval/meas"] = occupation_total[N]
                f["occupation/N=$N/neval=$neval/param"] = param
                f["occupation/N=$N/neval=$neval/kgrid"] = kgrid
            end
        end

        # Use LaTex fonts for plots
        plt.rc("text"; usetex=true)
        plt.rc("font"; family="serif")

        # Bare occupation fₖ on dense grid for plotting
        kgrid_fine = param.kF * np.linspace(0.0, 3.0; num=600)
        k_kf_grid_fine = np.linspace(0.0, 3.0; num=600)
        ϵk = kgrid_fine .^ 2 / (2 * param.me) .- param.μ
        bare_occupation_fine = -Spectral.kernelFermiT.(-1e-8, ϵk, param.β)

        # Plot the occupation number for each aggregate order
        fig, ax = plt.subplots()
        # Compare result to bare occupation fₖ
        ax.plot(k_kf_grid_fine, bare_occupation_fine, "k"; label="\$N=0\$ (exact)")
        ax.axvline(1.0; linestyle="--", linewidth=1, color="gray")
        for (i, N) in enumerate(min_order:max_order_plot)
            # N == 0 && continue
            # Get means and error bars from the result up to this order
            means = Measurements.value.(occupation_total[N])
            stdevs = Measurements.uncertainty.(occupation_total[N])
            marker = "o-"
            ax.plot(
                k_kf_grid,
                means,
                marker;
                markersize=2,
                color="C$(i-1)",
                label="\$N=$N\$ ($solver)",
            )
            ax.fill_between(
                k_kf_grid,
                means - stdevs,
                means + stdevs;
                color="C$(i-1)",
                alpha=0.4,
            )
        end
        ax.legend(; loc="upper right")
        ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
        # ax.set_ylim(nothing, 2)
        ax.set_xlabel("\$k / k_F\$")
        ax.set_ylabel("\$n_{k\\sigma}\$")
        xloc = 1.125
        # yloc = 0.4
        # ydiv = -0.1
        yloc = 0.75
        ydiv = -0.2
        ax.text(
            xloc,
            yloc,
            "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
            fontsize=14,
        )
        ax.text(
            xloc,
            yloc + ydiv,
            "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
            # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
            fontsize=14,
        )
        fig.tight_layout()

        # Plot inset
        ax_inset =
            il.inset_axes(ax; width="38%", height="28.5%", loc="lower left", borderpad=0)
        # Compare result to bare occupation fₖ
        ax_inset.plot(k_kf_grid_fine, bare_occupation_fine, "k"; label="\$N=0\$ (exact)")
        ax_inset.axvline(1.0; linestyle="--", linewidth=1, color="gray")
        # ax_inset.axhspan(0, 1; alpha=0.2, facecolor="k", edgecolor=nothing)
        ax_inset.axhline(0.0; linestyle="-", linewidth=0.5, color="k")
        ax_inset.axhline(1.0; linestyle="-", linewidth=0.5, color="k")
        for (i, N) in enumerate(min_order:max_order_plot)
            # N == 0 && continue
            # Get means and error bars from the result up to this order
            means = Measurements.value.(occupation_total[N])
            stdevs = Measurements.uncertainty.(occupation_total[N])
            marker = "o-"
            ax_inset.plot(
                k_kf_grid,
                means,
                marker;
                markersize=2,
                color="C$(i-1)",
                label="\$N=$N\$ ($solver)",
            )
            ax_inset.fill_between(
                k_kf_grid,
                means - stdevs,
                means + stdevs;
                color="C$(i-1)",
                alpha=0.4,
            )
        end
        xpad = 0.04
        ypad = 0.4
        ax_inset.set_xlim(0.8 - xpad, 1.2 + xpad)
        ax_inset.set_ylim(-ypad, 1 + ypad)
        ax_inset.set_xlabel("\$k / k_F\$"; labelpad=7)
        ax_inset.set_ylabel("\$n_{k\\sigma}\$")
        ax_inset.xaxis.set_label_position("top")
        ax_inset.yaxis.set_label_position("right")
        ax_inset.xaxis.tick_top()
        ax_inset.yaxis.tick_right()
        # fig.savefig(
        #     "results/occupation/occupation_N=$(max_order_plot)_rs=$(param.rs)_" *
        #     "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver).pdf",
        # )
        fig.savefig(
            "results/occupation/occupation_N=$(max_order_plot)_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver).pdf",
        )

        plt.close("all")
    end

    return
end

main()
