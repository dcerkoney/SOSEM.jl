using ElectronGas
using ElectronLiquid.UEG: ParaMC, short
using JLD2
using Measurements
using SOSEM
using PyCall

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

function main()
    # Debug mode
    if isinteractive()
        ENV["JULIA_DEBUG"] = SOSEM
    end

    settings = DiagGen.Settings(;
        observable=DiagGen.c1c,
        n_order=4,
        verbosity=DiagGen.info,
        expand_bare_interactions=true,
    )

    # UEG parameters for MC integration
    param = ParaMC(; order=settings.n_order, rs=2.0, beta=200.0, mass2=0.1, isDynamic=false)
    @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)"

    # K-mesh for measurement
    # k_kf_grid = [0.0]
    k_kf_grid = np.load("results/kgrids/kgrid_vegas_dimless_n=77_small.npy")
    # k_kf_grid = np.load("results/kgrids/kgrid_vegas_dimless_n=25_small.npy")
    kgrid = param.kF * k_kf_grid

    # Settings
    alpha = 2.0
    print = 0
    plot = true
    compare_bare = true
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 5e8

    # Build diagram and expression trees for all loop and counterterm partitions
    partitions, diagparams, diagtrees, exprtrees = DiagGen.build_nonlocal_with_ct(
        settings;
        fixed_order=false,
        renorm_mu=false,
        renorm_lambda=true,
    )

    println("Integrating partitions: $partitions")
    @debug "diagtrees: $diagtrees"
    @debug "exprtrees: $exprtrees"

    # Check the diagram tree
    # for d in diagtrees
    #     DiagGen.checktree(d, settings)
    # end

    # Bin external momenta, performing a single integration
    res = UEG_MC.integrate_nonlocal_with_ct(
        settings,
        param,
        diagparams,
        exprtrees;
        kgrid=kgrid,
        alpha=alpha,
        neval=neval,
        print=print,
        solver=solver,
    )

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if settings.expand_bare_interactions
        intn_str = "no_bare_"
    end

    # Save to JLD2 on main thread
    if !isnothing(res)
        savename =
            "results/data/c1c_n=$(param.order)_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
            "neval=$(neval)_$(intn_str)$(solver)_with_ct"
        jldopen("$savename.jld2", "a+") do f
            key = "$(short(param))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (settings, param, kgrid, partitions, res)
        end
    end

    if !isnothing(res) && plot
        # Plot the result
        fig, ax = plt.subplots()
        # Compare with bare quadrature results (stored in Hartree a.u.)
        if compare_bare
            # Since the bare result is independent of rs after non-dimensionalization, we
            # are free to mix rs of the current MC calculation with this result at rs = 2.
            # Similarly, the bare results were calculated at zero temperature (beta is arb.)
            rs_quad = 2.0
            sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=40.0.npz")
            # np.load("results/data/soms_rs=$(Float64(param.rs))_beta_ef=$(param.beta).npz")
            k_kf_grid_quad = np.linspace(0.0, 6.0; num=600)
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
        end
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
        ax.set_xlabel("\$k / k_F\$")
        ax.set_ylabel(
            "\$C^{(1c)}_{\\mathcal{P}}(\\mathbf{k}) \\,/\\, E^{2}_{\\mathrm{TF}}\$",
        )
        ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
        plt.tight_layout()
        fig.savefig(
            "results/c1c/n=$(param.order)/c1c_n=$(param.order)_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
            "neval=$(neval)_$(intn_str)$(solver)_with_ct.pdf",
        )
        plt.close("all")
    end
end

main()