using ElectronGas
using ElectronLiquid.UEG: ParaMC, short
using FeynmanDiagram
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
        observable=DiagGen.c1bL,
        min_order=3,  # no (2,0,0) partition for this observable (Γⁱ₃ > Γ₀),
        max_order=4,
        verbosity=DiagGen.quiet,
        expand_bare_interactions=false,
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
        # interaction=[FeynmanDiagram.Interaction(ChargeCharge, Dynamic)],  # TODO: test RPA-type interaction
    )

    # TODO: Implement min/max DiagGen orders. Add a method to select hard-coded
    #       partitions and update the JLD2 data of a single partition accordingly.

    # UEG parameters for MC integration
    param =
        ParaMC(; order=settings.max_order, rs=1.0, beta=200.0, mass2=2.0, isDynamic=false)
    @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)"

    # K-mesh for measurement
    k_kf_grid = np.load("results/kgrids/kgrid_vegas_dimless_n=77_small.npy")
    kgrid = param.kF * k_kf_grid

    # Settings
    alpha = 2.0
    print = 0
    plot = true
    compare_bare = true
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 5e8

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Build diagram and expression trees for all loop and counterterm partitions
    partitions, diagparams, diagtrees, exprtrees = DiagGen.build_nonlocal_with_ct(
        settings;
        renorm_mu=renorm_mu,
        renorm_lambda=renorm_lambda,
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

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # Save to JLD2 on main thread
    if !isnothing(res)
        savename =
            "results/data/c1bL_n=$(param.order)_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
            "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)"
        jldopen("$savename.jld2", "a+") do f
            key = "$(short(param))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (settings, param, kgrid, partitions, res)
        end

        if plot
            # Plot the result
            fig, ax = plt.subplots()
            # Compare with bare quadrature results (stored in Hartree a.u.)
            if compare_bare
                # NOTE: The bare results were calculated at zero temperature (beta is arb.)
                rs_quad = 1.0
                sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=200.0.npz")
                # np.load("results/data/soms_rs=$(Float64(param.rs))_beta_ef=$(param.beta).npz")
                k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
                # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
                param_quad = Parameter.atomicUnit(0, rs_quad)    # (dimensionless T, rs)
                eTF_quad = param_quad.qTF^2 / (2 * param_quad.me)
                c1b_lo_quad_dimless = sosem_quad.get("bare_b") / eTF_quad^2
                ax.plot(
                    k_kf_grid_quad,
                    c1b_lo_quad_dimless,
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
                # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
                #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
                means = 2 * means
                stdevs = 2 * stdevs

                # Data gets noisy above 1st Green's function counterterm order
                marker = partitions[o][2] > 1 ? "o-" : "-"
                # Data gets noisy above 3rd loop order
                # marker = partitions[o][1] > 3 ? "o-" : "-"
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
                "\$C^{(1b)}_{\\mathcal{P}}(\\mathbf{k}) \\,/\\, E^{2}_{\\mathrm{TF}}\$",
            )
            ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
            plt.tight_layout()
            fig.savefig(
                "results/c1b/n=$(param.order)/c1b_n=$(param.order)_rs=$(param.rs)_" *
                "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
                "neval=$(neval)_$(intn_str)$(solver)_$(ct_string).pdf",
            )
            plt.close("all")
        end
    end
end

main()
