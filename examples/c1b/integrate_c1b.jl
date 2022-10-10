using ElectronGas
using ElectronLiquid.UEG: ParaMC
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

    # Settings for diagram generation
    settings = DiagGen.Settings(;
        observable=DiagGen.c1bL0,
        n_order=2,
        verbosity=DiagGen.info,
        expand_bare_interactions=false,
    )

    # UEG parameters for MC integration
    mcparam =
        ParaMC(; order=settings.n_order, rs=2.0, beta=40.0, mass2=1e-8, isDynamic=false)
    @debug "β / EF = $(mcparam.beta), β = $(mcparam.β), EF = $(mcparam.EF)" maxlog = 1

    # K-mesh for measurement
    # k_kf_grid = [0.0]
    # k_kf_grid = np.load("results/kgrids/kgrid_vegas_dimless_n=103.npy")
    k_kf_grid = np.load("results/kgrids/kgrid_vegas_dimless_n=77_small.npy")
    kgrid = mcparam.kF * k_kf_grid

    # Settings
    alpha = 2.0
    print = 0
    plot = true
    compare_bare = true
    solver = :vegas

    # Number of evals below and above kF
    neval = 1e6

    # Generate the diagrams
    diagparam, diagtree, exprtree = DiagGen.build_nonlocal(settings)

    # Check the diagram tree
    DiagGen.checktree(diagtree, settings)

    # NOTE: We assume there is only a single root in the ExpressionTree
    @assert length(exprtree.root) == 1

    # Loop over external momenta and integrate
    means = Vector{Float64}()
    stdevs = Vector{Float64}()
    # Bin external momenta, performing a single integration
    res = UEG_MC.integrate_nonlocal(
        settings,
        mcparam,
        diagparam,
        exprtree;
        kgrid=kgrid,
        alpha=alpha,
        neval=neval,
        print=print,
    )
    # Extract the result on the main thread.
    if !isnothing(res)
        # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
        #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
        means = 2 * res.mean
        stdevs = 2 * res.stdev
    end

    # z-score test for uniform value for this SOSEM observable
    if mcparam.order == 2 && !isempty(means)
        # C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
        exact = 2 * DiagGen.get_exact_k0(settings.observable)
        # Test standard score (z-score) of the measurement
        meas = measurement(means[1], stdevs[1])
        score = stdscore(meas, exact)
        obsstring = DiagGen.get_bare_string(settings.observable)
        # Result should be accurate to within the specified standard score (by default, 5σ)
        println("""
                $obsstring ($solver):
                 • Exact: $exact
                 • Measured: $meas
                 • Standard score: $score
                """)
    end

    # Save the results of a calculation at multiple external k
    if length(means) > 1
        # Distinguish results with fixed vs re-expanded bare interactions
        intn_str = ""
        if settings.expand_bare_interactions
            intn_str = "no_bare_"
        end
        # Save the result
        savename =
            "results/data/c1b_n=$(mcparam.order)_rs=$(mcparam.rs)_" *
            "beta_ef=$(mcparam.beta)_lambda=$(mcparam.mass2)_" *
            "neval=$(neval)_$(intn_str)$(solver)"
        # Remove old data, if it exists
        rm(savename; force=true)
        # TODO: kwargs implementation (kgrid_<solver>...)
        np.savez(
            savename;
            mcparam=[
                mcparam.order,
                mcparam.rs,
                mcparam.beta,
                mcparam.kF,
                mcparam.qTF,
                mcparam.mass2,
            ],
            kgrid=kgrid,
            means=2 * means,
            stdevs=2 * stdevs,
        )
        if plot
            # Plot the result
            fig, ax = plt.subplots()
            if compare_bare
                # Compare with bare quadrature results (stored in Hartree a.u.);
                # since the bare result is independent of rs after non-dimensionalization, we
                # are free to mix rs of the current MC calculation with this result at rs = 2.
                # Similarly, the bare results were calculated at zero temperature (beta is arb.)
                rs_quad = 2.0
                sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=40.0.npz")
                # np.load("results/data/soms_rs=$(Float64(mcparam.rs))_beta_ef=$(mcparam.beta).npz")
                k_kf_grid_quad = np.linspace(0.0, 6.0; num=600)
                # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
                param_quad = Parameter.atomicUnit(0, rs_quad)    # (dimensionless T, rs)
                eTF_quad = param_quad.qTF^2 / (2 * param_quad.me)
                c1b_quad_dimless = sosem_quad.get("bare_b") / eTF_quad^2
                ax.plot(
                    k_kf_grid_quad,
                    c1b_quad_dimless;
                    color="k",
                    label="\$n=$(mcparam.order)\$ (quad)",
                )
            end
            ax.plot(
                k_kf_grid,
                means,
                "o-";
                markersize=2,
                color="C0",
                label="\$n=$(mcparam.order)\$ ($solver)",
            )
            ax.fill_between(
                k_kf_grid,
                means - stdevs,
                means + stdevs;
                color="C0",
                alpha=0.4,
            )
            ax.legend(; loc="best")
            ax.set_xlabel("\$k / k_F\$")
            ax.set_ylabel("\$C^{(1b)}(\\mathbf{k}) \\,/\\, E^{2}_{\\mathrm{TF}}\$")
            ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
            plt.tight_layout()
            fig.savefig(
                "results/c1b/n=$(mcparam.order)/c1b_n=$(mcparam.order)_rs=$(mcparam.rs)_" *
                "beta_ef=$(mcparam.beta)_lambda=$(mcparam.mass2)_" *
                "neval=$(neval)_$(intn_str)$(solver).pdf",
            )
            plt.close("all")
            return
        end
    end
end

main()