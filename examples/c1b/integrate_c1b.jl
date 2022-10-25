using ElectronGas
using ElectronLiquid.UEG: ParaMC
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
    @debug "β * EF = $(mcparam.beta), β = $(mcparam.β), EF = $(mcparam.EF)" maxlog = 1

    # K-mesh for measurement
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

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if settings.expand_bare_interactions
        intn_str = "no_bare_"
    end

    # Process result on main thread
    if !isnothing(res)
        # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
        #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
        means = 2 * res.mean
        stdevs = 2 * res.stdev
        # Save to JLD2
        savename =
            "results/data/c1b_n=$(mcparam.order)_rs=$(mcparam.rs)_" *
            "beta_ef=$(mcparam.beta)_lambda=$(mcparam.mass2)_" *
            "neval=$(neval)_$(intn_str)$(solver)"
        jldopen("$savename.jld2", "a+") do f
            key = "$(short(mcparam))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (settings, mcparam, kgrid, res)
        end

        # z-score test for uniform value for this SOSEM observable
        if mcparam.order == 2
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

        if plot
            # Plot the result
            fig, ax = plt.subplots()
            if compare_bare
                # Compare with bare quadrature results (stored in Hartree a.u.);
                # since the bare result is independent of rs after non-dimensionalization, we
                # are free to mix rs of the current MC calculation with this result at rs = 2.
                # Similarly, the bare results were calculated at zero temperature (beta is arb.)
                rs_quad = 2.0
                sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=200.0.npz")
                k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
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
            ax.set_ylabel("\$C^{(1b)L0}(\\mathbf{k}) \\,/\\, E^{2}_{\\mathrm{TF}}\$")
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