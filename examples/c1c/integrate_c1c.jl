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
        observable=DiagGen.c1c,
        n_order=2,
        verbosity=DiagGen.quiet,
        expand_bare_interactions=false,
    )

    # UEG parameters for MC integration
    param =
        ParaMC(; order=settings.n_order, rs=2.0, beta=200.0, mass2=0.1, isDynamic=false)
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
    neval = 1e8

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
        param,
        diagparam,
        exprtree;
        kgrid=kgrid,
        alpha=alpha,
        neval=neval,
        print=print,
    )
    # Extract the result on the main thread
    if !isnothing(res)
        means = res.mean
        stdevs = res.stdev
    end

    if length(means) == 1 && param.order == 2
        # Check result at single k-point
        res = measurement(means[1], stdevs[1])
        exact = -1
        score = stdscore(res, exact)
        println("Result = $(means[1]) ± $(stdevs[1])")
        println("Result - Exact (π²/8): $(res - exact)")
        println("Standard score: $(score)")
    elseif length(means) > 1 && plot
        if param.order == 2
            # Check result at k = 0
            res_k0 = measurement(means[1], stdevs[1])
            exact = -1
            score = stdscore(res_k0, exact)
            println("Result = $(means[1]) ± $(stdevs[1])")
            println("Result - Exact (π²/8): $(res_k0 - exact)")
            println("Standard score: $(score)")
        end
        # Distinguish results with fixed vs re-expanded bare interactions
        intn_str = ""
        if settings.expand_bare_interactions
            intn_str = "no_bare_"
        end
        # Save the result
        savename =
            "results/data/c1c_n=$(param.order)_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
            "neval=$(neval)_$(intn_str)$(solver)"
        # Remove old data, if it exists
        rm(savename; force=true)
        # TODO: kwargs implementation (kgrid_<solver>...)
        np.savez(
            savename;
            param=[param.order, param.rs, param.beta, param.kF, param.qTF, param.mass2],
            kgrid=kgrid,
            means=means,
            stdevs=stdevs,
        )
        # Plot the result
        fig, ax = plt.subplots()
        if compare_bare
            # Compare with bare quadrature results (stored in Hartree a.u.);
            # since the bare result is independent of rs after non-dimensionalization, we
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
            ax.plot(
                k_kf_grid_quad,
                c1c_quad_dimless,
                "k";
                label="\$n=$(param.order)\$ (quad)",
            )
        end
        ax.plot(
            k_kf_grid,
            means,
            "o-";
            markersize=2,
            color="C0",
            label="\$n=$(param.order)\$ ($solver)",
        )
        ax.fill_between(k_kf_grid, means - stdevs, means + stdevs; color="C0", alpha=0.4)
        ax.legend(; loc="best")
        ax.set_xlabel("\$k / k_F\$")
        ax.set_ylabel("\$C^{(1c)}(\\mathbf{k}) \\,/\\, E^{2}_{\\mathrm{TF}}\$")
        ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
        plt.tight_layout()
        fig.savefig(
            "results/c1c/n=$(param.order)/c1c_n=$(param.order)_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
            "neval=$(neval)_$(intn_str)$(solver).pdf",
        )
        plt.close("all")
    end
end

main()