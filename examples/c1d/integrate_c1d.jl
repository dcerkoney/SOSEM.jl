using ElectronGas
using ElectronLiquid.UEG: ParaMC
using Measurements
using MPI
using SOSEM
using PyCall

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

function main()
    MPI.Init()
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = MPI.Comm_rank(mpi_comm)
    # mpi_size = MPI.Comm_size(mpi_comm)
    is_main_thread = (mpi_rank == 0)

    # Debug mode
    if isinteractive()
        ENV["JULIA_DEBUG"] = SOSEM
    end

    # Settings for diagram generation
    settings = DiagGen.Settings(;
        observable=DiagGen.c1d,
        n_order=2,
        verbosity=DiagGen.info,
        expand_bare_interactions=false,
    )

    # UEG parameters for MC integration
    # NOTE: To match units, we specify (beta / EF) = 2 * (heg_soms.beta)
    params =
        ParaMC(; order=settings.n_order, rs=0.5, isDynamic=false, beta=600.0, mass2=0.1)
    if is_main_thread
        @debug "β / EF = $(params.beta), β = $(params.β), EF = $(params.EF)" maxlog = 1
    end

    # K-mesh for measurement
    k_kf_grid = [0.0]
    # k_kf_grid = np.load("results/kgrids/kgrid_vegas_dimless_n=77_small.npy")
    # k_kf_grid = np.load("results/kgrids/kgrid_vegas_dimless_n=25_small.npy")
    kgrid = params.kF * k_kf_grid

    # Settings
    alpha = 2.5
    print = 0
    plot = true
    compare_bare = true
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 1e6

    # DiagGen config from settings
    cfg = DiagGen.Config(settings)

    # Generate the diagrams
    diag_tree, expr_tree = DiagGen.build_nonlocal(settings)

    # Check the diagram tree
    if is_main_thread
        DiagGen.checktree(diag_tree, settings)
    end

    # NOTE: We assume there is only a single root in the ExpressionTree
    @assert length(expr_tree.root) == 1

    means = Vector{Float64}()
    stdevs = Vector{Float64}()
    # Bin external momenta, performing a single integration
    res = UEG_MC.integrate_nonlocal(
        cfg,
        params,
        diag_tree,
        expr_tree;
        kgrid=kgrid,
        alpha=alpha,
        neval=neval,
        print=print,
        is_main_thread=is_main_thread,
    )
    # Extract the result on the main thread
    if !isnothing(res)
        means = res.mean
        stdevs = res.stdev
    end

    if length(means) == 1
        # Check result at single k-point
        res = measurement(means[1], stdevs[1])
        exact = pi^2 / 8
        score = stdscore(res, exact)
        println("Result = $(means[1]) ± $(stdevs[1])")
        println("Result - Exact (π²/8): $(res - exact)")
        println("Standard score: $(score)")
    elseif length(means) > 1 && plot
        # Check result at k = 0
        res_k0 = measurement(means[1], stdevs[1])
        exact = pi^2 / 8
        score = stdscore(res_k0, exact)
        println("Result = $(means[1]) ± $(stdevs[1])")
        println("Result - Exact (π²/8): $(res_k0 - exact)")
        println("Standard score: $(score)")
        # Save the result
        savename =
            "results/data/c1d_n=$(params.order)_rs=$(params.rs)_" *
            "beta_ef=$(params.beta)_lambda=$(params.mass2)_" *
            "neval=$(maxeval)_$(intn_str)$(solver)"
        # Remove old data, if it exists
        rm(savename; force=true)
        # TODO: kwargs implementation (kgrid_<solver>...)
        np.savez(
            savename;
            params=[
                params.order,
                params.rs,
                params.beta,
                params.kF,
                params.qTF,
                params.mass2,
            ],
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
            sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=40.0.npz")
            # np.load("results/data/soms_rs=$(Float64(params.rs))_beta_ef=$(params.beta).npz")
            k_kf_grid_quad = np.linspace(0.0, 6.0; num=600)
            # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
            params_quad = Parameter.atomicUnit(0, rs_quad)    # (dimensionless T, rs)
            eTF_quad = params_quad.qTF^2 / (2 * params_quad.me)
            c1d_quad_dimless = sosem_quad.get("bare_d") / eTF_quad^2
            ax.plot(
                k_kf_grid_quad,
                c1d_quad_dimless,
                "k";
                label="\$n=$(params.order)\$ (quad)",
            )
        end
        ax.plot(
            k_kf_grid,
            means,
            "o-";
            markersize=2,
            color="C0",
            label="\$n=$(params.order)\$ ($solver)",
        )
        ax.fill_between(k_kf_grid, means - stdevs, means + stdevs; color="C0", alpha=0.4)
        ax.legend(; loc="best")
        ax.set_xlabel("\$k / k_F\$")
        ax.set_ylabel("\$C^{(1d)}(\\mathbf{k}) \\,/\\, q^{4}_{\\mathrm{TF}}\$")
        ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
        plt.tight_layout()
        fig.savefig(
            "results/c1d/n=$(params.order)/c1d_n=$(params.order)_rs=$(params.rs)_" *
            "beta_ef=$(params.beta)_lambda=$(params.mass2)_" *
            "neval=$(maxeval)_$(intn_str)$(solver).pdf",
        )
        plt.close("all")
    end
end

main()