using ElectronLiquid.UEG: ParaMC
using MPI
using SOSEM
using Plots
using PyCall

# For loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

function main()
    MPI.Init()
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = MPI.Comm_rank(mpi_comm)
    mpi_size = MPI.Comm_size(mpi_comm)
    is_main_thread = (mpi_rank == 0)

    # Debug mode
    if isinteractive()
        ENV["JULIA_DEBUG"] = SOSEM
    end

    # Settings for diagram generation
    settings = DiagGen.Settings(; observable=DiagGen.c1d, n_order=2, verbosity=DiagGen.info)

    # UEG parameters for MC integration
    # NOTE: To match units, we specify (beta / EF) = 2 * (heg_soms.beta)
    params = ParaMC(; rs=2.0, isDynamic=false, beta=40.0, mass2=0.0001)
    if is_main_thread
        @debug "β / EF = $(params.beta), β = $(params.β), EF = $(params.EF)" maxlog = 1
    end

    # K-mesh for measurement
    # kgrid = [params.kF]
    k_kf_grid = np.load("results/kgrid_vegas_dimless_n=25.npy")
    kgrid = params.kF * k_kf_grid
    n_kgrid = length(kgrid)

    alpha = 3.0
    print = 0
    # neval = 1e5

    # Plot in post-processing instead
    plot = false

    # Number of evals below and above kF
    neval_le_kf = 1e6
    neval_gt_kf = 1e6
    maxeval = max(neval_le_kf, neval_gt_kf)

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

    # # Bin external momenta (single integral)
    # neval = maxeval * n_kgrid
    # binned_res = UEG_MC.integrate_nonlocal(
    #     cfg,
    #     params,
    #     diag_tree,
    #     expr_tree;
    #     kgrid=[k],
    #     alpha=alpha,
    #     neval=neval,
    #     print=print,
    #     is_main_thread=is_main_thread,
    # )

    # Loop over external momenta and integrate
    means = Vector{Float64}()
    stdevs = Vector{Float64}()
    for (ik, k) in enumerate(kgrid)
        if is_main_thread
            println("ik = $ik / $(n_kgrid)")
        end
        neval = (k ≤ params.kF) ? neval_le_kf : neval_gt_kf
        res = UEG_MC.integrate_nonlocal(
            cfg,
            params,
            diag_tree,
            expr_tree;
            kgrid=[k],
            alpha=alpha,
            neval=neval,
            print=print,
            is_main_thread=is_main_thread,
        )
        # Append the result on the main thread
        if !isnothing(res)
            append!(means, res.mean)
            append!(stdevs, res.stdev)
        end
    end

    if !isempty(means) && plot
        # Save the result
        np.savez(
            "results/data/c1d_rs=$(params.rs)_beta_ef=$(params.beta)_neval=$(maxeval)";
            params=[params.order, params.rs, params.beta, params.kF, params.qTF],
            kgrid=kgrid,
            means=means,
            stdevs=stdevs,
        )

        # Compare with quadrature results (stored in Hartree a.u.)
        sosem_quad =
            np.load("results/data/soms_rs=$(Float64(params.rs))_beta_ef=$(params.beta).npz")
        k_kf_grid_quad = np.linspace(0.0, 6.0; num=600)
        # NOTE: (q_TF a₀) is dimensionless, hence q_TF  is the same in Rydberg
        #       and Hartree a.u., and no additional conversion factor is needed
        c1d_quad_dimless = sosem_quad.get("bare_d") / params.qTF^4

        # Plot the result
        fig, ax = plt.subplots()
        ax.plot(k_kf_grid_quad, c1d_quad_dimless, "k"; label="\$n=$(params.order)\$ (quad)")
        ax.plot(k_kf_grid, means, "o-"; color="C0", label="\$n=$(params.order)\$ (vegas)")
        ax.fill_between(k_kf_grid, means - stdevs, means + stdevs; color="C0", alpha=0.4)
        ax.legend(; loc="best")
        ax.set_xlabel("\$k / k_F\$")
        ax.set_ylabel("\$C^{(1d)}(\\mathbf{k}) / q^{4}_{\\mathrm{TF}}\$")
        ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
        plt.tight_layout()
        fig.savefig(
            "results/c1d_n=$(params.order)_rs=$(params.rs)_" *
            "beta_ef=$(params.beta)_neval=$(maxeval)_dimless.pdf",
        )
        plt.close("all")
    end
end

main()