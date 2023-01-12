using CompositeGrids
using ElectronGas
using ElectronLiquid.UEG: ParaMC, short
using FeynmanDiagram
using JLD2
using Measurements
using SOSEM

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    # Debug mode
    if isinteractive()
        ENV["JULIA_DEBUG"] = SOSEM
    end

    settings = DiagGen.Settings(;
        observable=DiagGen.c1bL0,
        min_order=3,  # TODO: special-purpose integrator for (2,0,0) partition
        max_order=4,
        verbosity=DiagGen.quiet,
        expand_bare_interactions=false,
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
        # interaction=[FeynmanDiagram.Interaction(ChargeCharge, Dynamic)],  # TODO: test RPA-type interaction
    )

    # TODO: Write special-purpose integrator for bare result; currently using the old numerically
    #       exact results from quadrature, wich are valid for `expand_bare_interactions=false` only!
    @assert settings.expand_bare_interactions == false

    # UEG parameters for MC integration
    param =
        ParaMC(; order=settings.max_order, rs=1.0, beta=40.0, mass2=2.0, isDynamic=false)
    @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)"

    # K-mesh for measurement
    kgrid = [0.0]
    # minK = 0.2 * param.kF
    # Nk, korder = 4, 7
    # kgrid =
    #     CompositeGrid.LogDensedGrid(
    #         :uniform,
    #         [0.0, 3 * param.kF],
    #         [param.kF],
    #         Nk,
    #         minK,
    #         korder,
    #     ).grid

    # Settings
    alpha = 3.0
    print = 0
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 5e10

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
            "results/data/c1bL0_n=$(param.order)_rs=$(param.rs)_" *
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
    end
end

main()
