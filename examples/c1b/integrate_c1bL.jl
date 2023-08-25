using CodecZlib
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

    settings = DiagGen.Settings{DiagGen.Observable}(
        DiagGen.c1bL;
        min_order=3,  # no (2,0,0) partition for this observable (Γⁱ₃ > Γ₀),
        # max_order=3,
        max_order=5,
        verbosity=DiagGen.quiet,
        expand_bare_interactions=1,  # testing single V[V_λ] scheme
        # expand_bare_interactions=0,  # V, V scheme (no re-expand)
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
        # interaction=[FeynmanDiagram.Interaction(ChargeCharge, Dynamic)],  # TODO: test RPA-type interaction
    )
    if settings.expand_bare_interactions == 1
        cfg = DiagGen.Config(settings)
        @debug "External V orders: $(cfg.V.orders)"
        @assert cfg.V.orders == ([0, 0, 0, 0], [0, 0, 0, 1])  # V_left = V[V_λ], V_right = V
    end

    # UEG parameters for MC integration
    param =
        ParaMC(; order=settings.max_order, rs=1.0, beta=40.0, mass2=1.0, isDynamic=false)
    @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)"

    # println("lambda = $(param.mass2)")

    # K-mesh for measurement
    minK = 0.2 * param.kF
    Nk, korder = 4, 4
    kgrid =
        CompositeGrid.LogDensedGrid(
            :uniform,
            [0.0, 3 * param.kF],
            [param.kF],
            Nk,
            minK,
            korder,
        ).grid
    # kgrid = [0.0]  # start with k=0 only

    # Settings
    alpha = 3.0
    print = 0
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 5e5

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
    for d in diagtrees
        DiagGen.checktree(d, settings)
    end

    # Bin external momenta, performing a single integration
    res = UEG_MC.integrate_nonlocal_with_ct(
        # settings,
        param,
        diagparams,
        exprtrees;
        kgrid=kgrid,
        alpha=alpha,
        neval=neval,
        print=print,
        solver=solver,
    )

    # Distinguish results with fixed vs re-expanded bare interaction line
    intn_str = ""
    if settings.expand_bare_interactions == 2
        intn_str = "no_bare_"
    elseif settings.expand_bare_interactions == 1
        intn_str = "one_bare_"
    end

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # Tag k=0 measurement
    kgrid_string = kgrid == [0.0] ? "k=0.0_" : ""

    # Save to JLD2 on main thread
    if !isnothing(res)
        # Convert result to dictionary
        data = UEG_MC.MeasType{Any}()
        if length(partitions) == 1
            avg, std = res.mean, res.stdev
            data[partitions[1]] = measurement.(avg, std)
        else
            for o in eachindex(partitions)
                avg, std = res.mean[o], res.stdev[o]
                data[partitions[o]] = measurement.(avg, std)
            end
        end
        savename =
            "results/data/c1bL/c1bL_$(kgrid_string)n=$(param.order)_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
            "neval=$(neval)_$(intn_str)$(solver)$(ct_string)"
        jldopen("$savename.jld2", "a+"; compress=true) do f
            key = short(param)
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (settings, kgrid, partitions, data)
            # return f[key] = (settings, param, kgrid, partitions, res)
        end
    end

    # # Save to JLD2 on main thread using new format
    # if !isnothing(res)
    #     savename =
    #         "results/data/rs=$(param.rs)_beta_ef=$(param.beta)_" *
    #         "lambda=$(param.mass2)_$(intn_str)$(solver)$(ct_string)"
    #     jldopen("$savename.jld2", "a+"; compress=true) do f
    #         key = "c1bL_$(kgrid_string)n_min=$(settings.min_order)_n_max=$(settings.max_order)_neval=$(neval)"
    #         if haskey(f, key)
    #             @warn("replacing existing data for $key")
    #             delete!(f, key)
    #         end
    #         f["$key/res"] = res
    #         f["$key/settings"] = settings
    #         f["$key/param"] = param
    #         f["$key/kgrid"] = kgrid
    #         f["$key/partitions"] = partitions
    #         return
    #     end
    # end
end

main()
