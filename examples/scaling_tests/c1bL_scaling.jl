using CodecZlib
using CompositeGrids
using ElectronGas
using ElectronLiquid.UEG: ParaMC, short
using FeynmanDiagram
using JLD2
using MCIntegration
using Measurements
using SOSEM
using SOSEM.DiagGen

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

    orders = [2, 3, 4, 5, 6]
    min_order = minimum(orders)
    max_order = maximum(orders)

    # Composite observable; measure all non-local moments together
    settings = Settings{Observable}(
        c1bL;
        min_order=min_order,
        max_order=max_order,
        verbosity=quiet,
        expand_bare_interactions=1,  # single V[V_λ] scheme
        # expand_bare_interactions=0,  # re-expansionless scheme
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
    )
    @assert c1nl_ueg.observables == [c1bL0, c1bL, c1c, c1d]

    # UEG parameters for MC integration
    param = ParaMC(;
        order=settings.max_order,
        rs=1.0,
        beta=40.0,
        mass2=1.0,
        isDynamic=false,
        isFock=false,
    )
    @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)"

    # We measure the uniform (k=0) non-local moment
    extK = 0.0

    # Settings
    alpha = 3.0
    print = 0
    solver = :vegasmc

    # Check scaling wrt neval
    nevals = [1e5, 1e6, 1e7, 1e8]

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Distinguish results with fixed vs re-expanded bare interactions
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

    # Build diagram and expression trees for all loop and counterterm partitions
    partitions, diagparams, diagtrees, exprtrees = build_nonlocal_with_ct(
        # settings_list;
        settings;
        renorm_mu=renorm_mu,
        renorm_lambda=renorm_lambda,
    )

    println("Integrating partitions: $partitions")
    println("diagtrees: $diagtrees")
    println("exprtrees: $exprtrees")
    # @debug "diagtrees: $diagtrees"
    # @debug "exprtrees: $exprtrees"

    # # Check the diagram tree
    # for d in diagtrees
    #     checktree(d, settings[1])
    # end

    results = Result[]
    for neval in nevals
        # Bin external momenta, performing a single integration
        res = UEG_MC.integrate_nonlocal_with_ct_fixed_extK(
            param,
            diagparams,
            exprtrees;
            extK=extK,
            alpha=alpha,
            neval=neval,
            print=print,
            solver=solver,
        )
        # Push to results vector on main thread
        if !isnothing(res)
            push!(results, res)
        end
    end
    # Save to JLD2 on main thread
    if !isempty(results)
        savename =
            "results/data/c1bL_k=$(extK)_scaling_nevals=$(nevals)_n=$(param.order)_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_lambda=$(param.mass2)_$(intn_str)$(solver)$(ct_string)"
        jldopen("$savename.jld2", "a+"; compress=true) do f
            key = "$(short(param))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (settings, orders, param, extK, partitions, results)
        end
    end
end

main()
