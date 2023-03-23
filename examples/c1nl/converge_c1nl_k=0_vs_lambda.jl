using CodecZlib
using CompositeGrids
using ElectronGas
using ElectronLiquid.UEG: ParaMC, short
using FeynmanDiagram
using JLD2
using Measurements
using SOSEM
using SOSEM.DiagGen

function paraid_no_lambda(p::ParaMC)
    return Dict(
        "dim" => p.dim,
        "rs" => p.rs,
        "beta" => p.beta,
        "Fs" => p.Fs,
        "Fa" => p.Fa,
        "massratio" => p.massratio,
        "spin" => p.spin,
        "isFock" => p.isFock,
        "isDynamic" => p.isDynamic,
    )
end

function short_no_lambda(p::ParaMC)
    return join(["$(k)_$(v)" for (k, v) in sort(paraid_no_lambda(p))], "_")
end

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

    # Composite observable; measure all non-local moments together
    settings = Settings{CompositeObservable}(
        c1nl_ueg;
        min_order=2,  # TODO: special-purpose integrator for (2,0,0) partition
        max_order=4,
        verbosity=quiet,
        expand_bare_interactions=false,
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
    )
    @assert c1nl_ueg.observables == [c1bL0, c1bL, c1c, c1d]

    # Scanning λ to check relative convergence wrt perturbation order
    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]

    # K-mesh for measurement
    kgrid = [0.0]

    # Settings
    rs = 1.0
    beta = 40.0
    alpha = 3.0
    print = 0
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 1e7

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

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

    local res, param
    params = []
    res_list = []
    for lambda in lambdas
        # UEG parameters for MC integration
        param = ParaMC(;
            order=settings.max_order,
            rs=rs,
            beta=beta,
            mass2=lambda,
            isDynamic=false,
            isFock=false,
        )
        push!(params, param)
        @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)"

        # Bin external momenta, performing a single integration
        res = UEG_MC.integrate_full_nonlocal_with_ct(
            param,
            diagparams,
            exprtrees;
            kgrid=kgrid,
            alpha=alpha,
            neval=neval,
            print=print,
            solver=solver,
        )
        if !isnothing(res)
            push!(res_list, res)
        end
    end

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if settings.expand_bare_interactions
        intn_str = "no_bare_"
    end

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # Save to JLD2 on main thread
    if !isnothing(res)
        savename =
            "results/data/c1nl_k=0_n=$(param.order)_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_neval=$(neval)_" *
            "$(intn_str)$(solver)$(ct_string)_vs_lambda"
        jldopen("$savename.jld2", "a+"; compress=true) do f
            key = "$(short_no_lambda(param))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (settings, params, kgrid, lambdas, partitions, res_list)
        end
    end
end

main()
