using ElectronGas
using ElectronLiquid.UEG: ParaMC, short
using FeynmanDiagram
using JLD2
using SOSEM

@enum MeshType begin
    linear
    logarithmic
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

    settings = DiagGen.Settings(;
        observable=DiagGen.c1bL,
        min_order=3,  # no (2,0,0) partition for this observable (Γⁱ₃ > Γ₀),
        max_order=3,
        verbosity=DiagGen.quiet,
        expand_bare_interactions=false,
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
        # interaction=[FeynmanDiagram.Interaction(ChargeCharge, Dynamic)],  # TODO: test RPA-type interaction
    )

    # Measure uniform value (k = 0)
    kgrid = [0.0]

    # Physical params
    rs = 1.0
    mass2 = 2.0

    # Settings
    alpha = 2.0
    print = 0
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 1e10

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Either use a linear or logarithmic mesh
    mesh_type = linear::MeshType
    # mesh_type = logarithmic::MeshType
    meshtypestr = (mesh_type == linear) ? "linear_" : "log2_"

    results = []
    params = []
    # Grid of dimensionless cooling param (beta / EF)
    local beta_grid, param, partitions
    if mesh_type == linear
        beta_grid = collect(range(20; stop=200, length=10))
    else
        beta_grid = 2 .^ (range(0; stop=14, length=15))
    end
    for beta in beta_grid
        # UEG parameters for MC integration
        param = ParaMC(;
            order=settings.max_order,
            rs=rs,
            isDynamic=false,
            beta=beta,
            mass2=mass2,
        )
        @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)" maxlog = 1

        # Build diagram and expression trees for all loop and counterterm partitions
        partitions, diagparams, diagtrees, exprtrees = DiagGen.build_nonlocal_with_ct(
            settings;
            renorm_mu=renorm_mu,
            renorm_lambda=renorm_lambda,
        )

        println("Integrating partitions: $partitions")
        @debug "diagtrees: $diagtrees"
        @debug "exprtrees: $exprtrees"

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

        # Append the results for this beta to lists on the main thread
        if !isnothing(res)
            push!(results, res)
            push!(params, param)
        end
    end

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
    if !isempty(results)
        savename =
            "results/data/converge_beta_$(meshtypestr)c1bL_n=$(param.order)_" *
            "rs=$(param.rs)_lambda=$(param.mass2)_" *
            "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)"
        jldopen("$savename.jld2", "a+") do f
            key = "$(short(param))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (settings, param, kgrid, partitions, results)
        end
    end
end

main()
