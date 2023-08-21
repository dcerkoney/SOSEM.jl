using CodecZlib
using CompositeGrids
using ElectronGas
using ElectronLiquid.UEG: ParaMC, short
using FeynmanDiagram
using JLD2
using Measurements
using SOSEM
using SOSEM.DiagGen

# Measure at k = 0
const kval = 0.0

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

    # C⁽¹ᵇ⁾ᴸ non-local moment
    settings = DiagGen.Settings{DiagGen.Observable}(
        DiagGen.c1bL;
        min_order=3,  # no (2,0,0) partition for this observable (Γⁱ₃ > Γ₀),
        max_order=5,
        verbosity=DiagGen.quiet,
        # expand_bare_interactions=1,  # testing single V[V_λ] scheme
        expand_bare_interactions=0,  # testing V, V scheme (no re-expand)
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
    )
    if settings.expand_bare_interactions == 1
        cfg = DiagGen.Config(settings)
        @debug "External V orders: $(cfg.V.orders)"
        @assert cfg.V.orders == ([0, 0, 0, 0], [0, 0, 0, 1])  # V_left = V[V_λ], V_right = V
    end

    # K-mesh for measurement
    kgrid = [kval]

    # Scanning λ to check relative convergence wrt perturbation order
    lambdas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0]

    # Settings
    rs = 1.0
    beta = 40.0
    alpha = 3.0
    print = 0
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 1e9

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Build diagram and expression trees for all loop and counterterm partitions
    partitions, diagparams, diagtrees, exprtrees =
        build_nonlocal_with_ct(settings; renorm_mu=renorm_mu, renorm_lambda=renorm_lambda)

    println("Integrating partitions: $partitions")
    println("diagtrees: $diagtrees")
    println("exprtrees: $exprtrees")

    local param
    params = []
    datalist = []
    for lambda in lambdas
        # UEG parameters for MC integration
        param = ParaMC(;
            order=settings.max_order,
            rs=rs,
            beta=beta,
            mass2=lambda,
            isDynamic=false,
        )
        push!(params, param)
        @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)"

        # Bin external momenta, performing a single integration
        res = UEG_MC.integrate_nonlocal_with_ct(
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
            push!(datalist, data)
        end
    end

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

    # Save to JLD2 on main thread
    if !isempty(datalist)
        savename =
            "results/data/c1bL/c1bL_k=0_n=$(param.order)_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_neval=$(neval)_" *
            "$(intn_str)$(solver)$(ct_string)_vs_lambda"
        jldopen("$savename.jld2", "a+"; compress=true) do f
            for l in eachindex(lambdas)
                key = short(params[l])
                if haskey(f, key)
                    @warn("replacing existing data for $key")
                    delete!(f, key)
                end
                f[key] = (settings, kgrid, partitions, datalist[l])
            end
        end
    end
end

main()
