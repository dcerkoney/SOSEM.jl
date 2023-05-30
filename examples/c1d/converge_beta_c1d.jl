using CodecZlib
using ElectronGas
using ElectronLiquid.UEG: ParaMC
using JLD2
using Measurements
using SOSEM
using PyCall

# For saving/loading numpy data
@pyimport numpy as np

@enum MeshType begin
    linear
    logarithmic
end

function main()
    @todo  # TODO: refactor

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

    # Settings for diagram generation
    settings = DiagGen.Settings{DiagGen.Observable}(
        DiagGen.c1d;
        min_order=2,
        max_order=2,
        verbosity=DiagGen.quiet,
        expand_bare_interactions=false,
    )

    # Measure uniform value (k = 0)
    kgrid = [0.0]

    # Settings
    alpha = 2.0
    print = -1
    plot = true
    solver = :vegas

    # Number of evals below and above kF
    neval = 1e7

    # Either use a linear or logarithmic mesh
    # mesh_type = linear::MeshType
    mesh_type = logarithmic::MeshType
    meshtypestr = (mesh_type == linear) ? "linear_" : "log2_"

    local param
    params = Vector{Float64}()
    means = Vector{Float64}()
    stdevs = Vector{Float64}()
    # Grid of dimensionless cooling param (beta / EF)
    local beta_grid
    if mesh_type == linear
        beta_grid = collect(range(2; stop=64, length=11))
    else
        beta_grid = 2 .^ (range(0; stop=14, length=15))
    end
    for beta in beta_grid
        # UEG parameters for MC integration
        param =
            ParaMC(; order=settings.max_order, rs=2.0, isDynamic=false, beta=beta, mass2=2.0)
        @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)" maxlog = 1

        # Generate the diagrams
        diagparam, diagtree, exprtree = DiagGen.build_nonlocal_fixed_order(settings)

        # Check the diagram tree
        DiagGen.checktree(diagtree, settings)

        # NOTE: We assume there is only a single root in the ExpressionTree
        @assert length(exprtree.root) == 1

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
            # means = res.mean
            # stdevs = res.stdev
            @assert length(res.mean) == length(res.stdev) == 1
            push!(params, param)
            push!(means, res.mean[1])
            push!(stdevs, res.stdev[1])
            # z-score test for uniform value for this SOSEM observable
            if param.order == 2
                exact = DiagGen.get_exact_k0(settings.observable)
                # Test standard score (z-score) of the measurement
                meas = measurement(res.mean[1], res.stdev[1])
                score = stdscore(meas, exact)
                obsstring = DiagGen.get_bare_string(settings.observable)
                # Result should be accurate to within the specified standard score (by default, 5σ)
                println("""
                        $obsstring, β / ϵF = $beta ($solver):
                         • Exact: $exact
                         • Measured: $meas
                         • Standard score: $score
                        """)
            end
        end
    end

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if settings.expand_bare_interactions == 2
        intn_str = "no_bare_"
    elseif settings.expand_bare_interactions == 1
        intn_str = "one_bare_"
    end

    # Save the results of a uniform calculation at multiple beta
    if length(means) > 1
        savename =
            "results/data/converge_beta_$(meshtypestr)c1d_" *
            "n=$(params[1].order)_rs=$(params[1].rs)_lambda=$(params[1].mass2)_" *
            "neval=$(neval)_$(intn_str)$(solver)"
        jldopen("$savename.jld2", "a+"; compress=true) do f
            # UEG.short without beta
            short_no_beta =
                join(["$(k)_$(v)" for (k, v) in sort(delete!(paraid(p), "beta"))], "_")
            key = "$(short_no_beta(params[1]))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (settings, param, kgrid, means, stdevs)
        end
    end
end

main()