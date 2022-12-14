using ElectronGas
using ElectronLiquid.UEG: ParaMC
using JLD2
using Measurements
using SOSEM
using PyCall

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

@enum MeshType begin
    linear
    logarithmic
end

function main()
    # Debug mode
    if isinteractive()
        ENV["JULIA_DEBUG"] = SOSEM
    end

    # Settings for diagram generation
    settings = DiagGen.Settings(;
        observable=DiagGen.c1d,
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
            ParaMC(; order=settings.max_order, rs=2.0, isDynamic=false, beta=beta, mass2=0.1)
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
    if settings.expand_bare_interactions
        intn_str = "no_bare_"
    end

    # Save the results of a uniform calculation at multiple beta
    if length(means) > 1
        savename =
            "results/data/converge_beta_$(meshtypestr)c1d_" *
            "n=$(params[1].order)_rs=$(params[1].rs)_lambda=$(params[1].mass2)_" *
            "neval=$(neval)_$(intn_str)$(solver)"
        jldopen("$savename.jld2", "a+") do f
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

        if plot
            # Plot the result
            fig, ax = plt.subplots()
            # Compare with zero-temperature quadrature result for the uniform value.
            # Since the bare result is independent of rs after non-dimensionalization, we
            # are free to mix rs of the current MC calculation with this result at rs = 2.
            # Similarly, the bare results were calculated at zero temperature (beta is arb.)
            rs_quad = 2.0
            sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=40.0.npz")
            # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
            param_quad = Parameter.atomicUnit(0, rs_quad)    # (dimensionless T, rs)
            eTF_quad = param_quad.qTF^2 / (2 * param_quad.me)
            c1d_quad_unif_dimless = sosem_quad.get("bare_d")[1] / eTF_quad^2
            # Either linear or logarithmic grid
            beta_plot = (mesh_type == linear) ? beta_grid : log2.(beta_grid)
            ax.axhline(
                DiagGen.get_exact_k0(settings.observable);
                color="k",
                label="\$T=0\$ (exact)",
            )
            ax.axhline(
                c1d_quad_unif_dimless;
                linestyle="--",
                color="gray",
                label="\$T=0\$ (quad)",
            )
            ax.plot(
                beta_plot,
                means,
                "o-";
                markersize=2,
                color="C0",
                label="\$n=$(param.order)\$ ($solver)",
            )
            ax.fill_between(
                beta_plot,
                means - stdevs,
                means + stdevs;
                color="C0",
                alpha=0.4,
            )
            ax.legend(; loc="center right")
            if mesh_type == linear
                ax.set_xlabel("\$\\beta / \\epsilon_F\$")
            else
                ax.set_xlabel("\$\\log_2(\\beta / \\epsilon_F)\$")
            end
            ax.set_ylabel("\$C^{(1d)}(\\mathbf{k}) \\,/\\, E^{2}_{\\mathrm{TF}}\$")
            ax.set_xlim(minimum(beta_plot), maximum(beta_plot))
            # ax.set_xticks(collect(range(1, stop=14, step=2)), minor=true)
            plt.tight_layout()
            fig.savefig(
                "results/c1d/n=$(param.order)/converge_beta_$(meshtypestr)c1d_n=$(param.order)_" *
                "rs=$(param.rs)_lambda=$(param.mass2)_" *
                "neval=$(neval)_$(intn_str)$(solver).pdf",
            )
            plt.close("all")
        end
    end
end

main()