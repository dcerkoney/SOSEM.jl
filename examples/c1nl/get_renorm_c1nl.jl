using ElectronLiquid
using ElectronGas
using FeynmanDiagram
using Interpolations
using JLD2
using MCIntegration
using Measurements
using SOSEM
using PyCall

# For saving/loading numpy data
@pyimport numpy as np

# NOTE: Call from main project directory as: julia examples/c1nl/get_renorm_c1nl_k=0.jl

"""Renormalize the raw data for the given SOSEM measurement class and parameters."""
function renorm_measurement(
    class="c1bL0";
    jldformat="old",
    min_order=2,
    max_order,
    rs,
    beta,
    mass2,
    neval,
    solver=:vegasmc,
    expand_bare_interactions=false,
    renorm_mu=true,
    renorm_lambda=true,
    save=false,
)
    println("Renormalizing class $class data...")

    class_savenames =
        Dict("c1bL0" => "c1b0", "c1bL" => "c1b", "c1c" => "c1c", "c1d" => "c1d")

    # No 2nd order data for class c1bL measurement
    class == "c1bL" && @assert min_order > 2

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions
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

    if jldformat == "new"
        min_order_mc = min_order > 2 ? min_order : 3
        # Load the results using new JLD2 format
        filename =
            "results/data/rs=$(rs)_beta_ef=$(beta)_" *
            "lambda=$(mass2)_$(intn_str)$(solver)_$(ct_string)"
        f = jldopen("$filename.jld2", "r")
        key = "c1d_n_min=$(min_order_mc)_n_max=$(max_order)_neval=$(neval)"
        res = f["$key/res"]
        settings = f["$key/settings"]
        param = f["$key/param"]
        kgrid = f["$key/kgrid"]
        partitions = f["$key/partitions"]
        # Close the JLD2 file
        close(f)
    else
        # Load the results using old JLD2 format
        loadparam =
            UEG.ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)
        savename =
            "results/data/$(class)_n=$(max_order)_rs=$(rs)_" *
            "beta_ef=$(beta)_lambda=$(mass2)_" *
            "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)"
        settings, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
            key = "$(UEG.short(loadparam))"
            return f[key]
        end
    end

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])

    # Non-dimensionalize bare non-local moment
    rs_lo = 1.0
    sosem_lo = np.load("results/data/soms_rs=$(rs_lo)_beta_ef=40.0.npz")
    # Non-dimensionalize quadrature results by Thomas-Fermi energy
    param_lo = Parameter.atomicUnit(0, rs_lo)    # (dimensionless T, rs)
    eTF_lo = param_lo.qTF^2 / (2 * param_lo.me)

    # Fine kgrid for LO results
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)

    # Bare uniform results (stored in Hartree a.u.)
    if class == "c1bL0"
        # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
        #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
        c1class_bare_quad = sosem_lo.get("bare_b") / eTF_lo^2 / 2
    elseif class == "c1c"
        c1class_bare_quad = sosem_lo.get("bare_c") / eTF_lo^2
    elseif class == "c1d"
        c1class_bare_quad = sosem_lo.get("bare_d") / eTF_lo^2
    end

    # Get renormalized data
    if min_order == 2 && class != "c1bL"
        # Interpolate bare results and downsample to coarse k_kf_grid
        c1class_bare_interp =
            linear_interpolation(k_kf_grid_quad, c1class_bare_quad; extrapolation_bc=Line())
        c1class_2_exact = c1class_bare_interp(k_kf_grid)
        # Set bare results manually using exact data to avoid systematic error in (2,0,0) calculation
        merged_data[(2, 0)] = measurement.(c1class_2_exact, 0.0)  # treat quadrature data as numerically exact
    end
    if renorm_mu
        # The lowest order is 2 for all measurements except
        # for the vertex-corrected diagrams in class c1bL
        lowest_order = class == "c1bL" ? 3 : 2
        # Reexpand merged data in powers of μ
        z, μ = UEG_MC.load_z_mu(param)
        δz, δμ = CounterTerm.sigmaCT(max_order - 2, μ, z; verbose=1)
        println("Computed δμ: ", δμ)
        c1class = UEG_MC.chemicalpotential_renormalization_sosem(
            merged_data,
            δμ;
            lowest_order=lowest_order,
            min_order=min_order,
            max_order=max_order,
        )
    else
        c1class = merged_data
    end

    # Aggregate the full renormalized results for C⁽¹ᶜ⁾ up to order N
    c1class_total = UEG_MC.aggregate_orders(c1class)

    println(settings)
    println(UEG.paraid(param))
    println(res)
    println(partitions)
    println(c1class_total)

    if save
        savename =
            "results/data/rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_$(intn_str)$(solver)_$(ct_string)"
        f = jldopen("$savename.jld2", "a+"; compress=true)
        # Get savename for this measurement class
        class_savename = class_savenames[class]
        for N in min_order:max_order
            # NOTE: no bare result for c1b observable (accounted for in c1b0)
            if N == 2 && class == "c1bL"
                continue
            end
            if haskey(f, class_savename) &&
               haskey(f[class_savename], "N=$(N)_unif") &&
               haskey(f["$class_savename/N=$(N)_unif"], "neval=$(neval)")
                @warn("replacing existing data for N=$(N)_unif, neval=$(neval)")
                delete!(f["$class_savename/N=$(N)_unif"], "neval=$(neval)")
            end
            # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
            #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
            meas = class in ["c1bL0", "c1bL"] ? 2 * c1class_total[N] : c1class_total[N]
            f["$class_savename/N=$(N)_unif/neval=$neval/meas"] = meas
            f["$class_savename/N=$(N)_unif/neval=$neval/settings"] = settings
            f["$class_savename/N=$(N)_unif/neval=$neval/param"] = param
            f["$class_savename/N=$(N)_unif/neval=$neval/kgrid"] = kgrid
        end
        # Close the JLD2 file
        close(f)
    end
    return println("Done!\n")
end

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    rs = 1.0
    beta = 40.0
    neval = 1e10
    solver = :vegasmc

    lambdas = [0.5, 1.0, 1.5, 2.0, 3.0]
    # lambdas = [2.0]

    max_order = 4
    @assert max_order ≥ 3

    # Save total results
    save = true

    # Renormalize uniform measurements
    for lambda in lambdas
        renorm_measurement(
            "c1bL0";
            min_order=2,
            max_order=max_order,
            rs=rs,
            beta=beta,
            mass2=lambda,
            neval=neval,
            save=save,
        )
        renorm_measurement(
            "c1bL";
            min_order=3,
            max_order=max_order,
            rs=rs,
            beta=beta,
            mass2=lambda,
            neval=neval,
            save=save,
        )
        renorm_measurement(
            "c1c";
            min_order=2,
            max_order=max_order,
            rs=rs,
            beta=beta,
            mass2=lambda,
            neval=neval,
            save=save,
        )
        renorm_measurement(
            "c1d";
            jldformat="new",
            min_order=2,
            max_order=max_order,
            rs=rs,
            beta=beta,
            mass2=lambda,
            neval=neval,
            save=save,
        )
    end

    return
end

main()
