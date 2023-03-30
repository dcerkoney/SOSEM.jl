using CodecZlib
using DataFrames
using DelimitedFiles
using ElectronGas
using ElectronLiquid
using Interpolations
using JLD2
using Lehmann
using LsqFit
using Measurements
using Parameters
using Polynomials
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt
@pyimport mpl_toolkits.axes_grid1.inset_locator as il

"""Returns the static structure factor S(q) of the UEG."""
function static_structure_factor(q, param::ParaMC)
    x = q / param.kF
    if x < 2
        return 3x / 4.0 - x^3 / 16.0
    end
    return 1.0
end

"""Π₀(q, τ=0) = -n₀ S(q)"""
function bare_polarization_exact_t0(q, param::ParaMC)
    n0 = param.kF^3 / 3π^2
    return -n0 * static_structure_factor(q, param)
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
    mass2 = 1.0
    solver = :vegasmc

    # Number of evals
    neval = 5e10

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order = 1
    max_order = 2
    min_order_plot = 1
    max_order_plot = 2

    # Save total results
    save = true

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Ignore measured mu/lambda partitions?
    fix_mu = false
    fix_lambda = false
    fix_string = fix_mu || fix_lambda ? "_fix" : ""
    if fix_mu
        fix_string *= "_mu"
    end
    if fix_lambda
        fix_string *= "_lambda"
    end

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # UEG parameters for MC integration
    loadparam = ParaMC(;
        order=max_order,
        rs=rs,
        beta=beta,
        mass2=mass2,
        isDynamic=false,
        isFock=false,
    )

    # Load the raw data
    savename =
        "results/data/polarization_n=$(max_order)_rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(mass2)_neval=$(neval)_$(solver)$(ct_string)"
    # TODO: Rerun with new format,
    #   orders, param, kgrid, tgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
    orders, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(loadparam))"
        return f[key]
    end

    # T-mesh for measurement; we need a fine τ-grid for accurate integration
    n_tau = 1000
    tgrid = collect(LinRange(1e-8, param.β - 1e-8, n_tau))

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    # Get dimensionless t-grid (τ / β)
    tau_beta_grid = tgrid / param.β

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    for (k, v) in data
        data[k] = v / (factorial(k[2]) * factorial(k[3]))
    end
    # Zero out partitions with mu renorm if present (fix mu)
    if renorm_mu == false || fix_mu
        for P in keys(data)
            if P[2] > 0
                println("Fixing mu without lambda renorm, ignoring n_k partition $P")
                data[P] = zero(data[P])
            end
        end
    end
    # Zero out partitions with lambda renorm if present (fix lambda)
    if renorm_lambda == false || fix_lambda
        for P in keys(data)
            if P[3] > 0
                println("No lambda renorm, ignoring n_k partition $P")
                data[P] = zero(data[P])
            end
        end
    end

    println(typeof(data))
    for P in keys(data)
        # Is there a missing sign for Π2?
        # sum(P) == 2 && continue
        # TODO: Where does this extra overall sign come from, N&O definition of Π?
        data[P] *= -1
        # NOTE: Adding missing factor of β back to old data (now fixed in integration script)
        data[P] /= param.β
    end

    merged_data = UEG_MC.mergeInteraction(data)
    println(typeof(merged_data))

    # Get exact instantaneous bare polarization Π₀(q, τ=0)
    pi0_t0 = bare_polarization_exact_t0.(kgrid, [param])

    # Set bare result manually using exact function
    if haskey(merged_data, (1, 0)) == false
        # treat quadrature data as numerically exact
        merged_data[(1, 0)][:, 1] = measurement.(pi0_t0, 0.0)
    elseif min_order == 1
        stdscores = stdscore.(merged_data[(1, 0)][:, 1], pi0_t0)
        worst_score = argmax(abs, stdscores)
        println("Worst standard score for N=1 (measured): $worst_score")
        # @assert worst_score ≤ 10
    end

    # Reexpand merged data in powers of μ
    ct_filename = "examples/counterterms/data_Z$(ct_string).jld2"
    z, μ = UEG_MC.load_z_mu(param; ct_filename=ct_filename)
    # Add Taylor factors to CT data
    for (p, v) in z
        z[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    for (p, v) in μ
        μ[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    # Zero out partitions with mu renorm if present (fix mu)
    if renorm_mu == false || fix_mu
        for P in keys(μ)
            if P[2] > 0
                println("Fixing mu without lambda renorm, ignoring μ partition $P")
                μ[P] = zero(μ[P])
            end
        end
    end
    # Zero out partitions with lambda renorm if present (fix lambda)
    if renorm_lambda == false || fix_lambda
        for P in keys(μ)
            if P[3] > 0
                println("No lambda renorm, ignoring μ partition $P")
                μ[P] = zero(μ[P])
            end
        end
    end
    δz, δμ = CounterTerm.sigmaCT(max_order, μ, z; isfock=false, verbose=1)

    println("Computed δμ: ", δμ)
    polarization = UEG_MC.chemicalpotential_renormalization_poln(
        merged_data,
        δμ;
        min_order=1,
        max_order=max_order,
    )
    δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
    poln_2_manual = merged_data[(2, 0)] + δμ1_exact * merged_data[(1, 1)]
    scores_2 = stdscore.(polarization[2] - poln_2_manual, 0.0)
    worst_score_2 = argmax(abs, scores_2)
    println("2nd order renorm vs manual worst score: $worst_score_2")

    println(UEG.paraid(param))
    println(partitions)
    println(typeof(polarization))

    # Aggregate the full results for Σₓ up to order N
    polarization_total = UEG_MC.aggregate_orders(polarization)

    if save
        savename =
            "results/data/rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_$(intn_str)$(solver)$(ct_string)"
        f = jldopen("$savename.jld2", "a+"; compress=true)
        # NOTE: no bare result for c1b observable (accounted for in c1b0)
        for N in min_order_plot:max_order
            num_eval = neval
            # Update existing results if applicable
            if haskey(f, "polarization") &&
               haskey(f["polarization"], "N=$N") &&
               haskey(f["polarization/N=$N"], "neval=$num_eval")
                @warn("replacing existing data for N=$N, neval=$num_eval")
                delete!(f["polarization/N=$N"], "neval=$num_eval")
            end
            f["polarization/N=$N/neval=$num_eval/meas"] = polarization_total[N]
            f["polarization/N=$N/neval=$num_eval/param"] = param
            f["polarization/N=$N/neval=$num_eval/kgrid"] = kgrid
            f["polarization/N=$N/neval=$num_eval/tgrid"] = tgrid
        end
    end

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    qgrid_fine = param.kF * np.linspace(0.0, 3.0; num=600)
    q_2kf_grid_fine = np.linspace(0.0, 1.5; num=600)
    pi0_t0_fine = bare_polarization_exact_t0.(qgrid_fine, [param])
    n0 = param.kF^3 / 3π^2

    # Plot the instantaneous polarization for each aggregate order
    fig, ax = plt.subplots()
    ax.axvline(1.0; linestyle="--", linewidth=1, color="gray")
    ax.axhline(1.0; linestyle="--", linewidth=1, color="gray")
    # ax.axhline(n0; linestyle="--", linewidth=1, color="k", label="\$n_0\$")
    if min_order_plot == 1
        # Include exact instantaneous bare polarization in plot
        ax.plot(q_2kf_grid_fine, -pi0_t0_fine / n0, "k"; label="\$N=1\$ (exact)")
    end
    ic = 1
    colors = ["C0", "C1", "C2", "C3"]
    # colors = ["orchid", "cornflowerblue", "turquoise", "chartreuse"]
    for (i, N) in enumerate(min_order:max_order_plot)
        # N == 0 && continue
        # Plot measured data at first τ-grid point (τ = 1e-8)
        # TODO: Fix factor of 1/β
        poln_means = Measurements.value.(polarization_total[N][:, 1])
        poln_stdevs = Measurements.uncertainty.(polarization_total[N][:, 1])
        # S(q) = -Π(q, τ=0) / n₀
        static_structure_means = -poln_means / n0
        static_structure_stdevs = poln_stdevs / n0
        println(poln_means)
        marker = "o-"
        ax.plot(
            k_kf_grid / 2.0,  # q / 2kF
            static_structure_means,
            marker;
            markersize=2,
            color="$(colors[ic])",
            label="\$N=$N\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid / 2.0,  # q / 2kF
            static_structure_means - static_structure_stdevs,
            static_structure_means + static_structure_stdevs;
            color="$(colors[ic])",
            alpha=0.4,
        )
        ic += 1
        # Compare to results with sign flip on Π2
        if N == 2
            poln_2 = Measurements.value.(polarization[2][:, 1])
            sign_flip_means = static_structure_means + 2 * poln_2 / n0
            sign_flip_stdevs = poln_stdevs
            ax.plot(
                k_kf_grid / 2.0,  # q / 2kF
                sign_flip_means,
                marker;
                markersize=2,
                color="$(colors[ic])",
                label="\$N=$N\$ ($solver + sign flip)",
            )
            ax.fill_between(
                k_kf_grid / 2.0,  # q / 2kF
                sign_flip_means - sign_flip_stdevs,
                sign_flip_means + sign_flip_stdevs;
                color="$(colors[ic])",
                alpha=0.4,
            )
            ic += 1
        end
    end
    ax.legend(; loc="best")
    # ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    # ax.set_xlim(0.75, 1.25)
    ax.set_xlim(0, 1.5)
    # ax.set_ylim(nothing, 2)
    ax.set_xlabel("\$q / 2 k_F\$")
    ax.set_ylabel("\$S(q \\ne 0) = -\\Pi(q, \\tau=0) / n_0\$")
    # xloc = 1.5
    xloc = 0.225
    yloc = 0.2
    ydiv = -0.125
    ax.text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(
        "results/polarization/static_structure_factor_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver)$(ct_string)$(fix_string).pdf",
    )

    # Plot the polarization at a few Matsubara times
    # TODO

    # Plot the q=0 polarization vs τ
    # TODO

    plt.close("all")
    return
end

main()
