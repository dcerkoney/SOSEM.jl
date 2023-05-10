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

"""Returns the static structure factor S₀(q) of the UEG in the HF approximation."""
function static_structure_factor_hf(q, param::ParaMC)
    x = q / param.kF
    if x < 2
        return 3x / 4.0 - x^3 / 16.0
    end
    return 1.0
end

"""Π₀(q, τ=0) = χ₀(q, τ=0) = -n₀ S₀(q)"""
function bare_susceptibility_exact_t0(q, param::ParaMC)
    n0 = param.kF^3 / 3π^2
    return -n0 * static_structure_factor_hf(q, param)
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
    neval = 1e7

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order = 1
    max_order = 3
    min_order_plot = 1
    max_order_plot = 3

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
        "results/data/static_structure_factor_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval)_$(solver)$(ct_string)"
    # TODO: Rerun with new format,
    #   orders, param, kgrid, tgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
    orders, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(loadparam))"
        return f[key]
    end

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    # Non-interacting density
    n0 = param.kF^3 / 3π^2

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
        # Convert back to Mahan convention: χ_N&O = -χ_Mahan
        data[P] *= -1
    end

    merged_data = UEG_MC.mergeInteraction(data)
    println(typeof(merged_data))

    # Get exact Hartree-Fock static structure factor S₀(q) = -Π₀(q, τ=0) / n₀
    static_structure_hf_exact = static_structure_factor_hf.(kgrid, [param])

    # Set bare result manually using exact function
    # if haskey(merged_data, (1, 0)) == false
    if min_order > 1
        # treat quadrature data as numerically exact
        merged_data[(1, 0)] = measurement.(static_structure_hf_exact, 0.0)
    elseif min_order == 1
        stdscores = stdscore.(merged_data[(1, 0)], static_structure_hf_exact)
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
    static_structure = UEG_MC.chemicalpotential_renormalization_susceptibility(
        merged_data,
        δμ;
        min_order=1,
        max_order=max_order,
    )
    δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
    inst_poln_2_manual = merged_data[(2, 0)] + δμ1_exact * merged_data[(1, 1)]
    scores_2 = stdscore.(static_structure[2] - inst_poln_2_manual, 0.0)
    worst_score_2 = argmax(abs, scores_2)
    println("2nd order renorm vs manual worst score: $worst_score_2")

    println(UEG.paraid(param))
    println(partitions)
    println(typeof(static_structure))

    # Aggregate the full results for Σₓ up to order N
    static_structure_total = UEG_MC.aggregate_orders(static_structure)

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    qgrid_fine = param.kF * np.linspace(0.0, 3.0; num=600)
    q_kf_grid_fine = np.linspace(0.0, 3.0; num=600)
    static_structure_hf_exact_fine = static_structure_factor_hf.(qgrid_fine, [param])

    # Plot the static structure factor for each aggregate order
    fig, ax = plt.subplots()
    ax.axvline(2.0; linestyle="--", linewidth=1, color="gray")
    ax.axhline(1.0; linestyle="--", linewidth=1, color="gray")
    # ax.axhline(n0; linestyle="--", linewidth=1, color="k", label="\$n_0\$")
    if min_order_plot == 1
        # Include exact Hartree-Fock static structure factor in plot
        ax.plot(q_kf_grid_fine, static_structure_hf_exact_fine, "k"; label="\$N=1\$ (exact)")
    end
    ic = 1
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    # colors = ["orchid", "cornflowerblue", "turquoise", "chartreuse"]
    for (i, N) in enumerate(min_order:max_order_plot)
        # S(q) = -Π(q, τ=0) / n₀
        static_structure_means = Measurements.value.(static_structure_total[N])
        static_structure_stdevs = Measurements.uncertainty.(static_structure_total[N])
        marker = "o-"
        ax.plot(
            k_kf_grid,
            static_structure_means,
            marker;
            markersize=2,
            color="$(colors[ic])",
            label="\$N=$N\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid,
            static_structure_means - static_structure_stdevs,
            static_structure_means + static_structure_stdevs;
            color="$(colors[ic])",
            alpha=0.4,
        )
        ic += 1
    end
    ax.legend(; loc="best")
    ax.set_xlim(0, 3.0)
    # ax.set_ylim(nothing, 2)
    ax.set_xlabel("\$q / k_F\$")
    ax.set_ylabel("\$S(q) = -\\chi(q, \\tau=0) / n_0\$")
    # ax.set_ylabel("\$S(q)\$")
    # xloc = 1.5
    xloc = 0.65
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
        "results/static_structure_factor/static_structure_factor_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver)$(ct_string)$(fix_string).pdf",
    )

    plt.close("all")
    return
end

main()
