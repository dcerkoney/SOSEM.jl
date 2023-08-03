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
using PyPlot
using SOSEM

# For style "science"
@pyimport scienceplots

# For saving/loading numpy data
@pyimport numpy as np
@pyimport scipy.interpolate as interp

# @pyimport matplotlib.pyplot as plt
# @pyimport mpl_toolkits.axes_grid1.inset_locator as il

# Vibrant qualitative colour scheme from https://personal.sron.nl/~pault/
const cdict = Dict([
    "orange" => "#EE7733",
    "blue" => "#0077BB",
    "cyan" => "#33BBEE",
    "magenta" => "#EE3377",
    "red" => "#CC3311",
    "teal" => "#009988",
    "grey" => "#BBBBBB",
]);

function spline(x, y, e)
    # generate knots with spline without constraints
    w = 1.0 ./ e
    spl = interp.UnivariateSpline(x, y; w=w, k=3)
    __x = collect(LinRange(x[1], x[end], 1000))
    yfit = spl(__x)
    return __x, yfit
end

function spline_with_bc(x, y, e)
    w = 1.0 ./ e
    _x, _y = deepcopy(x[2:end]), deepcopy(y[2:end])
    _w = 1.0 ./ e[2:end]

    #enforce left boundary condition: the value at k=0 is zero
    rescale = 10000
    pushfirst!(_x, 0.0)
    pushfirst!(_y, 0.0)
    pushfirst!(_w, _w[1] * rescale)

    #enforce right boundary condition: the derivative at k=kmax is zero
    dx = 0.01
    rescale = 10000
    _w[end] *= rescale
    push!(_x, _x[end] + dx)
    push!(_y, _y[end])
    push!(_w, _w[end])

    # generate knots with spline without constraints
    spl = interp.UnivariateSpline(_x, _y; w=_w, k=3)
    __x = collect(LinRange(0.0, x[end], 1000))
    yfit = spl(__x)
    return __x, yfit
end

"""Returns the static structure factor S₀(q) of the UEG in the HF approximation."""
function static_structure_factor_hf(q, para::ParaMC)
    x = q / para.kF
    if x < 2
        return 3x / 4.0 - x^3 / 16.0
    end
    return 1.0
end

"""Π₀(q, τ=0) = χ₀(q, τ=0) = -n₀ S₀(q)"""
function bare_susceptibility_exact_t0(q, para::ParaMC)
    n0 = para.kF^3 / 3π^2
    return -n0 * static_structure_factor_hf(q, para)
end

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    # Setup plot styles
    style = PyPlot.matplotlib."style"
    style.use(["science", "std-colors"])
    color = [
        cdict["orange"],
        cdict["blue"],
        cdict["cyan"],
        cdict["magenta"],
        cdict["red"],
        # cdict["teal"],
    ]
    # color = [cdict["blue"], cdict["orange"], "green", cdict["red"], "black"]
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")

    # Use LaTex fonts for plots
    rcParams["font.size"] = 16
    rcParams["mathtext.fontset"] = "cm"
    # rcParams["font.family"] = "Times New Roman"

    fig = figure(; figsize=(6, 4))

    rs = 1.0
    beta = 40.0
    mass2 = 1.0
    solver = :vegasmc

    # Number of evals
    neval = 1e10

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order = 1
    # min_order = 2
    max_order = 4
    min_order_plot = 1
    max_order_plot = 4

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
    para = ParaMC(;
        order=max_order,
        rs=rs,
        beta=beta,
        mass2=mass2,
        isDynamic=false,
        isFock=false,
    )

    # Load the raw data
    local htf
    savename =
        "results/data/static_structure_factor/static_structure_factor_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval)_$(solver)$(ct_string)"
    # TODO: Rerun with new format,
    #   orders, para, kgrid, tgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
    orders, kgrid, partitions, data = jldopen("$savename.jld2", "a+") do f
        htf = f["has_taylor_factors"]
        key = "$(UEG.short(para))"
        return f[key]
    end

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / para.kF

    # Non-interacting density
    n0 = para.kF^3 / 3π^2

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    # data = UEG_MC.restodict(res, partitions)
    if htf == false
        for (k, v) in data
            data[k] = v / (factorial(k[2]) * factorial(k[3]))
        end
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
    static_structure_hf_exact = static_structure_factor_hf.(kgrid, [para])

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
    ct_filename = "examples/counterterms/data/data_Z.jld2"
    z, μ, has_taylor_factors = UEG_MC.load_z_mu_old(para; ct_filename=ct_filename)
    # Add Taylor factors to CT data
    if has_taylor_factors == false
        for (p, v) in z
            z[p] = v / (factorial(p[2]) * factorial(p[3]))
        end
        for (p, v) in μ
            μ[p] = v / (factorial(p[2]) * factorial(p[3]))
        end
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
    _, δμ, _ = CounterTerm.sigmaCT(max_order, μ, z; verbose=1)

    println("Computed δμ: ", δμ)
    static_structure = UEG_MC.chemicalpotential_renormalization_susceptibility(
        merged_data,
        δμ;
        min_order=1,
        max_order=max_order,
    )
    δμ1_exact = UEG_MC.delta_mu1(para)  # = ReΣ₁[λ](kF, 0)
    inst_poln_2_manual = merged_data[(2, 0)] + δμ1_exact * merged_data[(1, 1)]
    scores_2 = stdscore.(static_structure[2] - inst_poln_2_manual, 0.0)
    worst_score_2 = argmax(abs, scores_2)
    println("2nd order renorm vs manual worst score: $worst_score_2")

    println(UEG.paraid(para))
    println(partitions)
    println(typeof(static_structure))

    # Aggregate the full results for Σₓ up to order N
    static_structure_total = UEG_MC.aggregate_orders(static_structure)

    qgrid_fine = para.kF * np.linspace(0.0, 3.0; num=600)
    q_kf_grid_fine = np.linspace(0.0, 3.0; num=600)
    static_structure_hf_exact_fine = static_structure_factor_hf.(qgrid_fine, [para])

    # Plot the static structure factor for each aggregate order
    axvline(2.0; linestyle="--", color="gray")
    axhline(1.0; linestyle="--", color="gray")
    if min_order_plot == 1
        # Include exact Hartree-Fock static structure factor in plot
        plot(q_kf_grid_fine, static_structure_hf_exact_fine, "k"; label="HF")
    end
    ic = 1
    for (i, N) in enumerate(min_order:max_order_plot)
        # S(q) = -Π(q, τ=0) / n₀
        static_structure_means = Measurements.value.(static_structure_total[N])
        static_structure_stdevs = Measurements.uncertainty.(static_structure_total[N])
        marker = "o-"
        # _x, _y = spline(k_kf_grid, static_structure_means, static_structure_stdevs)
        _x, _y = spline_with_bc(k_kf_grid, static_structure_means, static_structure_stdevs)
        plot(
            _x,
            _y;
            color=color[i],
            linestyle=N > 0 ? "--" : "-",
            zorder=10 * i + 3,
            label=N == 0 ? "\$N = 0\$" : nothing,
        )
        errorbar(
            k_kf_grid,
            static_structure_means;
            yerr=static_structure_stdevs,
            color=color[i],
            capsize=2,
            markersize=2,
            fmt="o",
            markerfacecolor="none",
            label="\$N = $N\$",
            zorder=10 * i + 3,
        )
        # plot(
        #     k_kf_grid,
        #     static_structure_means,
        #     marker;
        #     markersize=2,
        #     color="$(colors[ic])",
        #     label="\$N=$N\$ ($solver)",
        # )
        # fill_between(
        #     k_kf_grid,
        #     static_structure_means - static_structure_stdevs,
        #     static_structure_means + static_structure_stdevs;
        #     color="$(colors[ic])",
        #     alpha=0.4,
        # )
        ic += 1
    end
    legend(; loc="best")
    # xlim(-0.1, 3.1)
    # ylim(nothing, 2)
    xlabel("\$q / k_F\$")
    # ylabel("\$S(q) = -\\chi(q, \\tau=0) / n_0\$")
    ylabel("\$S(q)\$")
    # xloc = 1.5
    xloc = 0.6
    yloc = 0.175
    ydiv = -0.125
    text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=12,
    )
    text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=12,
    )
    plt.tight_layout()
    savefig(
        "results/static_structure_factor/static_structure_factor_N=$(max_order_plot)_rs=$(para.rs)_" *
        "beta_ef=$(para.beta)_lambda=$(para.mass2)_neval=$(neval)_$(solver)$(ct_string)$(fix_string).pdf",
        # "beta_ef=$(para.beta)_lambda=$(para.mass2)_neval=$(neval)_$(solver)$(ct_string)$(fix_string)_exact_bare.pdf",
    )

    plt.close("all")
    return
end

main()
