using ElectronLiquid
using ElectronGas
using Interpolations
using FeynmanDiagram
using JLD2
using MCIntegration
using Measurements
using PyCall
using SOSEM: UEG_MC

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1d/plot_c1d_ct_check.jl

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    rs = 1.0
    beta = 40.0
    mass2 = 2.0
    # mass2 = 0.1
    solver = :vegasmc
    expand_bare_interactions = 0

    neval = 1e10
    n_min = 2  # True minimal loop order for this observable
    min_order = 3
    max_order = 4
    min_order_plot = 4
    max_order_plot = 4
    @assert max_order ≥ 3

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true
    @assert renorm_mu

    # Include unscreened bare result
    plot_bare = false

    plotparam =
        UEG.ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions == 2
        intn_str = "no_bare_"
    elseif expand_bare_interactions == 1
        intn_str = "one_bare_"
    end

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # colors = ["orchid", "cornflowerblue", "turquoise", "chartreuse", "greenyellow"]
    # markers = ["-", "-", "-", "-", "-"]

    # # Load the results from JLD2
    # savename =
    #     "results/data/c1d_n=$(max_order)_rs=$(rs)_" *
    #     "beta_ef=$(beta)_lambda=$(mass2)_" *
    #     "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)"
    # settings, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
    #     key = "$(UEG.short(plotparam))"
    #     return f[key]
    # end

    # Load the results using new JLD2 format
    filename =
        "results/data/rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(mass2)_$(intn_str)$(solver)_$(ct_string)"
    f = jldopen("$filename.jld2", "r")
    key = "c1d_n_min=$(min_order)_n_max=$(max_order)_neval=$(neval)"
    res = f["$key/res"]
    settings = f["$key/settings"]
    param = f["$key/param"]
    kgrid = f["$key/kgrid"]
    partitions = f["$key/partitions"]
    close(f)
    print(settings)
    print(param)
    print(kgrid)
    print(res)
    print(partitions)

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    println(settings)
    println(UEG.paraid(param))
    println(partitions)
    println(res)

    # Plot the results
    fig, ax = plt.subplots()

    # Load C⁽¹ᵈ⁾₂ quadrature results and interpolate on k_kf_grid
    rs_quad = rs
    # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    param_quad = Parameter.atomicUnit(0, rs_quad)    # (dimensionless T, rs)
    eTF_quad = param_quad.qTF^2 / (2 * param_quad.me)
    sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=40.0.npz")

    # Bare results (stored in Hartree a.u.)
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    c1d_bare_quad = sosem_quad.get("bare_d") / eTF_quad^2

    # Interpolate bare results and downsample to coarse k_kf_grid
    c1d_bare_interp = linear_interpolation(k_kf_grid_quad, c1d_bare_quad)
    c1d2_exact = c1d_bare_interp(k_kf_grid)

    if min_order_plot == 2
        ax.plot(
            k_kf_grid_quad,
            c1d_bare_quad,
            "k";
            label="\$\\widetilde{C}^{(1d)}_{(2,0,0)}\$",
        )
    end

    # Next available color for plotting
    next_color = 4
    for o in eachindex(partitions)
        if !(min_order_plot <= sum(partitions[o]) <= max_order_plot)
            continue
        end
        # Get means and error bars from the result for this partition
        local means, stdevs
        if res.config.N == 1
            # res gets automatically flattened for a single-partition measurement
            means, stdevs = res.mean, res.stdev
        else
            means, stdevs = res.mean[o], res.stdev[o]
        end
        # Data gets noisy above 1st Green's function counterterm order
        # marker =
        #     (partitions[o][2] > 1 || (partitions[o][1] > 3 && partitions[o][2] > 0)) ?
        #     "o-" : "-"
        marker = "-"
        ax.plot(
            k_kf_grid,
            means,
            marker;
            markersize=2,
            color=next_color == 3 ? "C0" : "C$next_color",
            # color="C$next_color",
            label="\$\\widetilde{C}^{(1d)}_{$(partitions[o])}\$",
            # label="\$\\widetilde{C}^{(1d)}_{$(partitions[o])}\$ ($solver)",
            # label="\$\\mathcal{P}=$(partitions[o])\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            color=next_color == 3 ? "C0" : "C$next_color",
            # color="C$next_color",
            alpha=0.4,
        )
        next_color -= 1
    end

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])

    if min_order_plot == 2
        # Set bare result manually using exact data to avoid systematic error in (2,0,0) calculation
        merged_data[(2, 0)] = measurement.(c1d2_exact, 0.0)  # treat quadrature data as numerically exact
    end

    # Reexpand merged data in powers of μ
    δμ = load_mu_counterterm(
            param;
            max_order=max_order - n_min,
            parafilename="examples/counterterms/data/para.csv",
            ct_filename="examples/counterterms/data/data_Z$(ct_string).jld2",
            verbose=1,
        )
    println("Computed δμ: ", δμ)
    c1d = UEG_MC.chemicalpotential_renormalization_sosem(
        merged_data,
        δμ;
        lowest_order=2,
        min_order=min_order_plot,
        max_order=max_order_plot,
    )

    # Check the counterterm cancellation to leading order in δμ
    if min_order_plot ≤ 3
        # Test manual renormalization with exact lowest-order chemical potential;
        # the first-order counterterm is: δμ1= ReΣ₁[λ](kF, 0)
        δμ1_exact = UEG_MC.delta_mu1(param)
        # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1} (exact δμ₁)
        c1d3_exact = merged_data[(3, 0)] + δμ1_exact * merged_data[(2, 1)]
        c1d3_means_exact = Measurements.value.(c1d3_exact)
        c1d3_errs_exact = Measurements.uncertainty.(c1d3_exact)
        println("Largest magnitude of C^{(1d)}_{n=3}(k): $(maximum(abs.(c1d3_exact)))")
        # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1} (calc δμ₁)
        c1d3_means = Measurements.value.(c1d[3])
        c1d3_errs = Measurements.uncertainty.(c1d[3])
        stdscores = stdscore.(c1d[3], c1d3_exact)
        worst_score = argmax(abs, stdscores)
        println("Exact δμ₁: ", δμ1_exact)
        println("Computed δμ₁: ", δμ[1])
        println(
            "Worst standard score for total result to 3rd " *
            "order (auto vs exact+manual): $worst_score",
        )

        c1d3_kind = ["exact", "calc."]
        c1d3_kind_means = [c1d3_means_exact, c1d3_means]
        c1d3_kind_errs = [c1d3_errs_exact, c1d3_errs]
        for (kind, means, errs) in zip(c1d3_kind, c1d3_kind_means, c1d3_kind_errs)
            ax.plot(
                k_kf_grid,
                means,
                "-";
                # "o-";
                markersize=2,
                color="C$next_color",
                label="\$\\widetilde{C}^{(1d)}_{n=3} = \\widetilde{C}^{(1d)}_{(3,0)} " *
                      " + \\delta\\mu_1 \\widetilde{C}^{(1d)}_{(2,1)}\$ ($kind \$\\delta\\mu\$, $solver)",
            )
            ax.fill_between(
                k_kf_grid,
                means - errs,
                means + errs;
                color="C$next_color",
                alpha=0.4,
            )
            next_color -= 1
        end
    end

    # Plot the counterterm cancellation at next-leading order in δμ
    if max_order_plot ≥ 4
        c1d4_means = Measurements.value.(c1d[4])
        c1d4_errs = Measurements.uncertainty.(c1d[4])
        ax.plot(
            k_kf_grid,
            c1d4_means,
            "-";
            # "o-";
            # markersize=3,
            linewidth=2,
            color=next_color == 0 ? "r" : "C$next_color",
            label="\$\\widetilde{C}^{(1d)}_{n=4}\$",
        )
        ax.fill_between(
            k_kf_grid,
            c1d4_means - c1d4_errs,
            c1d4_means + c1d4_errs;
            color=next_color == 0 ? "r" : "C$next_color",
            alpha=0.4,
        )
        ax.text(
            0.25,
            -0.02,
            "\$\\widetilde{C}^{(1d)}_{n=4} = \\widetilde{C}^{(1d)}_{(4,0)} + " *
            "\\delta\\mu_1 \\widetilde{C}^{(1d)}_{(3,1)}\$";
            fontsize=12,
        )
        ax.text(
            0.465,
            -0.0325,
            "\$ + \\delta\\mu^2_1 \\widetilde{C}^{(1d)}_{(2,2)} + " *
            "\\delta\\mu_2 \\widetilde{C}^{(1d)}_{(2,1)}\$";
            fontsize=12,
        )
    end

    # Plot labels and legend
    ax.legend(; loc="lower right", ncol=2)
    ax.set_xlim(minimum(k_kf_grid), min(2.0, maximum(k_kf_grid)))
    # ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    # ax.set_ylim(; bottom=-0.2, top=0.5)
    ax.set_ylim(; bottom=-0.04, top=0.1)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$\\widetilde{C}^{(1d)}_{(\\,\\cdot\\,)}(k) " *
        " \\equiv C^{(1d)}_{(\\,\\cdot\\,)}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    # # (nmax = 3)
    # xloc = 1.125
    # yloc = 0.45
    # ydiv = -0.06
    # (nmax = 4)
    xloc = 1.15
    yloc = 0.0875
    ydiv = -0.0125
    ax.text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = 2\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + 2 * ydiv,
        "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=12,
    )
    # plt.title("Using fixed bare Coulomb interactions \$V_1\$, \$V_2\$")
    # plt.title(
    #     "Using re-expanded Coulomb interactions \$V_1[V_\\lambda]\$, \$V_2[V_\\lambda]\$",
    # )
    plt.tight_layout()
    fig.savefig(
        "results/c1d/c1d_n=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)_ct_cancellation.pdf",
    )
    plt.close("all")
    return
end

main()
