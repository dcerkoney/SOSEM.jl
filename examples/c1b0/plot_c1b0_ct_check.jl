using ElectronLiquid
using ElectronGas
using FeynmanDiagram
using JLD2
using MCIntegration
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1b0/plot_c1b0_ct_check.jl

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
    expand_bare_interactions = false

    neval = 3e10
    min_order = 3
    max_order = 4
    min_order_plot = 3
    max_order_plot = 3
    @assert max_order ≥ 3
    @assert min_order_plot ≥ min_order && max_order_plot ≤ max_order

    # Load data from multiple fixed-order runs
    fixed_orders = collect(min_order:max_order)

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true
    @assert renorm_mu

    # Include unscreened bare result
    plot_bare = false

    plotparams = [
        UEG.ParaMC(; order=order, rs=rs, beta=beta, mass2=mass2, isDynamic=false) for
        order in fixed_orders
    ]

    plotsettings = DiagGen.Settings(;
        observable=DiagGen.c1bL,
        min_order=min_order,
        max_order=max_order,
        expand_bare_interactions=expand_bare_interactions,
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
    )

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

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # colors = ["orchid", "cornflowerblue", "turquoise", "chartreuse", "greenyellow"]
    # markers = ["-", "-", "-", "-", "-"]

    # Load the results from JLD2
    savename =
        "results/data/c1bL0_n=$(max_order)_rs=$(rs)_" *
        # "results/data/c1bL0_n=$(min_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)"
    settings, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(plotparams[1]))"
        return f[key]
    end
    data = UEG_MC.restodict(res, partitions)
    println(data)

    # # Load the results from multiple JLD2 files
    # # data_fixed_orders = [3, 4]
    # data_fixed_orders = [3]
    # filenames = [
    #     "results/data/c1bL0_n=$(order)_rs=$(rs)_" *
    #     "beta_ef=$(beta)_lambda=$(mass2)_" *
    #     "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)" for
    #     order in data_fixed_orders
    # ]
    # settings, param, kgrid, partitions_list, res_list =
    #     UEG_MC.load_fixed_order_data_jld2(filenames, plotsettings, plotparams)

    # Convert fixed-order data to dictionary
    # data = UEG_MC.restodict(res_list, partitions_list)

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    println(settings)
    println(UEG.paraid(param))
    # println(res_list)
    # println(partitions_list)
    println(res)
    println(partitions)

    # Plot the results
    fig, ax = plt.subplots()

    # Non-dimensionalize bare and RPA+FL non-local moments
    rs_quad = 1.0
    sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=40.0.npz")
    # np.load("results/data/soms_rs=$(Float64(param.rs))_beta_ef=$(param.beta).npz")
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    param_quad = Parameter.atomicUnit(0, rs_quad)    # (dimensionless T, rs)
    eTF_quad = param_quad.qTF^2 / (2 * param_quad.me)
    c1b_lo_quad_dimless = sosem_quad.get("bare_b") / eTF_quad^2
    # RPA+FL results for class (b) moment
    c1b_rpa_fl = sosem_quad.get("rpa+fl_b") / eTF_quad^2
    c1b_rpa_fl_err = sosem_quad.get("rpa+fl_b_err") / eTF_quad^2
    if plot_bare
        ax.plot(k_kf_grid_quad, c1b_lo_quad_dimless, "k"; linestyle="--", label="LO (quad)")
        ax.plot(k_kf_grid_quad, c1b_rpa_fl, "k"; label="RPA\$+\$FL (vegas)")
        ax.fill_between(
            k_kf_grid_quad,
            c1b_rpa_fl - c1b_rpa_fl_err,
            c1b_rpa_fl + c1b_rpa_fl_err;
            color="k",
            alpha=0.3,
        )
    end

    # Next available color for plotting
    next_color = 4
    # for partitions in partitions_list
    for p in partitions
        if !(min_order_plot <= sum(p) <= max_order_plot)
            continue
        end
        # Get means and error bars from the result up to this order
        # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
        #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
        means = 2 * Measurements.value.(data[p])
        stdevs = 2 * Measurements.uncertainty.(data[p])
        # Data gets noisy above 1st Green's function counterterm order
        # marker =
        #     (p[2] > 1 || (p[1] > 3 && p[2] > 0)) ?
        #     "o-" : "-"
        marker = "-"
        ax.plot(
            k_kf_grid,
            means,
            marker;
            markersize=2,
            color=next_color == 3 ? "C0" : "C$next_color",
            # color="C$next_color",
            label="\$\\widetilde{C}^{(1b0)}_{$p}\$",
            # label="\$\\widetilde{C}^{(1b0)}_{$p}\$ ($solver)",
            # label="\$\\mathcal{P}=$p\$ ($solver)",
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
        # end
    end

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])

    # Reexpand merged data in powers of μ
    z, μ = UEG_MC.load_z_mu(param)
    δz, δμ = CounterTerm.sigmaCT(max_order - 2, μ, z; verbose=1)
    println("Computed δμ: ", δμ)
    c1b0 = UEG_MC.chemicalpotential_renormalization(
        merged_data,
        δμ;
        min_order=min_order_plot,
        max_order=max_order_plot,
    )

    # Check the counterterm cancellation to leading order in δμ
    if min_order_plot ≤ 3
        # Test manual renormalization with exact lowest-order chemical potential;
        # the first-order counterterm is: δμ1= ReΣ₁[λ](kF, 0)
        δμ1_exact = UEG_MC.delta_mu1(param)
        # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1} (exact δμ₁)
        c1b03_exact = merged_data[(3, 0)] + δμ1_exact * merged_data[(2, 1)]
        c1b03_means_exact = Measurements.value.(c1b03_exact)
        c1b03_errs_exact = Measurements.uncertainty.(c1b03_exact)
        println("Largest magnitude of C^{(1b0)}_{n=3}(k): $(maximum(abs.(c1b03_exact)))")
        # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1} (calc δμ₁)
        c1b03 = c1b0[3]  # c1b0 = [c1b02, c1b03, ...]
        c1b03_means = Measurements.value.(c1b03)
        c1b03_errs = Measurements.uncertainty.(c1b03)
        stdscores = stdscore.(c1b03, c1b03_exact)
        worst_score = argmax(abs, stdscores)
        println("Exact δμ₁: ", δμ1_exact)
        println("Computed δμ₁: ", δμ[1])
        println(
            "Worst standard score for total result to 3rd " *
            "order (auto vs exact+manual): $worst_score",
        )

        c1b03_kind = ["exact", "calc."]
        c1b03_kind_means = [c1b03_means_exact, c1b03_means]
        c1b03_kind_errs = [c1b03_errs_exact, c1b03_errs]
        for (kind, means, errs) in zip(c1b03_kind, c1b03_kind_means, c1b03_kind_errs)
            ax.plot(
                k_kf_grid,
                means,
                "-";
                # "o-";
                markersize=2,
                color="C$next_color",
                label="\$\\widetilde{C}^{(1b0)}_{n=3} = \\widetilde{C}^{(1b0)}_{(3,0)} " *
                      " + \\delta\\mu_1 \\widetilde{C}^{(1b0)}_{(2,1)}\$ ($kind \$\\delta\\mu\$, $solver)",
            )
            ax.fill_between(
                k_kf_grid,
                means - errs,
                means + errs;
                color="C$next_color",
                alpha=0.4,
            )
            next_color += 1
        end
    end

    # Plot the counterterm cancellation at next-leading order in δμ
    if max_order_plot ≥ 4
        c1b04_means = Measurements.value.(c1b0[4])
        c1b04_errs = Measurements.uncertainty.(c1b0[4])
        ax.plot(
            k_kf_grid,
            c1b04_means,
            "-";
            # "o-";
            # markersize=3,
            linewidth=2,
            color=next_color == 0 ? "r" : "C$next_color",
            label="\$\\widetilde{C}^{(1b0)}_{n=4}\$",
        )
        ax.fill_between(
            k_kf_grid,
            c1b04_means - c1b04_errs,
            c1b04_means + c1b04_errs;
            color=next_color == 0 ? "r" : "C$next_color",
            alpha=0.4,
        )
        ax.text(
            0.1025,
            # -0.75,
            -0.2,
            "\$\\widetilde{C}^{(1b0)}_{n=4} = \\widetilde{C}^{(1b0)}_{(4,0)} + " *
            "\\delta\\mu_1 \\widetilde{C}^{(1b0)}_{(3,1)}\$";
            fontsize=12,
        )
        ax.text(
            0.31,
            # -1.025,
            -0.275,
            "\$ + \\delta\\mu^2_1 \\widetilde{C}^{(1b0)}_{(2,2)} + " *
            "\\delta\\mu_2 \\widetilde{C}^{(1b0)}_{(2,1)}\$";
            fontsize=12,
        )
    end

    # Plot labels and legend
    ax.legend(; loc="lower right", ncol=2)
    ax.set_xlim(minimum(k_kf_grid), min(2.0, maximum(k_kf_grid)))
    # ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    ax.set_ylim(-0.4, nothing)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$\\widetilde{C}^{(1b0)}_{(\\,\\cdot\\,)}(k) " *
        " \\equiv C^{(1b0)}_{(\\,\\cdot\\,)}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    # # (nmax = 3)
    xloc = 0.12
    yloc = -0.225
    ydiv = -0.035
    # (nmax = 4)
    # xloc = 1.125
    # yloc = 0.375
    # ydiv = -0.075
    ax.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
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
        "results/c1b0/c1b0_n=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)_ct_cancellation.pdf",
    )
    plt.close("all")
    return
end

main()
