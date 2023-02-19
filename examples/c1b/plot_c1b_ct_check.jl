using CodecZlib
using ElectronLiquid
using ElectronGas
using FeynmanDiagram
using JLD2
using MCIntegration
using Measurements
using PyCall
using SOSEM

# TODO: init δμ[n] to zero for n < lowest_order = 3 (no counterterms at order 3 for this observable)
# SOSEM.@todo

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1b/plot_c1b_ct_check.jl

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

    neval = 1e10
    min_order = 3
    max_order = 4
    min_order_plot = 4
    max_order_plot = 4
    @assert max_order ≥ 4
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

    plotsettings = DiagGen.Settings{DiagGen.Observable}(
        DiagGen.c1bL;
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
        "results/data/c1bL_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)"
    settings, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+"; compress=true) do f
        key = "$(UEG.short(plotparams[end]))"
        return f[key]
    end
    data = UEG_MC.restodict(res, partitions)
    println(data)

    # # Load the results from multiple JLD2 files
    # # data_fixed_orders = [3, 4]
    # data_fixed_orders = [3]
    # filenames = [
    #     "results/data/c1bL_n=$(order)_rs=$(rs)_" *
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
    rs_lo = 1.0
    sosem_lo = np.load("results/data/soms_rs=$(rs_lo)_beta_ef=40.0.npz")
    # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    param_lo = Parameter.atomicUnit(0, rs_lo)    # (dimensionless T, rs)
    eTF_lo = param_lo.qTF^2 / (2 * param_lo.me)
    c1b_lo_quad = sosem_lo.get("bare_b") / eTF_lo^2
    # delta RPA results for class (b) moment
    delta_c1b_rpa = sosem_lo.get("delta_rpa_b_vegas_N=1e+07.npy") / eTF_lo^2
    delta_c1b_rpa_err = sosem_lo.get("delta_rpa_b_err_vegas_N=1e+07.npy") / eTF_lo^2
    # delta RPA+FL results for class (b) moment
    delta_c1b_rpa_fl = sosem_lo.get("delta_rpa+fl_b_vegas_N=1e+07.npy") / eTF_lo^2
    delta_c1b_rpa_fl_err = sosem_lo.get("delta_rpa+fl_b_err_vegas_N=1e+07.npy") / eTF_lo^2
    if plot_bare
        # ax.plot(
        #     k_kf_grid_quad,
        #     c1b_lo_quad,
        #     "k";
        #     linestyle="--",
        #     label="LO (quad)",
        # )
        ax.plot(k_kf_grid, delta_c1b_rpa, "k"; linestyle="--", label="RPA (vegas)")
        ax.fill_between(
            k_kf_grid,
            (delta_c1b_rpa - delta_c1b_rpa_err),
            (delta_c1b_rpa + delta_c1b_rpa_err);
            color="k",
            alpha=0.3,
        )
        ax.plot(k_kf_grid, delta_c1b_rpa_fl, "k"; label="RPA\$+\$FL (vegas)")
        ax.fill_between(
            k_kf_grid,
            (delta_c1b_rpa_fl - delta_c1b_rpa_fl_err),
            (delta_c1b_rpa_fl + delta_c1b_rpa_fl_err);
            color="k",
            alpha=0.3,
        )
    end

    # Next available color for plotting
    next_color = 0
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
            # color=next_color == 3 ? "C0" : "C$next_color",
            color="C$next_color",
            label="\$\\widetilde{C}^{(1b)}_{$p}\$",
            # label="\$\\widetilde{C}^{(1b)}_{$p}\$ ($solver)",
            # label="\$\\mathcal{P}=$p\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            # color=next_color == 3 ? "C0" : "C$next_color",
            color="C$next_color",
            alpha=0.4,
        )
        next_color += 1
    end

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])

    # Reexpand merged data in powers of μ
    z, μ = UEG_MC.load_z_mu(param)
    δz, δμ = CounterTerm.sigmaCT(max_order - 2, μ, z; verbose=1)
    println("Computed δμ: ", δμ)
    c1bL = UEG_MC.chemicalpotential_renormalization_sosem(
        merged_data,
        δμ;
        lowest_order=3,  # there is no second order for this observable
        min_order=min_order_plot,
        max_order=max_order_plot,
    )

    # Test manual renormalization with exact lowest-order chemical potential;
    # the first-order counterterm is: δμ1= ReΣ₁[λ](kF, 0)
    # NOTE: For this observable, there is no second-order
    δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
    # C⁽¹ᵇ⁾₄ = 2(C⁽¹ᵇ⁾ᴸ_{4,0} + δμ₁ C⁽¹ᵇ⁾ᴸ_{3,1})
    c1bL4_exact = merged_data[(4, 0)] + δμ1_exact * merged_data[(3, 1)]
    c1b4_means_exact = 2 * Measurements.value.(c1bL4_exact)
    c1b4_errs_exact = 2 * Measurements.uncertainty.(c1bL4_exact)
    println("Largest magnitude of C^{(1b)}_{n=3}(k): $(2 * maximum(abs.(c1bL4_exact)))")
    # C⁽¹ᵇ⁾₄ = 2(C⁽¹ᵇ⁾ᴸ_{4,0} + δμ₁ C⁽¹ᵇ⁾ᴸ_{3,1}) (calc δμ₁)
    c1bL4 = c1bL[4]  # c1bL = [undef, c1bL3, c1bL4, ...]
    c1b4_means = 2 * Measurements.value.(c1bL4)
    c1b4_errs = 2 * Measurements.uncertainty.(c1bL4)
    stdscores = stdscore.(c1bL4, c1bL4_exact)
    worst_score = argmax(abs, stdscores)
    println("Exact δμ₁: ", δμ1_exact)
    println("Computed δμ₁: ", δμ[1])
    println(
        "Worst standard score for result at 4th " *
        "order (auto vs exact+manual): $worst_score",
    )

    # Check the counterterm cancellation to leading order in δμ
    if min_order_plot ≤ 4
        c1b4_kind = ["exact", "calc."]
        c1b4_kind_means = [c1b4_means_exact, c1b4_means]
        c1b4_kind_errs = [c1b4_errs_exact, c1b4_errs]
        for (kind, means, errs) in zip(c1b4_kind, c1b4_kind_means, c1b4_kind_errs)
            ax.plot(
                k_kf_grid,
                means,
                "-";
                # "o-";
                markersize=2,
                color="C$next_color",
                label="\$\\widetilde{C}^{(1b)}_{n=4} = \\widetilde{C}^{(1b)}_{(4,0)} " *
                      " + \\delta\\mu_1 \\widetilde{C}^{(1b)}_{(3,1)}\$ ($kind \$\\delta\\mu\$, $solver)",
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

    # Plot labels and legend
    ax.legend(; loc="lower right", ncol=2)
    ax.set_xlim(minimum(k_kf_grid), min(2.0, maximum(k_kf_grid)))
    # ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    ax.set_ylim(-0.11, nothing)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$\\widetilde{C}^{(1b)}_{(\\,\\cdot\\,)}(k) " *
        " \\equiv C^{(1b)}_{(\\,\\cdot\\,)}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    # (nmax = 4)
    xloc = 0.125
    yloc = 0.00125
    ydiv = -0.01
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
        "results/c1b/c1b_n=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)_ct_cancellation.pdf",
    )
    plt.close("all")
    return
end

main()
