using CodecZlib
using ElectronLiquid
using ElectronGas
using FeynmanDiagram
using Interpolations
using JLD2
using MCIntegration
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1b/plot_c1b_total.jl

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    rs = 2.0
    beta = 40.0
    mass2 = 0.4
    solver = :vegasmc
    expand_bare_interactions = false

    neval34 = 5e10
    neval5 = 5e10
    neval = max(neval34, neval5)
    # neval = neval34

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    n_min = 3  # True minimal loop order for this observable
    min_order = 3
    max_order = 4
    min_order_plot = 2
    max_order_plot = 4
    @assert max_order ≥ 3

    # Load data from multiple fixed-order runs
    # fixed_orders = collect(min_order:max_order)

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Manually perform chemical potential renormalization
    renorm_mu_lo_ex = false  # at lowest order
    renorm_mu_nlo_ex = false  # at next-lowest order

    # Save total results
    save = true

    # Include RPA(+FL) results?
    plot_rpa_fl = false

    plotparam =
        UEG.ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    # plotparams = [
    #     UEG.ParaMC(; order=order, rs=rs, beta=beta, mass2=mass2, isDynamic=false) for
    #     order in fixed_orders
    # ]

    # plotsettings = DiagGen.Settings{DiagGen.Observable}(;
    #     DiagGen.c1bL,
    #     min_order=min_order,
    #     max_order=max_order,
    #     expand_bare_interactions=expand_bare_interactions,
    #     filter=[NoHartree],
    #     interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
    # )

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions
        intn_str = "no_bare_"
    end

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
    renorm_string = (renorm_mu && renorm_mu_lo_ex) ? "_lo_mu_manual" : ""
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

    # Load the order 3-4 results from JLD2 (and μ data from csv, if applicable)
    if max_order == 5
        max_together = 4
    else
        max_together = max_order
    end
    savename =
        "results/data/c1bL_n=$(max_together)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval34)_$(intn_str)$(solver)$(ct_string)"
    settings, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(plotparam))"
        return f[key]
    end

    # Load the fixed order 5 result from JLD2
    local kgrid5, res5, partitions5
    if max_order == 5
        savename5 =
            "results/data/c1bL_n=$(max_order)_rs=$(rs)_" *
            "beta_ef=$(beta)_lambda=$(mass2)_" *
            "neval=$(neval5)_$(intn_str)$(solver)$(ct_string)"
        settings5, param5, kgrid5, partitions5, res5 = jldopen("$savename5.jld2", "a+") do f
            key = "$(UEG.short(plotparam))"
            return f[key]
        end
    end

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF
    if max_order == 5
        k_kf_grid5 = kgrid5 / param.kF
    end

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    for (k, v) in data
        data[k] = v / (factorial(k[2]) * factorial(k[3]))
    end

    # Add 5th order results to data dict
    if max_order == 5
        data5 = UEG_MC.restodict(res5, partitions5)
        for (k, v) in data5
            data5[k] = v / (factorial(k[2]) * factorial(k[3]))
        end
        merge!(data, data5)
    end
    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])

    if plot_rpa_fl
        # Non-dimensionalize bare and RPA+FL non-local moments
        rs_lo = rs
        sosem_lo = np.load("results/data/soms_rs=$(rs_lo)_beta_ef=40.0.npz")
        # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
        param_lo = Parameter.atomicUnit(0, rs_lo)    # (dimensionless T, rs)
        eTF_lo = param_lo.qTF^2 / (2 * param_lo.me)

        # Bare results (stored in Hartree a.u.)
        k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
        c1b_bare_quad = sosem_lo.get("bare_b") / eTF_lo^2

        # # Interpolate bare results and downsample to coarse k_kf_grid_vegas
        k_kf_grid_vegas = np.load("results/kgrids/kgrid_vegas_dimless_n=77_small.npy")

        # c1b_bare_interp = linear_interpolation(k_kf_grid_quad, c1b_bare_quad)
        # c1b2_exact = c1b_bare_interp(k_kf_grid)

        # RPA(+FL) corrections to LO for class (b) moment
        delta_c1b_rpa = sosem_lo.get("delta_rpa_b_vegas_N=1e+07.npy") / eTF_lo^2
        delta_c1b_rpa_err = sosem_lo.get("delta_rpa_b_err_vegas_N=1e+07.npy") / eTF_lo^2
        delta_c1b_rpa_fl = sosem_lo.get("delta_rpa+fl_b_vegas_N=1e+07.npy") / eTF_lo^2
        delta_c1b_rpa_fl_err =
            sosem_lo.get("delta_rpa+fl_b_err_vegas_N=1e+07.npy") / eTF_lo^2
    end

    # Get total data
    if renorm_mu
        if renorm_mu_lo_ex && max_order_plot == 4
            δμ1 = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
            # C⁽¹ᵇ⁾₄ = C⁽¹ᵇ⁾_{4,0} + δμ₁ C⁽¹ᵇ⁾_{3,1}
            c1b3L = merged_data[(3, 0)]
            c1b4L = merged_data[(4, 0)] + δμ1 * merged_data[(3, 1)]
            c1bL = SortedDict(3 => c1b3L, 4 => c1b4L)
        else
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
            # δz, δμ = CounterTerm.sigmaCT(2, μ, z; verbose=1)  # TODO: Debug 3rd order CTs
            δz, δμ = CounterTerm.sigmaCT(max_order - 2, μ, z; verbose=1)
            println("Computed δμ: ", δμ)
            c1bL = UEG_MC.chemicalpotential_renormalization_sosem(
                merged_data,
                δμ;
                lowest_order=3,  # there is no second order for this observable
                min_order=min_order,
                max_order=max(max_order, max_order_plot),
            )
            # Test manual renormalization with exact lowest-order chemical potential
            if !renorm_mu_lo_ex && max_order >= 4
                # NOTE: For this observable, there is no second-order
                δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
                # C⁽¹ᵇ⁾₄ = 2(C⁽¹ᵇ⁾ᴸ_{4,0} + δμ₁ C⁽¹ᵇ⁾ᴸ_{3,1})
                c1b4_manual =
                    2 * (
                        merged_data[(3, 0)] +
                        merged_data[(4, 0)] +
                        δμ1_exact * merged_data[(3, 1)]
                    )
                c1b4L = 2 * (c1bL[3] + c1bL[4])
                stdscores = stdscore.(c1b4L, c1b4_manual)
                worst_score = argmax(abs, stdscores)
                println("Exact δμ₁: ", δμ1_exact)
                println("Computed δμ₁: ", δμ[1])
                println(
                    "Worst standard score for total result to 4th " *
                    "order (auto vs exact+manual): $worst_score",
                )
            end
        end
    else
        c1bL = merged_data
    end

    # Aggregate the full results for C⁽¹ᶜ⁾ up to order N
    if renorm_mu_lo_ex
        c1bL_total = Dict(3 => c1b3L, 4 => c1b3L + c1b4L)
    else
        c1bL_total = UEG_MC.aggregate_orders(c1bL)
    end

    # partitions = collect(Iterators.flatten(partitions_list))

    println(settings)
    println(UEG.paraid(param))
    # println(res_list)
    # println(partitions_list)
    println(res)
    println(partitions)

    # Plot the results
    fig, ax = plt.subplots()

    if plot_rpa_fl && min_order_plot == 2
        ax.plot(
            k_kf_grid_vegas,
            delta_c1b_rpa,
            "k";
            linestyle="--",
            label="RPA (vegas)",
            # label="\$\\delta C^{(1b)}_{\\mathrm{RPA}}\$ (vegas)",
        )
        # ax.plot(k_kf_grid_vegas, c1b_rpa, "k"; linestyle="--", label="RPA (vegas)")
        ax.fill_between(
            k_kf_grid_vegas,
            (delta_c1b_rpa - delta_c1b_rpa_err),
            (delta_c1b_rpa + delta_c1b_rpa_err);
            # (c1b_rpa - c1b_rpa_err),
            # (c1b_rpa + c1b_rpa_err);
            color="k",
            alpha=0.3,
        )
        ax.plot(
            k_kf_grid_vegas,
            delta_c1b_rpa_fl,
            "k";
            label="RPA\$+\$FL (vegas)",
            # label="\$\\delta C^{(1b)}_{\\mathrm{RPA}+\\mathrm{FL}}\$ (vegas)",
        )
        # ax.plot(k_kf_grid_vegas, c1b_rpa_fl, "k"; label="RPA\$+\$FL (vegas)")
        ax.fill_between(
            k_kf_grid_vegas,
            (delta_c1b_rpa_fl - delta_c1b_rpa_fl_err),
            (delta_c1b_rpa_fl + delta_c1b_rpa_fl_err);
            # (c1b_rpa_fl - c1b_rpa_fl_err),
            # (c1b_rpa_fl + c1b_rpa_fl_err);
            color="k",
            alpha=0.3,
        )
        # ax.plot(k_kf_grid_vegas, c1b2_exact, "C0"; linestyle="-", label="\$N=2\$ (quad)")
    end

    if save
        savename =
            "results/data/rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_$(intn_str)$(solver)$(ct_string)"
        f = jldopen("$savename.jld2", "a+"; compress=true)
        # NOTE: no bare result for c1b observable (accounted for in c1b0)
        for N in min_order_plot:max_order
            # Add RPA & RPA+FL results to data group
            if N == 2
                if plot_rpa_fl
                    if haskey(f, "c1b")
                        if haskey(f["c1b"], "RPA") && haskey(f["c1b/RPA"], "neval=$(1e7)")
                            @warn("replacing existing data for RPA, neval=$(1e7)")
                            delete!(f["c1b/RPA"], "neval=$(1e7)")
                        end
                        if haskey(f["c1b"], "RPA+FL") &&
                           haskey(f["c1b/RPA+FL"], "neval=$(1e7)")
                            @warn("replacing existing data for RPA+FL, neval=$(1e7)")
                            delete!(f["c1b/RPA+FL"], "neval=$(1e7)")
                        end
                    end
                    # RPA
                    meas_rpa = measurement.(delta_c1b_rpa, delta_c1b_rpa_err)
                    # meas_rpa = measurement.(c1b_rpa, c1b_rpa_err)
                    f["c1b/RPA/neval=$(1e7)/meas"] = meas_rpa
                    f["c1b/RPA/neval=$(1e7)/param"] = param
                    f["c1b/RPA/neval=$(1e7)/kgrid"] = kgrid
                    # RPA+FL
                    meas_rpa_fl = measurement.(delta_c1b_rpa_fl, delta_c1b_rpa_fl_err)
                    # meas_rpa_fl = measurement.(c1b_rpa_fl, c1b_rpa_fl_err)
                    f["c1b/RPA+FL/neval=$(1e7)/meas"] = meas_rpa_fl
                    f["c1b/RPA+FL/neval=$(1e7)/param"] = param
                    f["c1b/RPA+FL/neval=$(1e7)/kgrid"] = kgrid
                end
            else
                num_eval = N == 5 ? neval5 : neval34
                if haskey(f, "c1b") &&
                   haskey(f["c1b"], "N=$N") &&
                   haskey(f["c1b/N=$N"], "neval=$num_eval")
                    @warn("replacing existing data for N=$N, neval=$num_eval")
                    delete!(f["c1b/N=$N"], "neval=$num_eval")
                end
                # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
                #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
                f["c1b/N=$N/neval=$num_eval/meas"] = 2 * c1bL_total[N]
                f["c1b/N=$N/neval=$num_eval/settings"] = settings
                f["c1b/N=$N/neval=$num_eval/param"] = param
                f["c1b/N=$N/neval=$num_eval/kgrid"] = kgrid
            end
        end
    end

    # Plot for each aggregate order
    # colors = ["C2", "C1", "red"]
    colors = ["C1", "C2", "C3"]
    for (i, N) in enumerate(min_order:max_order_plot)
        # NOTE: Currently using a different kgrid at order 5
        if max_order == 5
            k_over_kfs = k_kf_grid5
        else
            k_over_kfs = k_kf_grid
        end
        # Get means and error bars from the result up to this order
        # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
        #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
        means = 2 * Measurements.value.(c1bL_total[N])
        stdevs = 2 * Measurements.uncertainty.(c1bL_total[N])
        # Data gets noisy above 3rd loop order
        marker = "o-"
        # marker = "-"
        # marker = N > 3 ? "o-" : "-"
        ax.plot(
            k_kf_grid,
            means,
            marker;
            markersize=2,
            color=colors[i],
            # color="C$i",
            label="\$N=$(N)\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            color=colors[i],
            alpha=0.4,
        )
        # ax.fill_between(k_kf_grid, means - stdevs, means + stdevs; color="C$i", alpha=0.4)
        # if !renorm_mu_lo_ex && max_order <= 4 && N == 4
        #     ax.plot(
        #         k_kf_grid,
        #         Measurements.value.(c1b4_manual);
        #         color="r",
        #         linestyle="-",
        #         label="\$N=4\$ (manual, vegasmc)",
        #     )
        # end
    end
    ax.legend(; loc="lower right")
    ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    # ax.set_xlim(minimum(k_kf_grid), 2.0)
    # ax.set_ylim(-1.2, -0.675)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$C^{(1b)}_{N}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
        # "\$\\left(C^{(1b0)}_{2}(k) + C^{(1b)}_{N}(k)\\right) \\,\\Big/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    # # For C^{(1b)}_N
    xloc = 0.125
    # yloc = -0.0175
    # ydiv = -0.01
    yloc = -0.03
    ydiv = -0.025
    # For C^{(1b0)}_2 + C^{(1b)}_N
    # xloc = 0.125
    # yloc = -1.05
    # ydiv = -0.05
    ax.text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
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
        # "results/c1b/c1b0_2+c1b_N=$(max_order_plot)_rs=$(param.rs)_" *
        "results/c1b/c1b_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)$(ct_string)" *
        "$(renorm_string)_total.pdf",
    )
    plt.close("all")
    return
end

main()
