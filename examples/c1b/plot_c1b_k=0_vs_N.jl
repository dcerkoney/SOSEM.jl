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

# We plot the k=0 value
const extK_load = 0.0

# NOTE: Call from main project directory as: julia examples/c1b/plot_c1b_total.jl

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
    expand_bare_interactions = 1  # testing single V[V_λ] scheme

    # neval = 1e10
    neval = 1e8

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    n_min = 3  # True minimal loop order for this observable
    min_order = 3
    max_order = 6
    min_order_plot = 2
    max_order_plot = 6
    @assert max_order ≥ 3

    # Load data from multiple fixed-order runs
    # fixed_orders = collect(min_order:max_order)

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Save total results?
    save = true

    # Include RPA(+FL) results?
    plot_rpa_fl = true

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
    ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # Load the order 3-4 results from JLD2 (and μ data from csv, if applicable)
    # if max_order == 5
    #     max_together = 4
    # else
    #     max_together = max_order
    # end
    savename =
        "results/data/c1bL_k=$(extK_load)_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)$(ct_string)"
    settings, param, extK, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(plotparam))"
        return f[key]
    end
    @assert extK == 0.0

    # # Load the fixed order 5 result from JLD2
    # local kgrid5, res5, partitions5
    # if max_order == 5
    #     savename5 =
    #         "results/data/c1bL_n=$(max_order)_rs=$(rs)_" *
    #         "beta_ef=$(beta)_lambda=$(mass2)_" *
    #         "neval=$(neval5)_$(intn_str)$(solver)$(ct_string)"
    #     settings5, param5, kgrid5, partitions5, res5 = jldopen("$savename5.jld2", "a+") do f
    #         key = "$(UEG.short(plotparam))"
    #         return f[key]
    #     end
    # end

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    for (k, v) in data
        data[k] = v / (factorial(k[2]) * factorial(k[3]))
    end
    # # Add 5th order results to data dict
    # if max_order == 5
    #     data5 = UEG_MC.restodict(res5, partitions5)
    #     for (k, v) in data5
    #         data5[k] = v / (factorial(k[2]) * factorial(k[3]))
    #     end
    #     merge!(data, data5)
    # end
    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])

    if plot_rpa_fl
        # Non-dimensionalize bare and RPA+FL non-local moments
        rs_lo = rs
        sosem_lo = np.load("results/data/soms_rs=$(rs_lo)_beta_ef=40.0.npz")
        # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
        param_lo = Parameter.atomicUnit(0, rs_lo)    # (dimensionless T, rs)
        eTF_lo = param_lo.qTF^2 / (2 * param_lo.me)

        # # Interpolate bare results and downsample to coarse k_kf_grid_vegas
        k_kf_grid_vegas = np.load("results/kgrids/kgrid_vegas_dimless_n=77_small.npy")
        @assert k_kf_grid_vegas[1] == 0.0

        # RPA(+FL) corrections to LO for class (b) moment at k=0
        delta_c1b_rpa = sosem_lo.get("delta_rpa_b_vegas_N=1e+07.npy")[1] / eTF_lo^2
        delta_c1b_rpa_err = sosem_lo.get("delta_rpa_b_err_vegas_N=1e+07.npy")[1] / eTF_lo^2
        delta_c1b_rpa_fl = sosem_lo.get("delta_rpa+fl_b_vegas_N=1e+07.npy")[1] / eTF_lo^2
        delta_c1b_rpa_fl_err =
            sosem_lo.get("delta_rpa+fl_b_err_vegas_N=1e+07.npy")[1] / eTF_lo^2
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
    δz, δμ = CounterTerm.sigmaCT(max_order - n_min, μ, z; verbose=1)
    println("Computed δμ: ", δμ)
    c1bL = UEG_MC.chemicalpotential_renormalization_sosem(
        merged_data,
        δμ;
        lowest_order=3,  # there is no second order for this observable
        min_order=min_order,
        max_order=max(max_order, max_order_plot),
    )
    # Test manual renormalization with exact lowest-order chemical potential
    if max_order >= 4
        # NOTE: For this observable, there is no second-order
        δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
        # C⁽¹ᵇ⁾₄ = 2(C⁽¹ᵇ⁾ᴸ_{4,0} + δμ₁ C⁽¹ᵇ⁾ᴸ_{3,1})
        c1b4_manual =
            2 *
            (merged_data[(3, 0)] + merged_data[(4, 0)] + δμ1_exact * merged_data[(3, 1)])
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

    # Aggregate the full results for C⁽¹ᶜ⁾ up to order N
    c1bL_total = UEG_MC.aggregate_orders(c1bL)

    # partitions = collect(Iterators.flatten(partitions_list))

    println(settings)
    println(UEG.paraid(param))
    println(res)
    println(partitions)
    # println(res_list)
    # println(partitions_list)

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
                    if haskey(f, "c1b_k=0")
                        if haskey(f["c1b_k=0"], "RPA") &&
                           haskey(f["c1b_k=0/RPA"], "neval=$(1e7)")
                            @warn("replacing existing data for RPA, neval=$(1e7)")
                            delete!(f["c1b_k=0/RPA"], "neval=$(1e7)")
                        end
                        if haskey(f["c1b_k=0"], "RPA+FL") &&
                           haskey(f["c1b_k=0/RPA+FL"], "neval=$(1e7)")
                            @warn("replacing existing data for RPA+FL, neval=$(1e7)")
                            delete!(f["c1b_k=0/RPA+FL"], "neval=$(1e7)")
                        end
                    end
                    # RPA
                    meas_rpa = measurement.(delta_c1b_rpa, delta_c1b_rpa_err)
                    # meas_rpa = measurement.(c1b_rpa, c1b_rpa_err)
                    f["c1b_k=0/RPA/neval=$(1e7)/meas"] = meas_rpa
                    f["c1b_k=0/RPA/neval=$(1e7)/param"] = param
                    # RPA+FL
                    meas_rpa_fl = measurement.(delta_c1b_rpa_fl, delta_c1b_rpa_fl_err)
                    # meas_rpa_fl = measurement.(c1b_rpa_fl, c1b_rpa_fl_err)
                    f["c1b_k=0/RPA+FL/neval=$(1e7)/meas"] = meas_rpa_fl
                    f["c1b_k=0/RPA+FL/neval=$(1e7)/param"] = param
                end
            else
                # num_eval = N == 5 ? neval5 : neval34
                num_eval = neval
                if haskey(f, "c1b_k=0") &&
                   haskey(f["c1b_k=0"], "N=$N") &&
                   haskey(f["c1b_k=0/N=$N"], "neval=$num_eval")
                    @warn("replacing existing data for N=$N, neval=$num_eval")
                    delete!(f["c1b_k=0/N=$N"], "neval=$num_eval")
                end
                # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
                #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
                f["c1b_k=0/N=$N/neval=$num_eval/meas"] = 2 * c1bL_total[N]
                f["c1b_k=0/N=$N/neval=$num_eval/settings"] = settings
                f["c1b_k=0/N=$N/neval=$num_eval/param"] = param
            end
        end
    end

    # Plot results vs order N
    fig, ax = plt.subplots()
    orders = min_order:max_order_plot
    # Compare with RPA & RPA+FL results
    if plot_rpa_fl && min_order_plot == 2
        ax.plot(
            orders,
            delta_c1b_rpa * one.(orders),
            "k";
            linestyle="--",
            label="RPA (vegas)",
        )
        ax.fill_between(
            orders,
            (delta_c1b_rpa - delta_c1b_rpa_err) * one.(orders),
            (delta_c1b_rpa + delta_c1b_rpa_err) * one.(orders);
            color="k",
            alpha=0.3,
        )
        ax.plot(orders, delta_c1b_rpa_fl * one.(orders), "k"; label="RPA\$+\$FL (vegas)")
        ax.fill_between(
            orders,
            (delta_c1b_rpa_fl - delta_c1b_rpa_fl_err) * one.(orders),
            (delta_c1b_rpa_fl + delta_c1b_rpa_fl_err) * one.(orders);
            color="k",
            alpha=0.3,
        )
    end
    # Get means and error bars from the results vs order
    # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
    #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
    means =
        [2 * Measurements.value.(c1bL_total[N][1]) for N in min_order:max_order_plot]
    stdevs = [
        2 * Measurements.uncertainty.(c1bL_total[N][1]) for
        N in min_order:max_order_plot
    ]
    # Data gets noisy above 3rd loop order
    marker = "o-"
    ax.plot(orders, means, marker; markersize=4, color="C0", label="RPT ($solver)")
    ax.fill_between(orders, means - stdevs, means + stdevs; color="C0", alpha=0.4)
    ax.legend(; loc="best")
    ax.set_xticks(orders)
    ax.set_xlim(minimum(orders), maximum(orders))
    ax.set_xlabel("Perturbation order \$N\$")
    ax.set_ylabel("\$C^{(1b)}(k=0) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$")
    xloc = 1.6
    yloc = -0.085
    ydiv = -0.025
    # ax.text(
    #     xloc,
    #     yloc,
    #     "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
    #     fontsize=14,
    # )
    # ax.text(
    #     xloc,
    #     yloc + ydiv,
    #     "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
    #     # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
    #     fontsize=14,
    # )
    # ax.text(
    #     xloc,
    #     yloc + 2 * ydiv,
    #     "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
    #     fontsize=12,
    # )
    if expand_bare_interactions == 0
        plt.title("Using fixed bare Coulomb interactions \$V_1\$, \$V_2\$")
    elseif expand_bare_interactions == 1
        plt.title(
            "Using single re-expanded Coulomb interaction \$V_1[V_\\lambda]\$, \$V_2\$",
        )
    elseif expand_bare_interactions == 2
        plt.title(
            "Using re-expanded Coulomb interactions \$V_1[V_\\lambda]\$, \$V_2[V_\\lambda]\$",
        )
    end
    plt.tight_layout()
    fig.savefig(
        "results/c1b/c1b_k=0_vs_N_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)$(ct_string)_total.pdf",
    )
    plt.close("all")
    return
end

main()
