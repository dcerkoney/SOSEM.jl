using CodecZlib
using ElectronLiquid
using ElectronGas
using Interpolations
using JLD2
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1d/plot_c1d_total.jl

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

    # neval3 = 5e10
    neval34 = 5e10
    neval5 = 5e10
    neval = max(neval34, neval5)

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    n_min = 2  # True minimal loop order for this observable
    min_order = 3
    max_order = 4
    min_order_plot = 2
    max_order_plot = 4

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Save total results
    save = true

    # Include RPA(+FL) results?
    # TODO: run RPA(+FL) at rs = 2 and rs = 5.
    # NOTE: Non-dimensionalized LO data is rs-independent!
    plot_rpa_fl = false

    # UEG parameters for MC integration
    plotparam = ParaMC(;
        order=max_order,
        rs=rs,
        beta=beta,
        mass2=mass2,
        isDynamic=false,
        isFock=false,
    )

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions
        intn_str = "no_bare_"
    end

    # Full renormalization
    ct_string = "_with_ct_mu_lambda"

    # savename =
    #     "results/data/c1nl_n=$(max_order)_rs=$(rs)_" *
    #     "beta_ef=$(beta)_lambda=$(mass2)_" *
    #     "neval=$(neval)_$(intn_str)$(solver)$(ct_string)"
    # settings, param, kgrid, partitions, res =
    #     jldopen("$savename.jld2", "a+"; compress=true) do f
    #         key = "$(UEG.short(plotparam))"
    #         return f[key]
    #     end

    # Load the order 3-4 results from JLD2 (and μ data from csv, if applicable)
    if max_order == 5
        max_together = 4
    else
        max_together = max_order
    end
    savename =
        "results/data/c1nl_n=$(max_together)_rs=$(rs)_" *
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
            "results/data/c1nl_n=$(max_order)_rs=$(rs)_" *
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

    # Non-dimensionalize bare and RPA+FL non-local moments
    rs_lo = 1.0
    sosem_lo = np.load("results/data/soms_rs=$(rs_lo)_beta_ef=40.0.npz")
    # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    param_lo = Parameter.atomicUnit(0, rs_lo)    # (dimensionless T, rs)
    eTF_lo = param_lo.qTF^2 / (2 * param_lo.me)

    # Bare and RPA(+FL) results (stored in Hartree a.u.)
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    c1nl_lo =
        (sosem_lo.get("bare_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d")) /
        eTF_lo^2
    c1nl_rpa =
        (sosem_lo.get("rpa_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d")) / eTF_lo^2
    c1nl_rpa_fl =
        (sosem_lo.get("rpa+fl_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d")) /
        eTF_lo^2
    # RPA(+FL) means are error bars
    c1nl_rpa_means, c1nl_rpa_stdevs =
        Measurements.value.(c1nl_rpa), Measurements.uncertainty.(c1nl_rpa)
    c1nl_rpa_fl_means, c1nl_rpa_fl_stdevs =
        Measurements.value.(c1nl_rpa_fl), Measurements.uncertainty.(c1nl_rpa_fl)

    # Interpolate bare results and downsample to coarse k_kf_grid
    c1nl_lo_interp = linear_interpolation(k_kf_grid_quad, c1nl_lo; extrapolation_bc=Line())
    c1nl_2_exact = c1nl_lo_interp(k_kf_grid)

    # if min_order_plot == 2
    if min_order_plot == 2 && min_order > 2
        # Set bare result manually using exact data to avoid statistical error in (2,0,0) calculation
        merged_data[(2, 0)] = measurement.(c1nl_2_exact, 0.0)  # treat quadrature data as numerically exact
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
    # δz, δμ = CounterTerm.sigmaCT(2, μ, z; verbose=1)  # TODO: Debug 3rd order CTs
    δz, δμ = CounterTerm.sigmaCT(max_order - 2, μ, z; verbose=1)
    println("Computed δμ: ", δμ)
    c1nl = UEG_MC.chemicalpotential_renormalization_sosem(
        merged_data,
        δμ;
        lowest_order=n_min,
        min_order=min(min_order, min_order_plot),
        max_order=max(max_order, max_order_plot),
    )
    # Test manual renormalization with exact lowest-order chemical potential
    δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
    # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1}
    c1nl3_manual =
        merged_data[(2, 0)] + merged_data[(3, 0)] + δμ1_exact * merged_data[(2, 1)]
    stdscores = stdscore.(c1nl[2] + c1nl[3], c1nl3_manual)
    worst_score = argmax(abs, stdscores)
    println("Exact δμ₁: ", δμ1_exact)
    println("Computed δμ₁: ", δμ[1])
    println(
        "Worst standard score for total result to 3rd " *
        "order (auto vs exact+manual): $worst_score",
    )
    # Aggregate the full results for C⁽¹ᶜ⁾ up to order N
    c1nl_total = UEG_MC.aggregate_orders(c1nl)

    println(settings)
    println(UEG.paraid(param))
    println(partitions)
    println(res)
    # println(partitions_list)
    # println(res_list)

    # Save total renormalized results
    if save
        savename =
            "results/data/rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_$(intn_str)$(solver)$(ct_string)"
        f = jldopen("$savename.jld2", "a+"; compress=true)
        # NOTE: no bare result for c1b observable (accounted for in c1b0)
        for (i, N) in enumerate(min_order_plot:max_order)
            # Add RPA & RPA+FL results to data group
            if N == 2
                if haskey(f, "c1nl")
                    if haskey(f["c1nl"], "RPA") && haskey(f["c1nl/RPA"], "neval=$(1e7)")
                        @warn("replacing existing data for RPA, neval=$(1e7)")
                        delete!(f["c1nl/RPA"], "neval=$(1e7)")
                    end
                    if haskey(f["c1nl"], "RPA+FL") &&
                       haskey(f["c1nl/RPA+FL"], "neval=$(1e7)")
                        @warn("replacing existing data for RPA+FL, neval=$(1e7)")
                        delete!(f["c1nl/RPA+FL"], "neval=$(1e7)")
                    end
                end
                # RPA
                meas_rpa = measurement.(c1nl_rpa_means, c1nl_rpa_stdevs)
                f["c1nl/RPA/neval=$(1e7)/meas"] = meas_rpa
                f["c1nl/RPA/neval=$(1e7)/param"] = param
                f["c1nl/RPA/neval=$(1e7)/kgrid"] = kgrid
                # RPA+FL
                meas_rpa_fl = measurement.(c1nl_rpa_fl_means, c1nl_rpa_fl_stdevs)
                f["c1nl/RPA+FL/neval=$(1e7)/meas"] = meas_rpa_fl
                f["c1nl/RPA+FL/neval=$(1e7)/param"] = param
                f["c1nl/RPA+FL/neval=$(1e7)/kgrid"] = kgrid
            else
                # neval = nevals[i - 1]  # +1 shift due to extra exact data at N = 2 
                if haskey(f, "c1nl") &&
                   haskey(f["c1nl"], "N=$N") &&
                   haskey(f["c1nl/N=$N"], "neval=$(neval)")
                    @warn("replacing existing data for N=$N, neval=$(neval)")
                    delete!(f["c1nl/N=$N"], "neval=$(neval)")
                end
                f["c1nl/N=$N/neval=$(neval)/meas"] = c1nl_total[N]
                f["c1nl/N=$N/neval=$(neval)/settings"] = settings
                f["c1nl/N=$N/neval=$(neval)/param"] = param
                f["c1nl/N=$N/neval=$(neval)/kgrid"] = kgrid
            end
        end
    end

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # colors = ["orchid", "cornflowerblue", "turquoise", "chartreuse", "greenyellow"]
    # markers = ["-", "-", "-", "-", "-"]

    # Plot the results
    fig, ax = plt.subplots()
    if min_order_plot == 2
        if plot_rpa_fl
            # Plot the bare (LO) and RPA(+FL) results
            ax.plot(
                k_kf_grid_quad,
                c1nl_rpa_means,
                "k";
                linestyle="--",
                label="RPA (vegas)",
            )
            ax.fill_between(
                k_kf_grid_quad,
                (c1nl_rpa_means - c1nl_rpa_stdevs),
                (c1nl_rpa_means + c1nl_rpa_stdevs);
                color="k",
                alpha=0.3,
            )
            ax.plot(k_kf_grid_quad, c1nl_rpa_fl_means, "k"; label="RPA\$+\$FL (vegas)")
            ax.fill_between(
                k_kf_grid_quad,
                (c1nl_rpa_fl_means - c1nl_rpa_fl_stdevs),
                (c1nl_rpa_fl_means + c1nl_rpa_fl_stdevs);
                color="r",
                alpha=0.3,
            )
        end
        ax.plot(k_kf_grid_quad, c1nl_lo, "C0"; linestyle="-", label="\$N=2, T = 0\$ (quad)")
    end
    # Plot the results for each order ξ and compare to RPA(+FL)
    for (i, N) in enumerate(min_order:max_order_plot)
        # Get means and error bars from the result up to this order
        c1nl_N_means, c1nl_N_stdevs =
            Measurements.value.(c1nl_total[N]), Measurements.uncertainty.(c1nl_total[N])
        @assert length(k_kf_grid) == length(c1nl_N_means) == length(c1nl_N_stdevs)
        # ax.plot(k_kf_grid, c1nl_N_means, "o-"; color="C$i", markersize=2, label="\$N=$N\$ ($solver)")
        ax.plot(
            k_kf_grid,
            c1nl_N_means,
            "C$i";
            linestyle="-",
            label="\$N=$N, \\beta = $(beta)\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid,
            (c1nl_N_means - c1nl_N_stdevs),
            (c1nl_N_means + c1nl_N_stdevs);
            color="C$i",
            alpha=0.3,
        )
        ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    end
    # ax.set_xlim(minimum(k_kf_grid), 2.0)
    ax.set_ylim(; top=-0.195)
    ax.legend(; loc="best")
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$C^{(1)nl}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$")
    xloc = 1.7
    yloc = -0.5
    ydiv = -0.05
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
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + 2 * ydiv,
        "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(
        "results/c1nl/c1nl_N=$(max_order_plot)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_total_simul.pdf",
    )
    plt.close("all")
    return
end

main()
