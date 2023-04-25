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

    rs = 1.0
    beta = 40.0
    mass2 = 1.0
    solver = :vegasmc
    expand_bare_interactions = false

    neval = 5e10

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    n_min = 2  # True minimal loop order for this observable
    min_order = n_min
    max_order = 5
    min_order_plot = n_min
    max_order_plot = 5

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Manually perform chemical potential renormalization
    renorm_mu_lo_ex = false  # at lowest order
    renorm_mu_nlo_ex = false  # at next-lowest order

    # Save total results
    save = true

    plotparam =
        UEG.ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

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

    # Load the results from JLD2
    filename =
        "results/data/c1d_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)$(ct_string)"
    settings, param, kgrid, partitions, res = jldopen("$filename.jld2", "a+") do f
        key = "$(UEG.short(plotparam))"
        return f[key]
    end

    # # Load the results using new JLD2 format
    # if max_order == 5
    #     max_together = 4
    # else
    #     max_together = max_order
    # end
    # filename =
    #     "results/data/rs=$(rs)_beta_ef=$(beta)_" *
    #     "lambda=$(mass2)_$(intn_str)$(solver)$(ct_string)"
    # f = jldopen("$filename.jld2", "a+"; compress=true)
    # key = "c1d_n_min=$(min_order)_n_max=$(max_together)_neval=$(neval34)"
    # res = f["$key/res"]
    # settings = f["$key/settings"]
    # param = f["$key/param"]
    # kgrid = f["$key/kgrid"]
    # partitions = f["$key/partitions"]
    # if max_order == 5
    #     # 5th order 
    #     filename =
    #         "results/data/rs=$(rs)_beta_ef=$(beta)_" *
    #         "lambda=$(mass2)_$(intn_str)$(solver)$(ct_string)"
    #     f5 = jldopen("$filename.jld2", "a+"; compress=true)
    #     key5 = "c1d_n_min=$(max_order)_n_max=$(max_order)_neval=$(neval5)"
    #     res5 = f5["$(key5)/res"]
    #     settings5 = f5["$(key5)/settings"]
    #     param5 = f5["$(key5)/param"]
    #     kgrid5 = f5["$(key5)/kgrid"]
    #     partitions5 = f5["$(key5)/partitions"]
    # end
    # # Close the JLD2 file
    # close(f)

    print(settings)
    print(param)
    print(kgrid)
    print(res)
    print(partitions)

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

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
    c1d_bare_interp =
        linear_interpolation(k_kf_grid_quad, c1d_bare_quad; extrapolation_bc=Line())
    c1d2_exact = c1d_bare_interp(k_kf_grid)

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    for (k, v) in data
        data[k] = v / (factorial(k[2]) * factorial(k[3]))
    end
    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])

    if min_order_plot == 2 && min_order > 2
        # Set bare result manually using exact data to avoid systematic error in (2,0,0) calculation
        merged_data[(2, 0)] = measurement.(c1d2_exact, 0.0)  # treat quadrature data as numerically exact
    end

    # Get total data
    if renorm_mu
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
        δz, δμ = CounterTerm.sigmaCT(max_order - n_min, μ, z; verbose=1)
        println("Computed δμ: ", δμ)
        δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
        if renorm_mu_lo_ex && max_order_plot ≥ 3
            # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1}
            c1d3 = merged_data[(3, 0)] + δμ1_exact * merged_data[(2, 1)]
            c1d4 =
                merged_data[(4, 0)] +
                δμ1_exact * merged_data[(3, 1)] +
                δμ1_exact^2 * merged_data[(2, 2)] +
                δμ[2] * merged_data[(2, 1)]
            c1d = SortedDict(3 => c1d3, 4 => c1d4)
            if min_order_plot == 2
                c1d[2] = c1d2_exact
            end
        else
            c1d = UEG_MC.chemicalpotential_renormalization_sosem(
                merged_data,
                δμ;
                lowest_order=2,
                min_order=min(min_order, min_order_plot),
                max_order=max(max_order, max_order_plot),
            )
            # Test manual renormalization with exact lowest-order chemical potential
            if !renorm_mu_lo_ex && max_order >= 3
                # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1}
                c1d3_manual = merged_data[(3, 0)] + δμ1_exact * merged_data[(2, 1)]
                # C⁽¹⁾₄ = C⁽¹⁾_{4,0} + δμ₁ C⁽¹⁾_{3,1} + ⋯
                c1d4_manual =
                    c1d3_manual +
                    merged_data[(4, 0)] +
                    δμ1_exact * merged_data[(3, 1)] +
                    δμ1_exact^2 * merged_data[(2, 2)] +
                    δμ[2] * merged_data[(2, 1)]
                println("Exact δμ₁: ", δμ1_exact)
                println("Computed δμ₁: ", δμ[1])
                println("Standard score for calculated δμ₁: $(stdscore(δμ[1], δμ1_exact))")
                stdscores = stdscore.(c1d[3], c1d3_manual)
                worst_score = argmax(abs, stdscores)
                println(
                    "Worst standard score for total result to 3rd " *
                    "order (auto vs exact+manual): $worst_score",
                )
                stdscores = stdscore.(c1d[3] + c1d[4], c1d4_manual)
                worst_score = argmax(abs, stdscores)
                println(
                    "Worst standard score for total result to 4th " *
                    "order (auto vs exact+manual): $worst_score",
                )
            end
        end
    else
        c1d = merged_data
    end

    # Aggregate the full results for C⁽¹ᶜ⁾ up to order N
    if renorm_mu_lo_ex
        c1d_total = Dict(3 => c1d3_manual, 4 => c1d4_manual)
        if min_order_plot == 2
            c1d_total[2] = c1d2_exact
            c1d_total[3] += c1d2_exact
            c1d_total[4] += c1d2_exact
        end
    else
        c1d_total = UEG_MC.aggregate_orders(c1d)
    end

    println(settings)
    println(UEG.paraid(param))
    println(partitions)
    println(res)

    # Plot the results
    fig, ax = plt.subplots()

    if min_order_plot == 2
        # Plot the bare (LO) result; there are no RPA(+FL) corrections for the class (d) moment
        ax.plot(k_kf_grid_quad, c1d_bare_quad, "--"; color="C0", label="\$N=2\$ (quad)")
    end

    if save
        savename =
            "results/data/rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_$(intn_str)$(solver)$(ct_string)"
        f = jldopen("$savename.jld2", "a+"; compress=true)
        # NOTE: no bare result for c1b observable (accounted for in c1b0)
        for N in min_order_plot:max_order
            if haskey(f, "c1d") &&
               haskey(f["c1d"], "N=$N") &&
               haskey(f["c1d/N=$N"], "neval=$(neval)")
                @warn("replacing existing data for N=$N, neval=$(neval)")
                delete!(f["c1d/N=$N"], "neval=$(neval)")
            end
            f["c1d/N=$N/neval=$neval/meas"] = c1d_total[N]
            f["c1d/N=$N/neval=$neval/settings"] = settings
            f["c1d/N=$N/neval=$neval/param"] = param
            f["c1d/N=$N/neval=$neval/kgrid"] = kgrid
        end
    end

    # Plot for each aggregate order
    for (i, N) in enumerate(min_order:max_order_plot)
        # Get means and error bars from the result up to this order
        means = Measurements.value.(c1d_total[N])
        stdevs = Measurements.uncertainty.(c1d_total[N])
        # Data gets noisy above 3rd loop order
        marker = "o-"
        # marker = "-"
        # marker = N > 3 ? "o-" : "-"
        ax.plot(
            k_kf_grid,
            means,
            marker;
            markersize=2,
            color="C$(i-1)",
            label="\$N=$(N)\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            color="C$(i-1)",
            alpha=0.4,
        )
        if !renorm_mu_lo_ex && max_order <= 3 && N == 3
            ax.plot(
                k_kf_grid,
                Measurements.value.(c1d3_manual);
                color="r",
                linestyle="--",
                label="\$N=1\$ (manual, vegasmc)",
            )
        end
    end
    ax.legend(; loc="best")
    ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    # ax.set_xlim(0.5, 1.5)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$C^{(1d)}_{N}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    # xloc = 0.5
    # yloc = -0.075
    # ydiv = -0.009

    # xloc = 1.6
    # yloc = 2.0
    # ydiv = -0.3
    xloc = 1.6
    yloc = 0.8
    ydiv = -0.1
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
        "results/c1d/c1d_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)$(ct_string)" *
        "$(renorm_string)_total.pdf",
    )
    plt.close("all")
    return
end

main()
