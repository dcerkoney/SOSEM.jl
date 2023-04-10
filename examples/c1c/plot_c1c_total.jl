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

# NOTE: Call from main project directory as: julia examples/c1c/plot_c1c_total.jl

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

    neval34 = 5e10
    neval5 = 5e10
    neval = max(neval34, neval5)
    # neval = neval34

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    n_min = 2  # True minimal loop order for this observable
    min_order = 2
    max_order = 2
    min_order_plot = 2
    max_order_plot = 2
    # @assert max_order ≥ 3

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

    # colors = ["orchid", "cornflowerblue", "turquoise", "chartreuse", "greenyellow"]
    # markers = ["-", "-", "-", "-", "-"]

    # Load the order 3-4 results from JLD2 (and μ data from csv, if applicable)
    if max_order == 5
        max_together = 4
    else
        max_together = max_order
    end
    savename =
        "results/data/c1c_n=$(max_together)_rs=$(rs)_" *
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
            "results/data/c1c_n=$(max_order)_rs=$(rs)_" *
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

    # Load C⁽¹ᵈ⁾₂ quadrature results and interpolate on k_kf_grid
    rs_quad = 1.0
    # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    param_quad = Parameter.atomicUnit(0, rs_quad)    # (dimensionless T, rs)
    eTF_quad = param_quad.qTF^2 / (2 * param_quad.me)
    sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=40.0.npz")

    # Bare results (stored in Hartree a.u.)
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    c1c_bare_quad = sosem_quad.get("bare_c") / eTF_quad^2

    # Non-dimensionalize bare and RPA+FL non-local moments
    rs_lo = 1.0
    sosem_lo = np.load("results/data/soms_rs=$(rs_lo)_beta_ef=40.0.npz")
    # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    param_lo = Parameter.atomicUnit(0, rs_lo)    # (dimensionless T, rs)
    eTF_lo = param_lo.qTF^2 / (2 * param_lo.me)

    # Bare results (stored in Hartree a.u.)
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    c1c_bare_quad = sosem_lo.get("bare_c") / eTF_lo^2

    # Interpolate bare results and downsample to coarse k_kf_grid
    c1c_bare_interp =
        linear_interpolation(k_kf_grid_quad, c1c_bare_quad; extrapolation_bc=Line())
    c1c2_exact = c1c_bare_interp(k_kf_grid)

    if min_order_plot == 2
        # Set bare result manually using exact data to avoid statistical error in (2,0,0) calculation
        merged_data[(2, 0)] = measurement.(c1c2_exact, 0.0)  # treat quadrature data as numerically exact
    end

    # Get total data
    if renorm_mu
        if renorm_mu_lo_ex && max_order_plot == 3
            δμ1 = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
            # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1}
            c1c2 = merged_data[(2, 0)]
            c1c3 = merged_data[(3, 0)] + δμ1 * merged_data[(2, 1)]
            c1c = [c1c2, c1c3]
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
            c1c = UEG_MC.chemicalpotential_renormalization_sosem(
                merged_data,
                δμ;
                lowest_order=2,
                min_order=min(min_order, min_order_plot),
                max_order=max(max_order, max_order_plot),
            )
            # Test manual renormalization with exact lowest-order chemical potential
            if !renorm_mu_lo_ex && max_order >= 3
                δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
                # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1}
                c1c3_manual =
                    merged_data[(2, 0)] +
                    merged_data[(3, 0)] +
                    δμ1_exact * merged_data[(2, 1)]
                c1c3 = c1c[2] + c1c[3]
                stdscores = stdscore.(c1c3, c1c3_manual)
                worst_score = argmax(abs, stdscores)
                println("Exact δμ₁: ", δμ1_exact)
                println("Computed δμ₁: ", δμ[1])
                println(
                    "Worst standard score for total result to 3rd " *
                    "order (auto vs exact+manual): $worst_score",
                )
            end
        end
    else
        c1c = merged_data
    end

    # Aggregate the full results for C⁽¹ᶜ⁾ up to order N
    if renorm_mu_lo_ex
        c1c_total = Dict(2 => c1c2, 3 => c1c2 + c1c3)
    else
        c1c_total = UEG_MC.aggregate_orders(c1c)
    end

    println(settings)
    println(UEG.paraid(param))
    println(partitions)
    println(res)

    # Plot the results
    fig, ax = plt.subplots()

    if min_order_plot == 2
        # Plot the bare (LO) result; there are no RPA(+FL) corrections for the class (c) moment
        ax.plot(k_kf_grid_quad, c1c_bare_quad, "C0"; label="\$N=2\$ (quad)")
    end

    if save
        savename =
            "results/data/rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_$(intn_str)$(solver)$(ct_string)"
        f = jldopen("$savename.jld2", "a+"; compress=true)
        # NOTE: no bare result for c1b observable (accounted for in c1b0)
        for N in min_order_plot:max_order
            num_eval = N == 5 ? neval5 : neval34
            # num_eval = neval
            if haskey(f, "c1c") &&
               haskey(f["c1c"], "N=$N") &&
               haskey(f["c1c/N=$N"], "neval=$(num_eval)")
                @warn("replacing existing data for N=$N, neval=$(num_eval)")
                delete!(f["c1c/N=$N"], "neval=$(num_eval)")
            end
            f["c1c/N=$N/neval=$num_eval/meas"] = c1c_total[N]
            f["c1c/N=$N/neval=$num_eval/settings"] = settings
            f["c1c/N=$N/neval=$num_eval/param"] = param
            f["c1c/N=$N/neval=$num_eval/kgrid"] = kgrid
        end
    end

    # Plot for each aggregate order
    for (i, N) in enumerate(min_order:max_order_plot)
        # if N == 2
        #     continue
        # end
        # NOTE: Currently using a different kgrid at order 5
        k_over_kfs = N == 5 ? k_kf_grid5 : k_kf_grid
        # Get means and error bars from the result up to this order
        means = Measurements.value.(c1c_total[N])
        stdevs = Measurements.uncertainty.(c1c_total[N])
        # Data gets noisy above 3rd loop order
        marker = "o-"
        # marker = "-"
        # marker = N ≥ 3 ? "o-" : "-"
        ax.plot(
            k_over_kfs,
            means,
            marker;
            markersize=2,
            color="C$i",
            label="\$N=$(N)\$ ($solver)",
        )
        ax.fill_between(k_over_kfs, means - stdevs, means + stdevs; color="C$i", alpha=0.4)
        if !renorm_mu_lo_ex && max_order <= 3 && N == 3
            ax.plot(
                k_over_kfs,
                Measurements.value.(c1c3_manual);
                color="r",
                linestyle="--",
                label="\$N=1\$ (manual, vegasmc)",
            )
        end
    end
    ax.legend(; loc="lower right")
    ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$C^{(1c)}_{N}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    # xloc = 0.5
    # yloc = -0.075
    # ydiv = -0.009
    xloc = 1.5
    yloc = -0.5
    ydiv = -0.095
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
        "results/c1c/c1c_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)$(ct_string)" *
        "$(renorm_string)_total.pdf",
    )
    plt.close("all")
    return
end

main()
