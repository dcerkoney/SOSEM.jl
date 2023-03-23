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

function paraid_no_lambda(p::ParaMC)
    return Dict(
        "dim" => p.dim,
        "rs" => p.rs,
        "beta" => p.beta,
        "Fs" => p.Fs,
        "Fa" => p.Fa,
        "massratio" => p.massratio,
        "spin" => p.spin,
        "isFock" => p.isFock,
        "isDynamic" => p.isDynamic,
    )
end

function short_no_lambda(p::ParaMC)
    return join(["$(k)_$(v)" for (k, v) in sort(paraid_no_lambda(p))], "_")
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
    neval = 1e7
    solver = :vegasmc
    expand_bare_interactions = false

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Scanning λ to check relative convergence wrt perturbation order
    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]

    n_min = 2  # lowest possible loop order for this observable
    min_order = 2
    max_order = 4

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order_plot = 2
    max_order_plot = 4

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions
        intn_str = "no_bare_"
    end

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # Load the results from JLD2
    loadparam = ParaMC(; order=max_order, rs=rs, beta=beta, isDynamic=false)
    savename =
        "results/data/c1nl_k=0_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_neval=$(neval)_" *
        "$(intn_str)$(solver)$(ct_string)_vs_lambda"
    settings, params, kgrid, lambdas, partitions, res_list =
        jldopen("$savename.jld2", "a+") do f
            key = "$(short_no_lambda(loadparam))"
            return f[key]
        end

    c1nl_totals = []
    for (i, lambda) in enumerate(lambdas)
        # UEG parameters for MC integration
        loadparam =
            ParaMC(; order=max_order, rs=rs, beta=beta, mass2=lambda, isDynamic=false)

        # Convert results to a Dict of measurements at each order with interaction counterterms merged
        data = UEG_MC.restodict(res_list[i], partitions)
        for (k, v) in data
            data[k] = v / (factorial(k[2]) * factorial(k[3]))
        end
        merged_data = CounterTerm.mergeInteraction(data)
        println([k for (k, _) in merged_data])

        if min_order_plot == 2 && min_order > 2
            # if min_order_plot == 2
            # Set bare result manually using exact value to avoid statistical error in (2,0,0) calculation
            merged_data[(2, 0)] = [measurement(DiagGen.c1nl_ueg.exact_unif, 0.0)]
        end

        # Reexpand merged data in powers of μ
        ct_filename = "examples/counterterms/data_Z$(ct_string).jld2"
        z, μ = UEG_MC.load_z_mu(params[i]; ct_filename=ct_filename)
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
        c1nl_unif = UEG_MC.chemicalpotential_renormalization_sosem(
            merged_data,
            δμ;
            lowest_order=2,
            min_order=min(min_order, min_order_plot),
            max_order=max(max_order, max_order_plot),
        )
        # Test manual renormalization with exact lowest-order chemical potential
        δμ1_exact = UEG_MC.delta_mu1(params[i])  # = ReΣ₁[λ](kF, 0)
        # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1}
        c1nl3_manual =
            merged_data[(2, 0)] + merged_data[(3, 0)] + δμ1_exact * merged_data[(2, 1)]
        stdscores = stdscore.(c1nl_unif[2] + c1nl_unif[3], c1nl3_manual)
        worst_score = argmax(abs, stdscores)
        println("Exact δμ₁: ", δμ1_exact)
        println("Computed δμ₁: ", δμ[1])
        println(
            "Worst standard score for total result to 3rd " *
            "order (auto vs exact+manual): $worst_score",
        )
        # Aggregate the full results for C⁽¹ᶜ⁾ up to order N
        push!(c1nl_totals, UEG_MC.aggregate_orders(c1nl_unif))

        println(settings)
        println(UEG.paraid(params[i]))
        println(partitions)
        println(res_list[i])
    end

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # Plot the results for each order ξ vs lambda and compare to RPA(+FL)
    fig, ax = plt.subplots()
    # ax.axvline(1.0; linestyle="--", color="dimgray", label="\$\\lambda^\\star = 1\$")
    if min_order_plot == 2
        ax.plot(
            lambdas,
            DiagGen.c1nl_ueg.exact_unif * one.(lambdas),
            "-";
            color="k",
            markersize=3,
            label="\$N=2\$ (exact, \$T = 0\$)",
        )
    end
    c1nl_unif_N_means = []
    c1nl_unif_N_stdevs = []
    for (j, N) in enumerate(min_order:max_order_plot)
        # Get means and error bars from the result up to this order
        c1nl_unif_N_means = [c1nl_totals[i][j][1].val for i in eachindex(lambdas)]
        c1nl_unif_N_stdevs = [c1nl_totals[i][j][1].err for i in eachindex(lambdas)]
        ax.plot(
            lambdas,
            c1nl_unif_N_means,
            "o-";
            color="C$(j-1)",
            markersize=3,
            label="\$N=$N\$ ($solver)",
        )
        ax.fill_between(
            lambdas,
            (c1nl_unif_N_means - c1nl_unif_N_stdevs),
            (c1nl_unif_N_means + c1nl_unif_N_stdevs);
            color="C$(j-1)",
            alpha=0.3,
        )
    end
    # ax.set_xlim(0.5, 3.0)
    ax.set_ylim(; bottom=-0.75)
    ax.legend(; loc="best")
    ax.set_xlabel("\$\\lambda\$ (Ry)")
    ax.set_ylabel(
        "\$C^{(1)nl}(k=0,\\, \\lambda) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    xloc = 1.325
    yloc = -0.54
    ydiv = -0.025
    ax.text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta), N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + ydiv,
        "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(
        "results/c1nl/c1nl_k=0_rs=$(rs)_beta_ef=$(beta)_" *
        "neval=$(neval)_$(intn_str)$(solver)_vs_lambda.pdf",
    )
    plt.close("all")
    return
end

main()
