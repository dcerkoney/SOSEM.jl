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

    rs = 5.0
    beta = 40.0
    neval_lo = 1e10
    neval_hi = 5e10
    neval = min(neval_lo, neval_hi)
    solver = :vegasmc
    expand_bare_interactions = false

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Combining results from low & high accuracy runs?
    multirun = true

    # # Scanning λ to check relative convergence wrt perturbation order
    # lambdas_lo = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
    # lambdas_hi = [0.1, 0.125, 0.15]

    # Optimal lambdas_lo on this grid for rs = 1, 2, 5
    lambda_star = missing
    if rs ≈ 1
        lambda_star = 1.0
    elseif rs ≈ 2
        lambda_star = 0.4
    elseif rs ≈ 5
        lambda_star = 0.1375  # estimate at midpoint
    end

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
        "beta_ef=$(beta)_neval=$(neval_lo)_" *
        "$(intn_str)$(solver)$(ct_string)_vs_lambda"
    settings, params_lo, kgrid, lambdas_lo, partitions_lo, res_list_lo =
        jldopen("$savename.jld2", "a+") do f
            key = "$(short_no_lambda(loadparam))"
            return f[key]
        end
    if multirun
        # Load the higher-accuracy results from JLD2
        savename =
            "results/data/c1nl_k=0_n=$(max_order)_rs=$(rs)_" *
            "beta_ef=$(beta)_neval=$(neval_hi)_" *
            "$(intn_str)$(solver)$(ct_string)_vs_lambda"
        settings_hi, params_hi, kgrid_hi, lambdas_hi, partitions_hi, res_list_hi =
            jldopen("$savename.jld2", "a+") do f
                key = "$(short_no_lambda(loadparam))"
                return f[key]
            end
        @assert settings == settings_hi
        @assert kgrid == kgrid_hi

        # Merge lambda grid data in sorted order, storing the permutation vector
        all_lambdas = union(lambdas_lo, lambdas_hi)
        P = sortperm(all_lambdas)
        sort!(all_lambdas)
        # Merge params in the same order
        params = union(params_lo, params_hi)[P]
    else
        all_lambdas = lambdas_lo
        params = params_lo
        lambdas_hi = []
        res_list_hi = []
    end

    c1nl_totals = []
    for (i, lambda) in enumerate(all_lambdas)
        # Use highest-accuracy results available at this lambda
        idx_in_lambdas_lo = findall(x -> x == lambda, lambdas_lo)
        idx_in_lambdas_hi = findall(x -> x == lambda, lambdas_hi)
        idx_res = !isempty(idx_in_lambdas_hi) ? idx_in_lambdas_hi : idx_in_lambdas_lo
        @assert length(idx_res) == 1
        idx_res = idx_res[1]

        # UEG parameters for MC integration
        loadparam =
            ParaMC(; order=max_order, rs=rs, beta=beta, mass2=lambda, isDynamic=false)

        # Convert results to a Dict of measurements at each order with interaction counterterms merged
        if lambda ∈ lambdas_hi
            data = UEG_MC.restodict(res_list_hi[idx_res], partitions_hi)
        else
            data = UEG_MC.restodict(res_list_lo[idx_res], partitions_lo)
        end
        for (k, v) in data
            data[k] = v / (factorial(k[2]) * factorial(k[3]))
        end
        merged_data = CounterTerm.mergeInteraction(data)
        println([k for (k, _) in merged_data])

        # if min_order_plot == 2 && min_order > 2
        if min_order_plot == 2
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
    end
    @assert length(c1nl_totals) == length(all_lambdas)

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # Plot the results for each order ξ vs lambda and compare to RPA(+FL)
    colors_high_acc = ["darkblue", "chocolate", "darkgreen"]
    fig, ax = plt.subplots()
    if !ismissing(lambda_star)
        token = rs ≈ 5 ? "\\approx" : "="
        ax.axvline(
            lambda_star;
            linestyle="--",
            color="dimgray",
            label="\$\\lambda^\\star $token $lambda_star\$",
        )
    end
    if min_order_plot == 2
        ax.plot(
            all_lambdas,
            DiagGen.c1nl_ueg.exact_unif * one.(all_lambdas),
            "-";
            color="k",
            markersize=3,
            label="\$N=2\$ (exact, \$T = 0\$)",
        )
    end
    c1nl_unif_N_means = []
    c1nl_unif_N_stdevs = []
    idxs_high_acc = findall(x -> x ∈ lambdas_hi, all_lambdas)
    for (j, N) in enumerate(min_order:max_order_plot)
        # Get means and error bars from the result up to this order
        c1nl_unif_N_means = [c1nl_totals[j][N][1].val for j in eachindex(all_lambdas)]
        c1nl_unif_N_stdevs = [c1nl_totals[j][N][1].err for j in eachindex(all_lambdas)]
        ax.plot(
            all_lambdas,
            c1nl_unif_N_means,
            "o-";
            color="C$(j-1)",
            markersize=3,
            label="\$N=$N\$ ($solver)",
        )
        ax.fill_between(
            all_lambdas,
            (c1nl_unif_N_means - c1nl_unif_N_stdevs),
            (c1nl_unif_N_means + c1nl_unif_N_stdevs);
            color="C$(j-1)",
            alpha=0.3,
        )
        if multirun
            # Overlay darkened colors for points in higher-accuracy run
            ax.plot(
                all_lambdas[idxs_high_acc],
                c1nl_unif_N_means[idxs_high_acc],
                "o-";
                color=colors_high_acc[j],
                markersize=3,
            )
            ax.fill_between(
                all_lambdas[idxs_high_acc],
                (c1nl_unif_N_means[idxs_high_acc] - c1nl_unif_N_stdevs[idxs_high_acc]),
                (c1nl_unif_N_means[idxs_high_acc] + c1nl_unif_N_stdevs[idxs_high_acc]);
                color=colors_high_acc[j],
                alpha=0.3,
            )
        end
    end
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(; top=-0.2, bottom=-1.4)
    ax.legend(; loc="best")
    ax.set_xlabel("\$\\lambda\$ (Ry)")
    ax.set_ylabel(
        "\$C^{(1)nl}(k=0,\\, \\lambda) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    xloc = 0.2
    yloc = -0.3
    ydiv = -0.125
    ax.text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    # ax.text(xloc, yloc + ydiv, "\$N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$"; fontsize=14)
    ax.text(
        xloc,
        yloc + ydiv,
        # yloc + 2 * ydiv,
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
