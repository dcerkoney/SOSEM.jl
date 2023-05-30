using CodecZlib
using DataStructures
using DelimitedFiles
using ElectronLiquid
using FeynmanDiagram
using Interpolations
using JLD2
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

const vzn_dir = "results/vzn_paper"

function load_csv(filename)
    # assumes csv format: (x, y)
    d = readdlm(filename, ',')
    @assert ndims(d) == 2
    xdata = d[:, 1]
    ydata = d[:, 2]
    return xdata, ydata
end

function c1l2_over_eTF2_vlambda_vlambda(l)
    m = sqrt(l)
    I1 = (l / (l + 4) + log((l + 4) / l) - 1) / 4
    I2 = (l^2 / (l + 4) - (l + 4) + 2l * log((l + 4) / l)) / 48
    I3 = (π / 2m + 2 / (l + 4) - atan(2 / m) / m) / 3
    # I1 = (l / (l + 4) - log(l / (l + 4)) - 1) / 4
    # I2 = (l^2 / (l + 4) - (l + 4) - 2l * log(l / (l + 4))) / 64
    # I3 = 2(2 / (l + 4) - atan(2 / m) / m) / 3
    return (I1 + I2 + I3)
end

"""l = λ / kF^2"""
function c1l2_over_eTF2_v_vlambda(l)
    m = sqrt(l)
    return (π / 3m - 1 / 12) + (l / 12 + 1) * log((4 + l) / l) / 4 - (2 / 3m) * atan(2 / m)
end

"""MC tabulation of the total density."""
function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    # Debug mode
    if isinteractive()
        ENV["JULIA_DEBUG"] = Main
    end

    # Total loop order N
    orders = [1, 2, 3, 4]
    # orders = [1]
    sort!(orders)

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order = minimum(orders)
    max_order = 5
    all_orders = collect(min_order:max_order)
    # max_order = maximum(orders)
    min_order_plot = 1
    max_order_plot = 5

    # Settings
    alpha = 3.0
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 1e10
    neval5 = 1e9

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Remove Fock insertions?
    isFock = false

    # Plot the local moment vs N and compare with VZN QMC data?
    plot = true

    # Save to JLD2?
    save = false
    # save = true

    # UEG parameters for MC integration
    loadparam = ParaMC(;
        order=max_order,
        rs=1.0,
        beta=40.0,
        mass2=1.0,
        isDynamic=false,
        isFock=isFock,  # remove Fock insertions
    )
    @debug "β * EF = $(loadparam.beta), β = $(loadparam.β), EF = $(loadparam.EF)"

    # Distinguish results with different counterterm schemes used in the original run
    ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end
    ct_string_short = ct_string
    if isFock
        ct_string *= "_noFock"
    end

    # Load the raw data
    if max_order > 4
        max_together = 4
    else
        max_together = max_order
    end
    savename =
        "results/data/c1l_n=$(max_together)_rs=$(loadparam.rs)_beta_ef=$(loadparam.beta)_" *
        "lambda=$(loadparam.mass2)_neval=$(neval)_$(solver)$(ct_string)"
    println(savename)
    orders, param, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(loadparam))"
        return f[key]
    end
    println("done 4!")
    if max_order >= 5
        # 5th order 
        savename =
            "results/data/c1l_n=5_rs=$(loadparam.rs)_beta_ef=$(loadparam.beta)_" *
            "lambda=$(loadparam.mass2)_neval=$(neval5)_$(solver)$(ct_string)"
        println("Loading 5th order data from $savename...")
        orders5, param5, partitions5, res5 = jldopen("$savename.jld2", "a+") do f
            key = "$(UEG.short(loadparam))"
            return f[key]
        end
        println("done!")
    end
    println(partitions)
    println(partitions5)

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    println(data)
    # Add Taylor factors 1 / (n_μ! n_λ!)
    for (k, v) in data
        data[k] = v / (factorial(k[2]) * factorial(k[3]))
        # # Extra minus sign for missing factor of (-1)^F = -1?
        # data[k] = [-v / (factorial(k[2]) * factorial(k[3]))]
    end

    # Add 5th order results to data dict
    if max_order >= 5
        data5 = UEG_MC.restodict(res5, partitions5)
        for (k, v) in data5
            data5[k] = v / (factorial(k[2]) * factorial(k[3]))
        end
        merge!(data, data5)
    end
    println(data)

    println("\nPartitions (n_loop, n_λ, n_μ):\n")
    for P in keys(data)
        println(" • Partition $P:\t$(data[P][1])")
    end

    # Merge interaction and loop orders
    merged_data = CounterTerm.mergeInteraction(data)

    println("\nInteraction-merged partitions (n_loop, n_μ, n_λ):\n")
    for Pm in keys(merged_data)
        println(" • Partition $Pm:\t$(merged_data[Pm][1])")
    end

    # Load counterterm data
    ct_filename = "examples/counterterms/data_Z$(ct_string_short).jld2"
    z, μ = UEG_MC.load_z_mu(param; ct_filename=ct_filename)
    # Add Taylor factors to CT data
    for (p, v) in z
        z[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    for (p, v) in μ
        μ[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    δz, δμ = CounterTerm.sigmaCT(max_order, μ, z; isfock=isFock, verbose=1)
    println("Computed δμ: ", δμ)

    # Reexpand merged data in powers of μ
    c1l = UEG_MC.chemicalpotential_renormalization(
        merged_data,
        δμ;
        n_min=1,
        min_order=min_order,
        max_order=max_order,
    )
    c1l_total = UEG_MC.aggregate_orders(c1l)
    @assert all(length(c1l_total[o]) == 1 for o in all_orders)
    c1l_means = [Measurements.value(c1l_total[o][1]) for o in all_orders]
    c1l_stdevs = [Measurements.uncertainty(c1l_total[o][1]) for o in all_orders]

    # Save to JLD2
    if save
        savename =
            "results/data/rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_$(solver)$(ct_string)"
        # "lambda=$(param.mass2)_$(intn_str)$(solver)$(ct_string)"
        f = jldopen("$savename.jld2", "a+"; compress=true)
        for o in all_orders
            N = o + 1
            if o == 5
                num_eval = neval5
            else
                num_eval = neval
            end
            if haskey(f, "c1l") &&
               haskey(f["c1l"], "N=$N") &&
               haskey(f["c1l/N=$N"], "neval=$(num_eval)")
                @warn("replacing existing data for N=$N, neval=$(num_eval)")
                delete!(f["c1l/N=$N"], "neval=$(num_eval)")
            end
            f["c1l/N=$N/neval=$(num_eval)/meas"] = c1l_total[o][1]
            if o == 5
                f["c1l/N=$N/neval=$(num_eval)/param5"] = param5
            else
                f["c1l/N=$N/neval=$(num_eval)/param5"] = param
            end
        end
        close(f)  # close file
    end

    println("\nOrder-by-order local moment contributions:\n")
    for o in keys(c1l)
        println(" • n = $o:\t$(c1l[o][1])")
    end

    println("\nTotal local moment vs order N:\n")
    for o in sort(collect(keys(c1l_total)))
        println(" • N = $o:\t$(c1l_total[o][1])")
    end

    if min_order == 1
        # calc = data[(1, 0, 0)][1]
        calc = c1l_total[1][1]
        exact = c1l2_over_eTF2_v_vlambda(param.mass2 / param.kF^2)
        zscore = stdscore(calc, exact)
        println("\nExact c1l2:\t$exact")
        println("Computed c1l2:\t$calc")
        println("Standard score for c1l2:\t$zscore")
        # @assert zscore ≤ 20
    end

    if plot
        # Use LaTex fonts for plots
        plt.rc("text"; usetex=true)
        plt.rc("font"; family="serif")

        # Get RPA value of the local moment at this rs
        k_kf_grid_rpa, c1l_rpa_over_rs2 = load_csv("$vzn_dir/c1l_over_rs2_rpa.csv")
        P = sortperm(k_kf_grid_rpa)
        c1l_rpa_over_rs2_interp = linear_interpolation(
            k_kf_grid_rpa[P],
            c1l_rpa_over_rs2[P];
            extrapolation_bc=Line(),
        )
        eTF = param.qTF^2 / (2 * param.me)
        c1l_rpa = c1l_rpa_over_rs2_interp(param.rs) * param.rs^2
        c1l_rpa_over_eTF2 = c1l_rpa * (param.EF / eTF)^2
        println("C⁽¹⁾ˡ (RPA, rs = $(param.rs)): $c1l_rpa")
        println("C⁽¹⁾ˡ / eTF² (RPA, rs = $(param.rs)): $c1l_rpa_over_eTF2")

        # Get QMC value of the local moment at this rs
        k_kf_grid_qmc, c1l_qmc_over_rs2 = load_csv("$vzn_dir/c1l_over_rs2_qmc.csv")
        P = sortperm(k_kf_grid_qmc)
        c1l_qmc_over_rs2_interp = linear_interpolation(
            k_kf_grid_qmc[P],
            c1l_qmc_over_rs2[P];
            extrapolation_bc=Line(),
        )
        eTF = param.qTF^2 / (2 * param.me)
        c1l_qmc = c1l_qmc_over_rs2_interp(param.rs) * param.rs^2
        c1l_qmc_over_eTF2 = c1l_qmc * (param.EF / eTF)^2
        println("C⁽¹⁾ˡ (QMC, rs = $(param.rs)): $c1l_qmc")
        println("C⁽¹⁾ˡ / eTF² (QMC, rs = $(param.rs)): $c1l_qmc_over_eTF2")

        # Plot the local moment vs order N and compare to QMC value
        fig, ax = plt.subplots()
        # Ns = orders .+ 1
        Ns = all_orders .+ 1
        ax.axhline(c1l_qmc_over_eTF2; color="k", linestyle="--", label="QMC")
        marker = "o-"
        ax.plot(
            Ns,
            c1l_means,
            marker;
            color="C0",
            markersize=4,
            label="RPT ($solver, \$N_{eval} = $neval\$)",
        )
        ax.fill_between(
            Ns,
            c1l_means - c1l_stdevs,
            c1l_means + c1l_stdevs;
            color="C0",
            alpha=0.4,
        )
        # Darken last point (OOM lower eval)
        if max_order == 5
            ax.plot(
                Ns[end],
                c1l_means[end],
                marker;
                color="mediumblue",
                markersize=4,
                label="RPT ($solver, \$N_{eval} = $neval5\$)",
            )
            ax.fill_between(
                Ns[end],
                c1l_means[end] - c1l_stdevs[end],
                c1l_means[end] + c1l_stdevs[end];
                color="mediumblue",
                alpha=0.4,
            )
        end
        ax.legend(; loc="best")
        ax.set_xticks(Ns)
        ax.set_xlim(minimum(Ns), maximum(Ns))
        # ax.set_ylim(nothing, 1.25)
        ax.set_xlabel("Perturbation order \$N\$")
        # ax.set_ylabel("\$C^{(1)l} / \\epsilon^2_{\\mathrm{TF}}\$")
        ax.set_ylabel("\$C^{(1)l} \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$")
        # ax.set_ylabel("\$S(q)\$")
        # xloc = 1.5
        xloc = 3.5
        yloc = 1.04
        ydiv = -0.035
        ax.text(
            xloc,
            yloc,
            "\$r_s = $(param.rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(param.beta),\\, \\lambda = $(param.mass2)\\epsilon_{\\mathrm{Ry}}\$";
            fontsize=14,
        )
        ax.text(
            xloc,
            yloc + ydiv,
            # yloc + 2 * ydiv,
            "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
            fontsize=12,
        )
        # ax.text(
        #     xloc,
        #     yloc + ydiv,
        #     # "\$\\lambda = $(param.mass2)\\epsilon_{\\mathrm{Ry}}\$";
        #     # "\$\\lambda = $(param.mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
        #     fontsize=14,
        # )
        fig.tight_layout()
        fig.savefig(
            "results/c1l/c1l_N=$(max_order + 1)_rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_neval=$(neval)_$(neval5)_$(solver)$(ct_string).pdf",
        )
        plt.close("all")
    end

    println("Done!")
    return
end

main()
