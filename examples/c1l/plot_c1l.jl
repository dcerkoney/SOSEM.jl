using CodecZlib
using DataStructures
using DelimitedFiles
using ElectronLiquid
using FeynmanDiagram
using Interpolations
using JLD2
using Measurements
using PyCall
using PyPlot
using SOSEM

# For style "science"
@pyimport scienceplots

# For saving/loading numpy data
@pyimport numpy as np
@pyimport scipy.interpolate as interp

# @pyimport matplotlib.pyplot as plt
# @pyimport mpl_toolkits.axes_grid1.inset_locator as il

# Vibrant qualitative colour scheme from https://personal.sron.nl/~pault/
const cdict = Dict([
    "orange" => "#EE7733",
    "blue" => "#0077BB",
    "cyan" => "#33BBEE",
    "magenta" => "#EE3377",
    "red" => "#CC3311",
    "teal" => "#009988",
    "grey" => "#BBBBBB",
]);

const vzn_dir = "results/vzn_paper"
const parafilename = "examples/counterterms/data/para.csv"

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
    n_min = 1

    # We measure the 5th order (and above) individually
    if max_order > 4
        max_together = 4
    else
        max_together = max_order
    end

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

    # Save to JLD2?
    save = false
    # save = true

    # UEG parameters for MC integration
    param = ParaMC(;
        order=max_together,
        rs=1.0,
        beta=40.0,
        mass2=1.0,
        isDynamic=false,
        isFock=isFock,  # remove Fock insertions
    )
    if max_order == 5
        param5 = ParaMC(;
            order=5,
            rs=1.0,
            beta=40.0,
            mass2=1.0,
            isDynamic=false,
            isFock=isFock,  # remove Fock insertions
        )
    end
    @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)"

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
    savename =
        "results/data/c1l/c1l_n=$(max_together)_rs=$(param.rs)_beta_ef=$(param.beta)_" *
        "lambda=$(param.mass2)_neval=$(neval)_$(solver)$(ct_string)"
    println(savename)
    orders, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(param))"
        return f[key]
    end
    println("done 4!")
    if max_order >= 5
        # 5th order 
        savename =
            "results/data/c1l/c1l_n=5_rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_neval=$(neval5)_$(solver)$(ct_string)"
        println("Loading 5th order data from $savename...")
        orders5, partitions5, res5 = jldopen("$savename.jld2", "a+") do f
            key = "$(UEG.short(param5))"
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
    zparam = param
    zparam.order = max_order - n_min
    # mu, sw = CounterTerm.getSigma(zparam; parafile=parafilename, root_dir=ENV["SOSEM_HOME"])
    # _, δμ, _ = CounterTerm.sigmaCT(max_order, mu, sw)

    # δμ = load_mu_counterterm(
    #     zparam;
    #     max_order=max_order,
    #     parafilename="examples/counterterms/data/para.csv",
    #     ct_filename="examples/counterterms/data/data_Z.jld2",
    #     # ct_filename="examples/counterterms/data/data_Z$(ct_string_short).jld2",
    #     isFock=isFock,
    #     verbose=1,
    # )
    δμ = UEG_MC.load_mu_counterterm(zparam)
    println("Computed δμ: ", δμ)

    # Reexpand merged data in powers of μ
    c1l = UEG_MC.chemicalpotential_renormalization(
        merged_data,
        δμ;
        n_min=n_min,
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
            "results/data/processed/rs=1.0/rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_$(solver)$(ct_string)_archive1"
        # "lambda=$(param.mass2)_$(solver)$(ct_string)"
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
    println(c1l_total)

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

    # Setup plot styles
    style = PyPlot.matplotlib."style"
    style.use(["science", "std-colors"])
    color = [
        cdict["orange"],
        cdict["blue"],
        cdict["cyan"],
        cdict["magenta"],
        cdict["red"],
        cdict["teal"],
    ]
    # color = [cdict["blue"], cdict["orange"], "green", cdict["red"], "black"]
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")

    # Use LaTex fonts for plots
    rcParams["font.size"] = 16
    rcParams["mathtext.fontset"] = "cm"
    # rcParams["font.family"] = "Times New Roman"

    figure(; figsize=(6, 4))

    # Get RPA value of the local moment at this rs
    k_kf_grid_rpa, c1l_rpa_over_rs2 = load_csv("$vzn_dir/c1l_over_rs2_rpa.csv")
    P = sortperm(k_kf_grid_rpa)
    c1l_rpa_over_rs2_interp =
        linear_interpolation(k_kf_grid_rpa[P], c1l_rpa_over_rs2[P]; extrapolation_bc=Line())
    eTF = param.qTF^2 / (2 * param.me)
    c1l_rpa = c1l_rpa_over_rs2_interp(param.rs) * param.rs^2
    c1l_rpa_over_eTF2 = c1l_rpa * (param.EF / eTF)^2
    println("C⁽¹⁾ˡ (RPA, rs = $(param.rs)): $c1l_rpa")
    println("C⁽¹⁾ˡ / eTF² (RPA, rs = $(param.rs)): $c1l_rpa_over_eTF2")

    # Get QMC value of the local moment at this rs
    k_kf_grid_qmc, c1l_qmc_over_rs2 = load_csv("$vzn_dir/c1l_over_rs2_qmc.csv")
    P = sortperm(k_kf_grid_qmc)
    c1l_qmc_over_rs2_interp =
        linear_interpolation(k_kf_grid_qmc[P], c1l_qmc_over_rs2[P]; extrapolation_bc=Line())
    eTF = param.qTF^2 / (2 * param.me)
    c1l_qmc = c1l_qmc_over_rs2_interp(param.rs) * param.rs^2
    c1l_qmc_over_eTF2 = c1l_qmc * (param.EF / eTF)^2
    println("C⁽¹⁾ˡ (QMC, rs = $(param.rs)): $c1l_qmc")
    println("C⁽¹⁾ˡ / eTF² (QMC, rs = $(param.rs)): $c1l_qmc_over_eTF2")

    # Plot the local moment vs order N and compare to QMC value
    # Ns = orders .+ 1
    Ns = all_orders .+ 1
    axhline(c1l_qmc_over_eTF2; color="k", linestyle="--", label="QMC")
    # marker = "o-"
    # plot(
    #     Ns,
    #     c1l_means,
    #     marker;
    #     color="C0",
    #     markersize=4,
    #     label="RPT ($solver, \$N_\\mathrm{eval} = $neval\$)",
    # )
    # fill_between(
    #     Ns,
    #     c1l_means - c1l_stdevs,
    #     c1l_means + c1l_stdevs;
    #     color="C0",
    #     alpha=0.4,
    # )
    errorbar(
        Ns,
        c1l_means;
        yerr=c1l_stdevs,
        color=cdict["blue"],
        capsize=2,
        markersize=2,
        fmt="o-",
        # markerfacecolor="none",
        label="RPT (\$N_\\mathrm{eval} = $neval\$)",
        # label="RPT ($solver, \$N_\\mathrm{eval} = $neval\$)",
        zorder=10,
    )
    # Darken last point (OOM lower eval)
    if max_order == 5
        errorbar(
            Ns[end],
            c1l_means[end];
            yerr=c1l_stdevs[end],
            color=cdict["cyan"],
            capsize=2,
            markersize=2,
            fmt="o",
            # markerfacecolor="none",
            label="RPT (\$N_\\mathrm{eval} = $neval5\$)",
            # label="RPT ($solver, \$N_\\mathrm{eval} = $neval5\$)",
            zorder=10,
        )
        # fill_between(
        #     Ns[(end - 1):end],
        #     c1l_means[(end - 1):end] - c1l_stdevs[(end - 1):end],
        #     c1l_means[(end - 1):end] + c1l_stdevs[(end - 1):end];
        #     color="mediumblue",
        #     alpha=0.4,
        # )
    end
    legend(; loc="lower right")
    # legend(; loc="best")
    xticks(Ns)
    # xlim(minimum(Ns), maximum(Ns))
    xlim(1.8, 6.2)
    # ylim(nothing, 1.25)
    xlabel("Perturbation order \$N\$")
    # ylabel("\$C^{(1)l} / \\epsilon^2_{\\mathrm{TF}}\$")
    ylabel("\$C^{(1)l} \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$")
    # ylabel("\$S(q)\$")
    # xloc = 1.5
    xloc = 2.1
    yloc = 1.14
    ydiv = -0.035
    text(
        xloc,
        yloc,
        "\$r_s = $(param.rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(param.beta),\$";
        fontsize=14,
    )
    text(
        xloc,
        yloc + ydiv,
        # yloc + 2 * ydiv,
        "\$\\lambda = $(param.mass2)\\epsilon_{\\mathrm{Ry}}\$";
        # "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=14,
    )
    # text(
    #     xloc,
    #     yloc + ydiv,
    #     # "\$\\lambda = $(param.mass2)\\epsilon_{\\mathrm{Ry}}\$";
    #     # "\$\\lambda = $(param.mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
    #     fontsize=14,
    # )
    PyPlot.tight_layout()
    savefig(
        "results/c1l/c1l_N=$(max_order + 1)_rs=$(param.rs)_beta_ef=$(param.beta)_" *
        "lambda=$(param.mass2)_neval=$(neval)_$(neval5)_$(solver)$(ct_string).pdf",
    )
    close("all")

    println("Done!")
    return
end

main()
