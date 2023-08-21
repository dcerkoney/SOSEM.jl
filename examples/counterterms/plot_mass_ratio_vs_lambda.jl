using CodecZlib
using ElectronLiquid
using ElectronGas
using Interpolations
using JLD2
using Measurements
using Parameters
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

function mass_ratio_screened_fock(param::UEG.ParaMC)
    @unpack rs, kF, mass2 = param
    alpha = (4 / 9π)^(1 / 3)
    c_mu = mass2 / (mass2 + (2 * kF)^2)
    return (1 - (alpha * rs / π) * (1 + ((1 + c_mu) / (1 - c_mu)) * log(sqrt(c_mu))))^(-1)
end

function main()
    # # Change to project directory
    # if haskey(ENV, "SOSEM_CEPH")
    #     cd(ENV["SOSEM_CEPH"])
    # elseif haskey(ENV, "SOSEM_HOME")
    #     cd(ENV["SOSEM_HOME"])
    # end

    # Setup plot styles
    style = PyPlot.matplotlib."style"
    style.use(["science", "std-colors"])
    color = [
        "k",
        cdict["orange"],
        cdict["blue"],
        cdict["cyan"],
        cdict["magenta"],
        cdict["red"],
        # cdict["teal"],
    ]
    # color = [cdict["blue"], cdict["orange"], "green", cdict["red"], "black"]
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")

    # Use LaTex fonts for plots
    rcParams["font.size"] = 16
    rcParams["mathtext.fontset"] = "cm"
    # rcParams["font.family"] = "Times New Roman"

    fig = figure(; figsize=(6, 4))

    beta = 40.0
    solver = :mcmc
    # NOTE: neval4 = 1e11, neval5 = 1e10
    neval = 1e11

    # # NOTE: neval ∈ {1e10, 5e10, or 1e11} and varies case-by-case => no longer track it in plots!
    # #       It is only important to display it for the final mass results.
    # neval = 1e10  # 5th order, rs=5
    # neval = 5e10  # rs=1, rs=5
    # neval = 1e11    # rs=2, 3, 4, and N=4, rs=5

    ### rs = 1 ###
    # rs = 1.0
    # lambdas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
    # lambdas5 = nothing
    #lambdas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    ### rs = 2 ###
    # rs = 2.0
    # lambdas = [0.5, 1.0, 1.25, 1.5, 1.625, 1.75, 1.875, 2.0, 2.5, 3.0]
    # lambdas5 = [1.625, 1.75, 1.875, 2.0, 2.5]
    # lambdas5 = nothing
    # # lambdas = [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    ### rs = 3 ###
    rs = 3.0
    lambdas = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 1.75, 2.0]
    lambdas5 = [1.0, 1.125, 1.25, 1.5, 1.75, 2.0]

    # rs = 3.0
    # lambdas = [0.75, 0.875, 1.0, 1.125, 1.25, 1.5]
    # lambdas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    # lambdas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]

    ### rs = 4 ###
    # rs = 4.0
    # lambdas = [0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 2.0]
    # lambdas5 = [0.875, 1.0, 1.125, 1.25, 1.5]

    # lambdas5 = [0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 2.0]

    # lambdas = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 2.0]
    # lambdas = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5]
    # lambdas = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125]
    # lambdas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0]
    # lambdas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]

    ### rs = 5 ###
    # rs = 5.0
    # lambda = [0.1, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5]
    # lambdas = [0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5]
    # lambdas5 = [0.8125, 0.875, 0.9375]
    #
    # lambdas = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0]
    # lambdas = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0]
    # lambdas = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    min_order = 0
    # max_order = 4
    max_order = 5

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order_plot = 1
    # max_order_plot = 4
    max_order_plot = 5

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Remove Fock insertions?
    isFock = false

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # Use derivative estimate with δK = dks[idk] (grid points kgrid[ikF] and kgrid[ikF + idk])
    idk = 3   # dk = 0.06, max δK on new grid
    idk5 = 3  # dk = 0.06, max δK on new grid

    # Momentum spacing for finite-difference derivative of Sigma (in units of para.kF)
    δK = 0.01
    δK5 = 0.01

    # kgrid indices & spacings
    dks = δK * collect(-6:2:6)
    dks5 = δK * collect(-6:2:6)
    ikF = searchsortedfirst(dks, 0)
    ikF5 = searchsortedfirst(dks5, 0)
    dk = dks[ikF + idk]
    dk5 = dks5[ikF5 + idk5]
    @assert dk == dk5
    @assert dk ≈ 0.06

    # max_together = max_order            # all orders are run together
    max_together = min(max_order, 4)  # 5th order is run separately

    local param
    mass_ratios_lambda_vs_N = []
    for lambda in lambdas
        # UEG parameters for MC integration
        loadparam = ParaMC(;
            order=max_together,
            rs=rs,
            beta=beta,
            mass2=lambda,
            isDynamic=false,
            isFock=isFock,
        )
        # Load mass ratio data for each idk
        savename_mass = "data/mass_ratio"
        print("Loading data from $savename_mass (lambda = $lambda)...")
        ngrid, kgrid, mass_ratio = jldopen("$savename_mass.jld2", "r") do f
            key = "$(UEG.short(loadparam))_idk=$(idk)"
            param = UEG.ParaMC(string(split(key, "_idk=")[1]))
            return f[key]
        end
        # Derive kgrid indices & spacing δK from data
        data_dks = round.(kgrid / param.kF .- 1; sigdigits=13)
        @assert data_dks[ikF + idk] ≈ dk
        push!(mass_ratios_lambda_vs_N, mass_ratio)
        println("done!")
        @assert param.order == max_together
    end
    @assert allequal(length(row) for row in mass_ratios_lambda_vs_N)
    mass_ratios_N_vs_lambda = [
        [mass_ratios_lambda_vs_N[i][j] for i in eachindex(lambdas)] for
        j in eachindex(mass_ratios_lambda_vs_N[1])
    ]
    # 5th order mass ratio vs lambda.
    # NOTE: We only need to keep the 5th order results for the small lambda list;
    #       more accurate results for N ≤ 4 are obtained from the full lambda list
    local param5, mass_ratios_5_vs_lambda5
    if max_order == 5 && max_together != 5
        mass_ratios_5_vs_lambda5 = []
        for lambda5 in lambdas5
            # UEG parameters for MC integration
            loadparam = ParaMC(;
                order=5,
                rs=rs,
                beta=beta,
                mass2=lambda5,
                isDynamic=false,
                isFock=isFock,
            )
            # Load mass ratio data for each idk
            savename_mass = "data/mass_ratio"
            print("Loading 5th-order data from $savename_mass...")
            ngrid5, kgrid5, mass_ratio = jldopen("$savename_mass.jld2", "r") do f
                key = "$(UEG.short(loadparam))_idk=$(idk5)"
                param5 = UEG.ParaMC(string(split(key, "_idk=")[1]))
                return f[key]
            end
            println(mass_ratio)
            # N = 0, 1, 2, 3, 4, 5 => idx5 = 6
            @assert eachindex(mass_ratio) == Base.OneTo(6)
            push!(mass_ratios_5_vs_lambda5, mass_ratio[6])
            println("done!")
            @assert param5.order == 5
        end
    else
        mass_ratios_5_vs_lambda5 = nothing
    end

    # println("mass_ratios_lambda_vs_N:\n$mass_ratios_lambda_vs_N")
    println("\nmass_ratios_N_vs_lambda:\n$mass_ratios_N_vs_lambda")
    if max_order == 5 && max_together != 5
        println("\nmass_ratios_5_vs_lambda5:\n$mass_ratios_5_vs_lambda5")
    end

    # Plot the results for each order ξ vs lambda
    for (i, N) in enumerate(0:max_together)
        N == 0 && continue  # Ignore zeroth order
        # Get means and error bars from the result up to this order
        means = Measurements.value.(mass_ratios_N_vs_lambda[i])
        stdevs = Measurements.uncertainty.(mass_ratios_N_vs_lambda[i])
        # small lambda list at fifth order, full list otherwise
        errorbar(
            lambdas,
            means;
            yerr=stdevs,
            fmt="o-",
            color=color[i - 1],
            # capsize=4,
            markersize=3,
            label="\$N=$N\$",
            # label="\$N=$N\$ ($solver)",
            zorder=10 * i,
        )
        # plot(
        #     lambdas,
        #     means,
        #     "o-";
        #     color=color[i - 1],
        #     markersize=3,
        #     label="\$N=$N\$ ($solver)",
        # )
        # fill_between(
        #     lambdas,
        #     (means - stdevs),
        #     (means + stdevs);
        #     color=color[i - 1],
        #     alpha=0.3,
        # )
    end
    if max_order == 5 && max_together != 5
        # Plot 5th order over small lambda list
        # Get means and error bars from the result up to this order
        means5 = Measurements.value.(mass_ratios_5_vs_lambda5)
        stdevs5 = Measurements.uncertainty.(mass_ratios_5_vs_lambda5)
        # small lambda list at fifth order, full list otherwise
        errorbar(
            lambdas5,
            means5;
            yerr=stdevs5,
            fmt="o-",
            color=color[5],
            capsize=4,
            markersize=3,
            label="\$N=5\$",
            # label="\$N=5\$ ($solver)",
            zorder=1000,
        )
        # plot(lambdas5, means5, "o-"; color="k", markersize=3, label="\$N=5\$ ($solver)")
        # fill_between(
        #     lambdas5,
        #     (means5 - stdevs5),
        #     (means5 + stdevs5);
        #     color="k",
        #     alpha=0.3,
        # )
    end

    xloc = 1.25
    xlim(minimum(lambdas), maximum(lambdas))
    if rs == 1.0
        xloc = 1.35
        yloc = 0.9775
        ydiv = -0.0125
        # xlim(0.48, 2.0)
        # xlim(0.75, 2.0)
        # ylim(0.87, 0.99)
        xlim(0.75, 4.0)
        ylim(0.87, 0.99)
    elseif rs == 2.0
        xloc = 1.35
        yloc = 0.93
        ydiv = -0.0125
        # xlim(0.48, 2.0)
        # xlim(0.75, 2.0)
        xlim(0.45, 3.05)
        ylim(0.87, 1.0)
    elseif rs == 3.0
        xloc = 0.6
        yloc = 0.9875
        ydiv = -0.01125
        xlim(0.375, 2.0)
        ylim(0.88, 1.0)
    elseif rs == 4.0
        xloc = 0.5
        # yloc = 0.99125
        yloc = 0.99
        ydiv = -0.01125
        ylim(0.91, 1.0)
        # yloc = 1.0275
        # ydiv = -0.0125
        # ylim(0.95, 1.04)
    elseif rs == 5.0
        xloc = 0.6
        yloc = 0.9575
        ydiv = -0.00875
        xlim(0.375, 1.5)
        ylim(0.93, 1.0)
        # ylim(0.375, 1.5)
        # yloc = 1.0275
        # ydiv = -0.0125
        # ylim(0.95, 1.04)
    else
        yloc = 1.0375
        ydiv = -0.02
        ylim(0.85, 1.06)
    end
    if rs == 1.0
        # axvline(1.0; linestyle="--", color="dimgray", label="\$\\lambda^\\star = 1\$")
    elseif rs == 2.0
        # axvline(1.0; linestyle="--", color="dimgray", label="\$\\lambda^\\star = 1\$")
    elseif rs == 3.0
        opt2 = Measurements.value.(mass_ratios_N_vs_lambda[3])[lambdas .== 0.75]
        opt3 = Measurements.value.(mass_ratios_N_vs_lambda[4])[lambdas .== 1.0]
        opt4 = Measurements.value.(mass_ratios_N_vs_lambda[5])[lambdas .== 1.25]
        opt5 = Measurements.value.(mass_ratios_5_vs_lambda5)[lambdas5 .== 1.75]
        scatter(0.75, opt2; s=80, color=color[2], marker="*", zorder=1)
        scatter(1.0, opt3; s=80, color=color[3], marker="*", zorder=101)
        scatter(1.25, opt4; s=80, color=color[4], marker="*", zorder=102)
        scatter(1.75, opt5; s=80, color=color[5], marker="*", zorder=103)
    elseif rs == 4.0
        opt2 = Measurements.value.(mass_ratios_N_vs_lambda[3])[lambdas .== 0.625]
        opt3 = Measurements.value.(mass_ratios_N_vs_lambda[4])[lambdas .== 0.75]
        opt4 = Measurements.value.(mass_ratios_N_vs_lambda[5])[lambdas .== 1.0]
        opt5 = Measurements.value.(mass_ratios_5_vs_lambda5)[lambdas5 .== 1.125]
        scatter(0.625, opt2; s=80, color=color[2], marker="*", zorder=1)
        scatter(0.75, opt3; s=80, color=color[3], marker="*", zorder=101)
        scatter(1.0, opt4; s=80, color=color[4], marker="*", zorder=102)
        scatter(1.125, opt5; s=80, color=color[5], marker="*", zorder=103)
    elseif rs == 5.0
        opt2 = Measurements.value.(mass_ratios_N_vs_lambda[3])[lambdas .== 0.5]
        opt3 = Measurements.value.(mass_ratios_N_vs_lambda[4])[lambdas .== 0.625]
        opt4 = Measurements.value.(mass_ratios_N_vs_lambda[5])[lambdas .== 0.875]
        opt5 = Measurements.value.(mass_ratios_N_vs_lambda[6])[lambdas .== 1.0]
        scatter(0.5, opt2; s=80, color=color[2], marker="*", zorder=100)
        scatter(0.625, opt3; s=80, color=color[3], marker="*", zorder=101)
        scatter(0.875, opt4; s=80, color=color[4], marker="*", zorder=102)
        scatter(1.0, opt5; s=80, color=color[5], marker="*", zorder=103)
    end
    legend(; loc="lower right")
    xlabel("\$\\lambda\$ (Ry)")
    ylabel("\$m^\\star / m\$")
    # xloc = rs
    # xloc = 2.0
    text(
        xloc,
        yloc,
        # "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta), \\delta K = $(dk) k_F\$";
        fontsize=12,
    )
    # text(
    #     xloc,
    #     yloc + ydiv,
    #     # "\$\\delta K = $(dk) k_F\$";
    #     "\$N_{\\mathrm{eval}} = \\mathrm{$(neval)}, \\delta K = $(dk) k_F\$";
    #     fontsize=14,
    # )
    plt.tight_layout()
    savefig(
        "../../results/effective_mass_ratio/effective_mass_ratio_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_$(intn_str)$(solver)_$(ct_string)_deltaK=$(dk)kF_vs_lambda.pdf",
        # "beta_ef=$(param.beta)_neval=$(neval)_$(intn_str)$(solver)_$(ct_string)_deltaK=$(dk)kF_vs_lambda.pdf",
    )
    plt.close("all")
    return
end

main()
