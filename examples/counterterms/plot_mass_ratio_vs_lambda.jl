using CodecZlib
using ElectronLiquid
using ElectronGas
using Interpolations
using JLD2
using Measurements
using Parameters
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

function mass_ratio_screened_fock(param::UEG.ParaMC)
    @unpack rs, kF, mass2 = param
    alpha = (4 / 9π)^(1 / 3)
    c_mu = mass2 / (mass2 + (2 * kF)^2)
    return (1 - (alpha * rs / π) * (1 + ((1 + c_mu) / (1 - c_mu)) * log(sqrt(c_mu))))^(-1)
end

function main()
    # Change to project directory
    # if haskey(ENV, "SOSEM_CEPH")
    #     cd(ENV["SOSEM_CEPH"])
    # elseif haskey(ENV, "SOSEM_HOME")
    #     cd(ENV["SOSEM_HOME"])
    # end

    beta = 40.0
    neval = 1e10
    # rs = 2.0
    # lambdas = [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    # lambdas = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    # lambdas = [1.0, 3.0]
    ### rs = 3 ###
    # rs = 3.0
    # lambdas = [0.75, 0.875, 1.0, 1.125, 1.25, 1.5]
    # lambdas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    # lambdas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    ### rs = 4 ###
    # rs = 4.0
    # lambdas = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5]
    # lambdas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0]
    # lambdas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
    ### rs = 5 ###
    rs = 5.0
    # lambdas = [0.375, 0.5, 0.625, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.5]
    lambdas = [0.8125, 0.875, 0.9375]
    lambdas5 = [0.8125, 0.875, 0.9375]
    # lambdas = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0]
    # lambdas = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0]
    # lambdas = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    solver = :mcmc

    min_order = 0
    max_order = 5

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order_plot = 1
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
    # idk = 1
    # idk = 3  # From lambda = 0.4 stationarity test (δK = 0.015kF)
    # idk = 6  # Try the same grid spacing as rs=1 (δK = 0.03kF)
    # idk = 15
    idk = 7
    # idk = 12
    idk5 = 7

    # Momentum spacing for finite-difference derivative of Sigma (in units of para.kF)
    δK = 0.01
    # δK = 0.005  # spacings n*δK = 0.15–0.3 not relevant for rs = 1.0 => reduce δK by half

    # We estimate the derivative wrt k using grid points kgrid[ikF] and kgrid[ikF + idk]
    idks = eachindex(collect(-6:2:6))
    # idks = collect(-6:2:6)
    # idks = 1:30
    # idks = 1:15
    # idks = -15:15  # TODO: test central difference method

    # kgrid indices & spacings
    dks = δK * collect(-6:2:6)
    # dks = δK * collect(idks)

    dk = dks[idk]

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    local param
    mass_ratios_lambda_vs_N = []
    for lambda in lambdas
        # UEG parameters for MC integration
        loadparam = ParaMC(;
            order=max_order,
            rs=rs,
            beta=beta,
            mass2=lambda,
            isDynamic=false,
            isFock=isFock,
        )
        # Load mass ratio data for each idk
        savename_mass = "data/mass_ratio_from_sigma_kF_gridtest"
        mass_ratios = []
        print("Loading data from $savename_mass...")
        if lambda in lambdas5
            index = idk5
        else
            index = idk
        end
        param, ngrid, kgrid, mass_ratio = jldopen("$savename_mass.jld2", "r") do f
            key = "$(UEG.short(loadparam))_idk=$(index)"
            return f[key]
        end
        push!(mass_ratios_lambda_vs_N, mass_ratio)
        println("done!")
        @assert param.order == max_order
    end
    @assert allequal(length(row) for row in mass_ratios_lambda_vs_N)
    mass_ratios_N_vs_lambda = [
        [mass_ratios_lambda_vs_N[i][j] for i in eachindex(lambdas)] for
        j in eachindex(1:4)
    ]
    mass_ratios_N_vs_lambda5 = [
        [mass_ratios_lambda_vs_N[i][j] for i in eachindex(lambdas5)] for
        j in eachindex(1:5)
    ]

    # mass_ratios_N_vs_lambda = collect(eachrow(reduce(hcat, mass_ratios_lambda_vs_N)))
    # mass_ratios_N_vs_lambda = collect(zip(mass_ratios_lambda_vs_N...))
    println("mass_ratios_lambda_vs_N:\n$mass_ratios_lambda_vs_N")
    println("\nmass_ratios_N_vs_lambda:\n$mass_ratios_N_vs_lambda")
    println("\nmass_ratios_N_vs_lambda5:\n$mass_ratios_N_vs_lambda5")

    # Plot the results for each order ξ vs lambda
    fig, ax = plt.subplots()
    if rs == 3.0
        ax.axvline(1.0; linestyle="--", color="dimgray", label="\$\\lambda^\\star = 1\$")
    end
    for (i, N) in enumerate(0:4)
        N == 0 && continue  # Ignore zeroth order
        # Get means and error bars from the result up to this order
        means = Measurements.value.(mass_ratios_N_vs_lambda[i])
        stdevs = Measurements.uncertainty.(mass_ratios_N_vs_lambda[i])
        # small lambda list at fifth order, full list otherwise
        ax.plot(
            lambdas,
            means,
            "o-";
            color="C$(i-2)",
            markersize=3,
            label="\$N=$N\$ ($solver)",
        )
        ax.fill_between(
            lambdas,
            (means - stdevs),
            (means + stdevs);
            color="C$(i-2)",
            alpha=0.3,
        )
    end
    # Plot 5th order over small lambda list
    # Get means and error bars from the result up to this order
    means5 = Measurements.value.(mass_ratios_N_vs_lambda5)
    stdevs5 = Measurements.uncertainty.(mass_ratios_N_vs_lambda5)
    # small lambda list at fifth order, full list otherwise
    ax.plot(
        lambdas5,
        means5,
        "o-";
        color="k",
        markersize=3,
        label="\$N=5\$ ($solver)",
    )
    ax.fill_between(
        lambdas5,
        (means5 - stdevs5),
        (means5 + stdevs5);
        color="k",
        alpha=0.3,
    )

    xloc = 1.25
    ax.set_xlim(0.125, 3.125)
    if rs == 3.0
        yloc = 1.0375
        ydiv = -0.02
        ax.set_ylim(0.89, 1.06)
    elseif rs == 4.0
        xloc = 0.6
        yloc = 0.94
        ydiv = -0.01125
        ax.set_xlim(0.375, 1.5)
        ax.set_ylim(0.91, 1.0)
        # yloc = 1.0275
        # ydiv = -0.0125
        # ax.set_ylim(0.95, 1.04)
    elseif rs == 5.0
        xloc = 0.6
        yloc = 0.9575
        ydiv = -0.00875
        ax.set_xlim(0.375, 1.5)
        ax.set_ylim(0.93, 1.0)
        # ax.set_ylim(0.375, 1.5)
        # yloc = 1.0275
        # ydiv = -0.0125
        # ax.set_ylim(0.95, 1.04)
    else
        yloc = 1.0375
        ydiv = -0.02
        ax.set_ylim(0.85, 1.06)
    end
    if rs == 4.0
        opt2 = Measurements.value.(mass_ratios_N_vs_lambda[3])[lambdas .== 0.625]
        opt3 = Measurements.value.(mass_ratios_N_vs_lambda[4])[lambdas .== 0.75]
        opt4 = Measurements.value.(mass_ratios_N_vs_lambda[5])[lambdas .== 1.0]
        ax.scatter(0.625, opt2; s=80, color="C1", marker="*", zorder=100)
        ax.scatter(0.75, opt3; s=80, color="C2", marker="*", zorder=101)
        ax.scatter(1.0, opt4; s=80, color="C3", marker="*", zorder=102)
    elseif rs == 5.0
        opt2 = Measurements.value.(mass_ratios_N_vs_lambda[3])[lambdas .== 0.5]
        opt3 = Measurements.value.(mass_ratios_N_vs_lambda[4])[lambdas .== 0.625]
        opt4 = Measurements.value.(mass_ratios_N_vs_lambda[5])[lambdas .== 0.875]
        opt5 = Measurements.value.(mass_ratios_N_vs_lambda[6])[lambdas .== 0.875]
        ax.scatter(0.5, opt2; s=80, color="C1", marker="*", zorder=100)
        ax.scatter(0.625, opt3; s=80, color="C2", marker="*", zorder=101)
        ax.scatter(0.875, opt4; s=80, color="C3", marker="*", zorder=102)
        ax.scatter(0.875, opt5; s=80, color="k", marker="*", zorder=102)
    end
    ax.legend(; loc="lower right")
    ax.set_xlabel("\$\\lambda\$ (Ry)")
    ax.set_ylabel("\$m^\\star / m\$")
    # xloc = rs
    # xloc = 2.0
    ax.text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta)\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + ydiv,
        "\$N_{\\mathrm{eval}} = \\mathrm{$(neval)}, \\delta K = $(dk) k_F\$";
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(
        "../../results/effective_mass_ratio/effective_mass_ratio_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_neval=$(neval)_$(intn_str)$(solver)_$(ct_string)_deltaK=$(dk)kF_vs_lambda.pdf",
    )
    plt.close("all")
    return
end

main()
