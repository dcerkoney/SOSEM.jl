using CodecZlib
using ElectronLiquid
using ElectronGas
using JLD2
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

function main()
    # Change to counterterm directory
    if haskey(ENV, "SOSEM_CEPH")
        cd("$(ENV["SOSEM_CEPH"])/examples/counterterms")
    elseif haskey(ENV, "SOSEM_HOME")
        cd("$(ENV["SOSEM_HOME"])/examples/counterterms")
    end

    rs = 1.0
    beta = 40.0
    mass2 = 1.0
    solver = :vegasmc

    # Total number of MCMC evaluations
    neval = 5e10

    # Physical params matching data for SOSEM observables
    min_order = 0  # C^{(1)}_{N≤5} includes CTs up to 4th order
    max_order = 4

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

    # Momentum spacing for finite-difference derivative of Sigma (in units of kF)
    δK = 0.01

    # We estimate the derivative wrt k using grid points kgrid[ikF] and kgrid[ikF + idk]
    idks = 1:10
    dks = δK * collect(idks)

    # UEG parameters for MC integration
    loadparam = ParaMC(;
        order=max_order,
        rs=rs,
        beta=beta,
        mass2=mass2,
        isDynamic=false,
        isFock=isFock,
    )

    signstrs = ["++", "--", "+-", "-+"]
    for signstr in signstrs
        # Load mass ratio data for each idk
        # savename_mass = "mass_ratio_from_sigma"
        # savename_zinv = "inverse_zfactor"
        # savename_zapprox = "zfactor_approx"
        savename_mass = "mass_ratio_from_sigma_gridtest_$(signstr)"
        savename_zinv = "inverse_zfactor_gridtest_$(signstr)"
        savename_zapprox = "zfactor_approx_gridtest_$(signstr)"
        local param
        mass_ratios = []
        print("Loading data from $savename_mass...")
        for idk in idks
            # param, ngrid, kgrid, mass_ratio = jldopen("$savename_mass.jld2", "a+") do f
            param, ngrid, kgrid, mass_ratio = jldopen("$savename_mass.jld2", "a+") do f
                key = "$(UEG.short(loadparam))_idk=$(idk)"
                return f[key]
            end
            push!(mass_ratios, mass_ratio)
        end
        print("Loading data from $savename_zinv...")
        local zinv
        for idk in idks
            # param, ngrid, kgrid, zinv = jldopen("$savename_zinv.jld2", "a+") do f
            param, ngrid, kgrid, zinv = jldopen("$savename_zinv.jld2", "a+") do f
                key = "$(UEG.short(loadparam))"
                return f[key]
            end
        end
        print("Loading data from $savename_zapprox...")
        local zapprox
        for idk in idks
            # param, ngrid, kgrid, zapprox = jldopen("$savename_zapprox.jld2", "a+") do f
            param, ngrid, kgrid, zapprox = jldopen("$savename_zapprox.jld2", "a+") do f
                key = "$(UEG.short(loadparam))"
                return f[key]
            end
        end
        println("done!")
        @assert param.order == max_order

        # Use LaTex fonts for plots
        plt.rc("text"; usetex=true)
        plt.rc("font"; family="serif")

        fig, ax = plt.subplots()
        orders = 0:max_order
        for idk in idks
            means, stdevs = Measurements.value.(mass_ratios[idk]),
            Measurements.uncertainty.(mass_ratios[idk])
            ax.errorbar(
                orders,
                means,
                stdevs;
                capsize=4,
                zorder=10 * idk,
                label="\$\\delta K = $(dks[idk]) k_F\$",
            )
        end
        xloc = 1.2
        # yloc = 0.8
        # ydiv = -0.03
        # yloc = 0.9875
        # ydiv = -0.01
        yloc = 0.9675
        ydiv = -0.0125
        ax.text(
            xloc,
            yloc,
            "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
            fontsize=14,
        )
        ax.text(
            xloc,
            yloc + ydiv,
            "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
            fontsize=14,
        )
        ax.set_xlabel("\$N\$")
        ax.set_ylabel("\$m^\\star / m\$")
        ax.set_xticks(orders)
        ax.set_xticklabels(orders)
        # ax.set_ylim(0.8, nothing)
        # ax.legend(; loc="lower right", ncol=2)
        ax.legend(; loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(
            "../../results/effective_mass_ratio/sign_tests/effective_mass_ratio_rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string)_gridtest_$(signstr).pdf",
            # "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string).pdf",
        )

        fig, ax = plt.subplots()
        orders = 0:max_order
        means, stdevs = Measurements.value.(zinv), Measurements.uncertainty.(zinv)
        ax.errorbar(orders, means, stdevs; capsize=4, zorder=10)
        xloc = 2.125
        yloc = 1.03
        ydiv = -0.01
        ax.text(
            xloc,
            yloc,
            "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
            fontsize=14,
        )
        ax.text(
            xloc,
            yloc + ydiv,
            "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
            fontsize=14,
        )
        ax.set_xlabel("\$N\$")
        ax.set_ylabel("\$Z^{-1}\$")
        ax.set_xticks(orders)
        ax.set_xticklabels(orders)
        fig.tight_layout()
        fig.savefig(
            "../../results/effective_mass_ratio/sign_tests/inverse_zfactor_rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string)_gridtest_$(signstr).pdf",
            # "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string).pdf",
        )

        fig, ax = plt.subplots()
        orders = 0:max_order
        means, stdevs = Measurements.value.(zapprox), Measurements.uncertainty.(zapprox)
        ax.errorbar(orders, means, stdevs; capsize=4, zorder=10)
        xloc = 2.125
        yloc = 0.98
        ydiv = -0.01
        ax.text(
            xloc,
            yloc,
            "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
            fontsize=14,
        )
        ax.text(
            xloc,
            yloc + ydiv,
            "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
            fontsize=14,
        )
        ax.set_xlabel("\$N\$")
        ax.set_ylabel("\$Z_N \\approx 1 / (Z^{-1})_N\$")
        ax.set_xticks(orders)
        ax.set_xticklabels(orders)
        fig.tight_layout()
        fig.savefig(
            "../../results/effective_mass_ratio/sign_tests/zfactor_approximation_rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string)_gridtest_$(signstr).pdf",
            # "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string).pdf",
        )
        plt.close("all")
    end
    return
end

main()
