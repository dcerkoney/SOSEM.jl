using CodecZlib
using ElectronLiquid
using ElectronGas
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
    # Change to counterterm directory
    if haskey(ENV, "SOSEM_CEPH")
        cd("$(ENV["SOSEM_CEPH"])/examples/counterterms")
    elseif haskey(ENV, "SOSEM_HOME")
        cd("$(ENV["SOSEM_HOME"])/examples/counterterms")
    end

    rs = 3.0
    # rslist = [2.0]
    # rslist = [1.0, 2.0, 5.0]
    beta = 40.0
    # mass2 = 1.0
    # mass2list = [1.5, 1.75, 2.0]
    mass2list = [1.5]
    # mass2list = [1.75]
    solver = :mcmc
    # solver = :vegasmc

    # Using mass2 from optimization of C⁽¹⁾ⁿˡ(k = 0)
    # TODO: Optimize specifically for ReΣ_N(para.kF, ik0) convergence vs N
    # c1nl_mass2_optima = Dict{Float64,Float64}(1.0 => 1.0, 2.0 => 0.4, 5.0 => 0.1375)

    # Total number of MCMC evaluations
    neval = 1e11

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
    δK = 0.005  # spacings n*δK = 0.15–0.3 not relevant for rs = 1.0 => reduce δK by half
    # δK = 0.01

    # We estimate the derivative wrt k using grid points kgrid[ikF] and kgrid[ikF + idk]
    # idks = 1:10
    # idks = -15:15
    idks = 1:15
    dks = δK * collect(idks)

    for mass2 in mass2list
    # for rs in rslist
        # mass2 = c1nl_mass2_optima[rs]
        # UEG parameters for MC integration
        loadparam = ParaMC(;
            order=max_order,
            rs=rs,
            beta=beta,
            mass2=mass2,
            isDynamic=false,
            isFock=isFock,
        )

        # Load mass ratio data for each idk
        # savename_mass = "data/mass_ratio_from_sigma"
        # savename_zinv = "data/inverse_zfactor"
        # savename_zapprox = "data/zfactor_approx"
        savename_mass = "data/mass_ratio_from_sigma_kF_gridtest"
        savename_zinv = "data/inverse_zfactor"
        savename_zapprox = "data/zfactor_approx"
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

        # Collect first order results
        r1s = [r[2] for r in mass_ratios]
        r1_means, r1_stdevs = Measurements.value.(r1s), Measurements.uncertainty.(r1s)

        # Collect max order results
        rmaxs = [r[end] for r in mass_ratios]
        rmax_means, rmax_stdevs =
            Measurements.value.(rmaxs), Measurements.uncertainty.(rmaxs)

        # Check stationarity at each order
        println(mass_ratios[1])
        println(mass_ratios[2])
        max_idks = Int[]
        for N in 0:max_order
            # Keep iterating until we find a measurement outside current valid window
            max_idk = length(idks)
            valid_window = [-Inf, Inf]
            println("\nOrder $N:")
            for (idk, mass_ratio) in enumerate(mass_ratios)
                # println(mass_ratio)
                println(valid_window)
                m = mass_ratio[N + 1]
                m_min, m_max = m.val - m.err, m.val + m.err
                # Narrow the valid window
                curr_min, curr_max = valid_window
                valid_window = [max(curr_min, m_min), min(curr_max, m_max)]
                # If the window is empty, we found the max idk (at the previous step)
                if valid_window[1] > valid_window[2]
                    max_idk = idk - 1
                    break
                end
            end
            println("max_idk = $max_idk")
            push!(max_idks, max_idk)
        end
        max_stationary_idk = minimum(max_idks)
        max_stationary_idk_1 = max_idks[2]
        max_stationary_idk_max_order = max_idks[end]

        println("\nmax_idks = $max_idks")
        println("Total maximum stationary index: $max_stationary_idk")

        # Get exact zero-temperature first order (HF) result
        fock_mass_ratio_ex = mass_ratio_screened_fock(param)

        # Get maximum T=0 compatible index
        max_zt_compatible_idk = 1
        for (idk, mass_ratio) in enumerate(mass_ratios)
            top = mass_ratio[2].val + mass_ratio[2].err
            bottom = mass_ratio[2].val - mass_ratio[2].err
            # This k-index agrees with the zero-temperature value to within 1σ
            if (bottom < fock_mass_ratio_ex < top)
                max_zt_compatible_idk = idk
            end
        end
        println("Maximum T=0 compatible index: $max_zt_compatible_idk")

        # Plot first order vs δK and compare to exact zero temperature result
        fig, ax = plt.subplots()
        ax.axvline(
            dks[max_zt_compatible_idk];
            label="Max \$T=0\$ compatible \$\\delta K = $(dks[max_zt_compatible_idk]) k_F\$",
            color="k",
            linestyle="--",
        )
        ax.axvline(
            dks[max_stationary_idk];
            label="Max stationary \$\\delta K = $(dks[max_stationary_idk]) k_F\$",
            color="gray",
            linestyle="--",
        )
        ax.axhline(fock_mass_ratio_ex; label="Exact (\$T = 0\$)", color="r")
        ax.errorbar(dks, r1_means, r1_stdevs; label="MCMC", capsize=4)
        ax.set_xlabel("\$\\delta K / k_F\$")
        ax.set_ylabel("\$\\left(m^\\star / m\\right)_1\$")
        # xloc = 0.095
        # yloc = 0.916
        # ydiv = -0.001
        ### rs = 2, lambda = 1.5 ###
        # xloc = 0.03
        # yloc = 0.97
        # ydiv = -0.003
        ### rs = 3, lambda = 1.5 ###
        xloc = 0.04
        yloc = 0.97675
        ydiv = -0.00025
        ### lambda = 1.75 ###
        # xloc = 0.03
        # yloc = 0.964
        # ydiv = -0.0005
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
        ax.legend(; loc="best")
        fig.tight_layout()
        fig.savefig(
            "../../results/effective_mass_ratio/first_order_mass_ratio_vs_dK_" *
            "rs=$(param.rs)_beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
            "neval=$(neval)_$(solver)_$(ct_string)_kF_gridtest.pdf",
            # "$(solver)_$(ct_string).pdf",
        )

        # Plot max order vs δK
        fig, ax = plt.subplots()
        ax.axvline(
            dks[max_zt_compatible_idk];
            label="Max \$T=0\$ compatible \$\\delta K = $(dks[max_zt_compatible_idk]) k_F\$",
            color="k",
            linestyle="--",
        )
        ax.axvline(
            dks[max_stationary_idk];
            label="Max stationary \$\\delta K = $(dks[max_stationary_idk]) k_F\$",
            color="gray",
            linestyle="--",
        )
        ax.errorbar(dks, rmax_means, rmax_stdevs; label="MCMC", capsize=4)
        ax.set_xlabel("\$\\delta K / k_F\$")
        ax.set_ylabel("\$\\left(m^\\star / m\\right)_$(max_order)\$")
        # xloc = 0.125
        # yloc = 0.931
        ### rs = 2, lambda = 1.5 ###
        # xloc = 0.0325
        # yloc = 0.93
        # ydiv = -0.01
        ### rs = 3, lambda = 1.5 ###
        xloc = 0.04
        yloc = 0.9605
        ydiv = -0.001
        ### lambda = 1.75 ###
        # xloc = 0.03
        # yloc = 0.960
        # ydiv = -0.0025
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
        ax.legend(; loc="lower right")
        # ax.legend(; loc="best")
        fig.tight_layout()
        fig.savefig(
            "../../results/effective_mass_ratio/mass_ratio_N=$(max_order)_vs_dK_" *
            "rs=$(param.rs)_beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
            "neval=$(neval)_$(solver)_$(ct_string)_kF_gridtest.pdf",
            # "$(solver)_$(ct_string).pdf",
        )

        fig, ax = plt.subplots()
        orders = 0:max_order
        # Using maximum stationary δK scheme
        scheme_max_idks = [max_stationary_idk]
        scheme_strs = ["Max stationary"]
        # scheme_max_idks = [max_zt_compatible_idk, max_stationary_idk]
        # scheme_strs = ["Max \$T=0\$ compatible", "Max stationary"]
        for (idk, scheme_str) in zip(scheme_max_idks, scheme_strs)
            means, stdevs = Measurements.value.(mass_ratios[idk]),
            Measurements.uncertainty.(mass_ratios[idk])
            println("\nEffective mass ratios at δK = $(dks[idk]):")
            for o in eachindex(orders)
                println(" • (m⋆/m)_$(orders[o]) = $(mass_ratios[idk][o])")
            end
            ax.errorbar(
                orders,
                means,
                stdevs;
                capsize=4,
                zorder=10 * idk,
                label="$scheme_str \$\\delta K = $(dks[idk]) k_F\$",
            )
        end
        # xloc = 1.2
        # yloc = 0.8
        # ydiv = -0.03
        # yloc = 0.9875
        # ydiv = -0.01
        # yloc = 0.9675
        # yloc = 0.985
        # xloc = 0.2
        # yloc = 0.97
        # ydiv = -0.0125
        # xloc = 1.5
        # yloc = 0.99
        # ydiv = -0.0075
        xloc = 1.5
        yloc = 0.995
        ydiv = -0.005
        ax.text(
            xloc,
            yloc,
            "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta), \\delta K = $(dks[max_stationary_idk]) k_F,\$";
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
            "\$m^\\star / m \\approx $(round(mass_ratios[max_stationary_idk][end].val; digits=4)) \\pm $(round(mass_ratios[max_stationary_idk][end].err; digits=4))\$";
            fontsize=14,
        )
        ax.set_xlabel("\$N\$")
        ax.set_ylabel("\$m^\\star / m\$")
        ax.set_xticks(orders)
        ax.set_xticklabels(orders)
        # ax.legend(; loc="best")
        # ax.legend(; loc="lower left")
        fig.tight_layout()
        fig.savefig(
            "../../results/effective_mass_ratio/effective_mass_ratio_rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string)_kF.pdf",
            # "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string)_kF_gridtest.pdf",
            # "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string).pdf",
        )

        fig, ax = plt.subplots()
        orders = 0:max_order
        means, stdevs = Measurements.value.(zinv), Measurements.uncertainty.(zinv)
        ax.errorbar(orders, means, stdevs; capsize=4, zorder=10)
        xloc = 0.25
        yloc = 1.065
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
            "../../results/effective_mass_ratio/inverse_zfactor_rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string)_kF.pdf",
            # "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string)_kF_gridtest.pdf",
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
            "../../results/effective_mass_ratio/zfactor_approximation_rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string)_kF.pdf",
            # "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string)_kF_gridtest.pdf",
            # "lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string).pdf",
        )
        plt.close("all")
    end
    return
end

main()
