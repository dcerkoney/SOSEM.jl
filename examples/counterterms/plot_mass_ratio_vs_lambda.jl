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

function mass_ratio_screened_fock(param::UEG.ParaMC)
    @unpack rs, kF, mass2 = param
    alpha = (4 / 9π)^(1 / 3)
    c_mu = mass2 / (mass2 + (2 * kF)^2)
    return (1 - (alpha * rs / π) * (1 + ((1 + c_mu) / (1 - c_mu)) * log(sqrt(c_mu))))^(-1)
end

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    rs = 2.0
    beta = 40.0
    neval = 1e9
    lambdas = [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    # lambdas = [1.0, 3.0]
    solver = :mcmc

    min_order = 1
    max_order = 4

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order_plot = 1
    max_order_plot = 4

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
    idk = 3  # From lambda = 0.4 stationarity test (δK = 0.015)

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
        param, ngrid, kgrid, mass_ratio = jldopen("$savename_mass.jld2", "r") do f
            key = "$(UEG.short(loadparam))_idk=$(idk)"
            return f[key]
        end
        push!(mass_ratios_lambda_vs_N, mass_ratio)
        println("done!")
        @assert param.order == max_order
    end
    mass_ratios_N_vs_lambda = zip(mass_ratios_lambda_vs_N...)
    println("mass_ratios_lambda_vs_N:\n$mass_ratios_lambda_vs_N")
    println("\nmass_ratios_N_vs_lambda:\n$mass_ratios_N_vs_lambda")

    # Plot the results for each order ξ vs lambda
    fig, ax = plt.subplots()
    # ax.axvline(1.0; linestyle="--", color="dimgray", label="\$\\lambda^\\star = 1\$")
    for (i, N) in enumerate(min_order:max_order_plot)
        # Get means and error bars from the result up to this order
        means = Measurements.value(mass_ratios_N_vs_lambda[i])
        stdevs = Measurements.uncertainty(mass_ratios_N_vs_lambda[i])
        ax.plot(lambdas, means, "o-"; color="C$i", markersize=3, label="\$N=$N\$ ($solver)")
        ax.fill_between(lambdas, (means - stdevs), (means + stdevs); color="C$i", alpha=0.3)
    end
    ax.set_xlim(0.1, 2.0)
    ax.legend(; loc="best")
    ax.set_xlabel("\$\\lambda\$ (Ry)")
    ax.set_ylabel("\$m^\\star / m\$")
    xloc = 1.325
    yloc = -0.54
    ydiv = -0.025
    # ax.text(
    #     xloc,
    #     yloc,
    #     "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta), N_{\\mathrm{eval}} = \\mathrm{$(5e9)},\$";
    #     fontsize=14,
    # )
    # ax.text(
    #     xloc,
    #     yloc + ydiv,
    #     "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
    #     fontsize=12,
    # )
    plt.tight_layout()
    fig.savefig(
        "../../results/effective_mass_ratio/effective_mass_ratio_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_neval=$(neval)_$(intn_str)$(solver)_$(ct_string)_vs_lambda.pdf",
    )
    plt.close("all")
    return
end

main()
