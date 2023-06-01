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

    plot_rpa = false

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
    return

    # Plot the results for each order ξ vs lambda and compare to RPA(+FL)
    fig, ax = plt.subplots()
    ax.axvline(1.0; linestyle="--", color="dimgray", label="\$\\lambda^\\star = 1\$")
    if min_order_plot == 2
        if plot_rpa
            ax.plot(
                lambdas,
                c1nl_rpa_means,
                "o--";
                color="k",
                markersize=3,
                label="RPA (vegas)",
            )
            ax.fill_between(
                lambdas,
                (c1nl_rpa_means - c1nl_rpa_stdevs),
                (c1nl_rpa_means + c1nl_rpa_stdevs);
                color="k",
                alpha=0.3,
            )
            ax.plot(
                lambdas,
                c1nl_rpa_fl_means,
                "o-";
                color="k",
                markersize=3,
                label="RPA\$+\$FL (vegas)",
            )
            ax.fill_between(
                lambdas,
                (c1nl_rpa_fl_means - c1nl_rpa_fl_stdevs),
                (c1nl_rpa_fl_means + c1nl_rpa_fl_stdevs);
                color="r",
                alpha=0.3,
            )
        end
        # ax.plot(
        #     lambdas,
        #     c1nl_los,
        #     "o-";
        #     color="C0",
        #     markersize=3,
        #     label="\$N=2\$ (quad, \$T = 0\$)",
        # )
        ax.plot(
            lambdas,
            -0.5 * one.(lambdas),
            "-";
            color="C0",
            markersize=3,
            label="\$N=2\$ (exact, \$T = 0\$)",
        )
    end
    for (i, N) in enumerate(min_order:max_order_plot)
        c1nl_N_means = repeat([Inf], length(lambdas))
        c1nl_N_stdevs = repeat([Inf], length(lambdas))
        for (j, filename) in enumerate(filenames)
            if N == 5 && j != 4
                # Currently no data for N = 5, lambda != 2
                continue
            end
            if j == 4
                println("\nN = $N, lambda = $(lambdas[j]):")
            end
            f = jldopen("$filename.jld2", "r")
            if j == 4 && N == 5
                # Load N = 5 data for lambda = 2 (currently, mixed nevals and multi-k)
                k1 = f["c1b0/N=5/neval=2.0e10/kgrid"][[1]]
                k2 = f["c1c/N=5/neval=1.0e9/kgrid"][[1]]
                k3 = f["c1d/N=5/neval=2.0e10/kgrid"][[1]]
                @assert k1 == k2 == k3 == [0.0]
                r1 = f["c1b0/N=5/neval=2.0e10/meas"][[1]]
                r2 = f["c1c/N=5/neval=1.0e9/meas"][[1]]
                r3 = f["c1d/N=5/neval=2.0e10/meas"][[1]]
            else
                # Load the data for each observable
                this_kgrid = f["c1d/N=$(N)_unif/neval=$neval/kgrid"]
                @assert this_kgrid == [0.0]
                r1 = f["c1b0/N=$(N)_unif/neval=$neval/meas"]
                r2 = f["c1c/N=$(N)_unif/neval=$neval/meas"]
                r3 = f["c1d/N=$(N)_unif/neval=$neval/meas"]
            end
            c1nl_N_total = r1 + r2 + r3
            # The c1b observable has no data for N = 2
            if N > 2
                if j == 4 && N == 5
                    # Load N = 5 data for lambda = 2 (currently, mixed nevals and multi-k)
                    k4 = f["c1b/N=5/neval=5.0e9/kgrid"][[1]]
                    @assert k4 == [0.0]
                    r4 = f["c1b/N=5/neval=5.0e9/meas"][[1]]
                    c1nl_N_total += r4
                else
                    r4 = f["c1b/N=$(N)_unif/neval=$neval/meas"]
                    c1nl_N_total += r4
                end
                if j == 4
                    println(
                        "c1b0_unif = $r1\nc1b_unif = $r4\nc1c_unif = $r2\nc1d_unif = $r3",
                    )
                end
            else
                if j == 4
                    println("c1b0_unif = $r1\nc1c_unif = $r2\nc1d_unif = $r3")
                end
            end
            if j == 4
                println("c1nl_N_total = $c1nl_N_total")
            end
            close(f)  # close file
            @assert length(c1nl_N_total) == 1

            # Get means and error bars from the result up to this order
            c1nl_N_means[j] = Measurements.value(c1nl_N_total[1])
            c1nl_N_stdevs[j] = Measurements.uncertainty(c1nl_N_total[1])
        end
        # TODO: more points and consistent neval
        label =
            N == 5 ? "\$N=$N, N_{\\mathrm{eval}}=\\mathrm{5.0e9}\$ ($solver)" :
            "\$N=$N\$ ($solver)"
        ax.plot(lambdas, c1nl_N_means, "o-"; color="C$i", markersize=3, label=label)
        ax.fill_between(
            lambdas,
            (c1nl_N_means - c1nl_N_stdevs),
            (c1nl_N_means + c1nl_N_stdevs);
            color="C$i",
            alpha=0.3,
        )
    end
    ax.set_xlim(0.5, 3.0)
    ax.set_ylim(; bottom=-0.75)
    ax.legend(; loc="best")
    ax.set_xlabel("\$\\lambda\$ (Ry)")
    ax.set_ylabel(
        "\$C^{(1)nl}(k=0,\\, \\lambda) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    xloc = 1.325
    yloc = -0.54
    ydiv = -0.025
    # xloc = 1.7
    # yloc = -0.5
    # ydiv = -0.05
    ax.text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta), N_{\\mathrm{eval}} = \\mathrm{$(5e9)},\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + ydiv,
        "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=12,
    )
    # plt.title("")
    plt.tight_layout()
    fig.savefig(
        "results/c1nl/c1nl_k=0_rs=$(rs)_" *
        "beta_ef=$(beta)_neval=$(5e9)_" *
        "$(intn_str)$(solver)_vs_lambda.pdf",
    )
    plt.close("all")
    return
end

main()
