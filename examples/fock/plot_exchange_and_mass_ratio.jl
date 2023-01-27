using ElectronGas
using ElectronLiquid
using Interpolations
using JLD2
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

"""
Exact expression for the Fock self-energy
in terms of the dimensionless Lindhard function.
"""
function fock_self_energy_exact(k, p::ParaMC)
    # The (dimensionful) value at k = 0 is minus the Thomas-Fermi energy
    eTF = p.qTF^2 / (2 * p.me)
    return -eTF * UEG_MC.lindhard(k / p.kF)
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
    mass2 = 1.0
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 1e8

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order = 2
    max_order = 4
    min_order_plot = 1
    max_order_plot = 4

    # Save total results
    save = true

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""

    # Full renormalization
    ct_string = "with_ct_mu_lambda"

    # UEG parameters for MC integration
    loadparam = ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    savename =
        "results/data/sigma_x_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval)_$(solver)"
    orders, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(loadparam))"
        return f[key]
    end

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])
    println(merged_data)

    if min_order_plot == 1
        # Set bare result manually using exact Fock self-energy / eTF
        sigma_fock_over_eTF_exact = -UEG_MC.lindhard.(k_kf_grid)
        merged_data[(1, 0)] = measurement.(sigma_fock_over_eTF_exact, 0.0)  # treat quadrature data as numerically exact
    end

    # Reexpand merged data in powers of μ
    z, μ = UEG_MC.load_z_mu(param)
    δz, δμ = CounterTerm.sigmaCT(max_order, μ, z; verbose=1)
    println("Computed δμ: ", δμ)
    sigma_x = UEG_MC.chemicalpotential_renormalization(merged_data, δμ; max_order=max_order)
    # Test manual renormalization with exact lowest-order chemical potential
    δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
    # Σₓ⁽²⁾ = Σₓ_{2,0} + δμ₁ Σₓ_{1,1}
    sigma_x_2_manual = merged_data[(2, 0)] + δμ1_exact * merged_data[(1, 1)]
    stdscores = stdscore.(sigma_x[2], sigma_x_2_manual)
    worst_score = argmax(abs, stdscores)
    println("Exact δμ₁: ", δμ1_exact)
    println("Computed δμ₁: ", δμ[1])
    println(
        "Worst standard score for total result to 3rd " *
        "order (auto vs exact+manual): $worst_score",
    )
    # Aggregate the full results for Σₓ up to order N
    sigma_x_over_eTF_total = UEG_MC.aggregate_orders(sigma_x)

    println(UEG.paraid(param))
    println(partitions)
    println(res)

    if save
        savename =
            "results/data/rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_$(intn_str)$(solver)_$(ct_string)"
        f = jldopen("$savename.jld2", "a+")
        # NOTE: no bare result for c1b observable (accounted for in c1b0)
        for N in min_order_plot:max_order
            # Skip exact Fock (N = 1) result
            N == 1 && continue
            if haskey(f, "sigma_x") &&
               haskey(f["sigma_x"], "N=$N") &&
               haskey(f["sigma_x/N=$N"], "neval=$(neval)")
                @warn("replacing existing data for N=$N, neval=$(neval)")
                delete!(f["sigma_x/N=$N"], "neval=$(neval)")
            end
            f["sigma_x/N=$N/neval=$neval/meas"] = sigma_x_over_eTF_total[N]
            f["sigma_x/N=$N/neval=$neval/param"] = param
            f["sigma_x/N=$N/neval=$neval/kgrid"] = kgrid
        end
    end

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # Plot for each aggregate order

    # Σₓ(k) / eTF (dimensionless moment)
    fig1, ax1 = plt.subplots()
    # Compare result to exact non-dimensionalized Fock self-energy (-F(k / kF))
    ax1.plot(k_kf_grid, -UEG_MC.lindhard.(k_kf_grid), "k"; label="\$N=1\$ (exact, \$T=0\$)")
    for (i, N) in enumerate(min_order:max_order_plot)
        N == 1 && continue
        # Get means and error bars from the result up to this order
        means = Measurements.value.(sigma_x_over_eTF_total[N])
        stdevs = Measurements.uncertainty.(sigma_x_over_eTF_total[N])
        # Data gets noisy above 3rd loop order
        # marker = N > 2 ? "o-" : "-"
        marker = "o-"
        ax1.plot(
            k_kf_grid,
            means,
            marker;
            markersize=2,
            color="C$(i-1)",
            label="\$N=$N\$ ($solver)",
        )
        ax1.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            color="C$(i-1)",
            alpha=0.4,
        )
    end
    ax1.legend(; loc="lower right")
    ax1.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    ax1.set_xlabel("\$k / k_F\$")
    ax1.set_ylabel("\$\\Sigma_{x}(k) \\,/\\, \\epsilon_{\\mathrm{TF}}\$")
    xloc = 1.75
    yloc = -0.5
    ydiv = -0.095
    ax1.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax1.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    fig1.tight_layout()
    fig1.savefig(
        "results/fock/sigma_x_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver).pdf",
    )

    # Thomas-Fermi energy
    eTF = param.qTF^2 / (2 * param.me)

    # Bare dispersion in units of the Thomas-Fermi energy (for effective mass related plots)
    Ek = kgrid .^ 2 / (2 * param.me)
    Ek_over_eTF = Ek / eTF

    sigma_fock_exact = [fock_self_energy_exact(k, param) for k in kgrid]

    # Moment quasiparticle energy
    fig2, ax2 = plt.subplots()
    ax2.axhline(0; linestyle="--", color="gray")
    ax2.plot(k_kf_grid, Ek_over_eTF, "k"; label="\$\\epsilon_k / \\epsilon_{\\mathrm{TF}} = (k / q_{\\mathrm{TF}})^2\$")
    ax2.plot(k_kf_grid, Ek_over_eTF .- UEG_MC.lindhard.(k_kf_grid), "C0"; label="\$N=1\$ (exact, \$T=0\$)")
    # ax2.plot(k_kf_grid, Ek_over_eTF, "k"; label="\$\\epsilon_k / \\epsilon_{\\mathrm{TF}} = (k / q_{\\mathrm{TF}})^2\$")
    # ax2.plot(k_kf_grid, Ek_over_eTF .- UEG_MC.lindhard.(k_kf_grid), "C0"; label="\$N=1\$ (exact, \$T=0\$)")
    for (i, N) in enumerate(min_order:max_order_plot)
        N == 1 && continue
        # Eqp = ϵ(k) + Σₓ(k)
        Eqp = Ek_over_eTF .+ sigma_x_over_eTF_total[N]
        # Get means and error bars
        means_qp = Measurements.value.(Eqp)
        stdevs_qp = Measurements.uncertainty.(Eqp)

        # Data gets noisy above 3rd loop order
        # marker = N > 2 ? "o-" : "-"
        marker = "o-"
        ax2.plot(
            k_kf_grid,
            means_qp,
            marker;
            markersize=2,
            color="C$i",
            label="\$N=$N\$ ($solver)",
        )
        ax2.fill_between(
            k_kf_grid,
            means_qp - stdevs_qp,
            means_qp + stdevs_qp;
            color="C$i",
            alpha=0.4,
        )
    end
    ax2.legend(; loc="lower right")
    ax2.set_xlim(minimum(k_kf_grid), 1.5)
    ax2.set_ylim(-1.5, 3.0)
    ax2.set_xlabel("\$k / k_F\$")
    ax2.set_ylabel(
        "\$\\epsilon_{\\mathrm{momt.}}(k) \\,/\\, \\epsilon_{\\mathrm{TF}} =  \\left(\\epsilon_{k} + \\Sigma_{x}(k)\\right) \\,/\\, \\epsilon_{\\mathrm{TF}} \$",
    )
    xloc = 0.1
    yloc = 2.5
    ydiv = -0.5
    ax2.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax2.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    ax2.text(
        xloc,
        yloc + 2 * ydiv,
        "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=12,
    )
    fig2.tight_layout()
    fig2.savefig(
        "results/fock/moment_qp_energy_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver).pdf",
    )

    # Mass ratio
    fig3, ax3 = plt.subplots()
    ax3.axhline(1; linestyle="--", color="gray")
    ax3.plot(
        k_kf_grid,
        1.0 .+ sigma_fock_exact ./ Ek,
        "k";
        label="\$N=1\$ (exact, \$T=0\$)",
    ) 
    for (i, N) in enumerate(min_order:max_order_plot)
        N == 1 && continue
        # Σₓ(k) / ϵ(k)
        sigma_x_over_Ek = eTF * sigma_x_over_eTF_total[N] ./ Ek
        # m_e / m_{momt}(k) ≈ 1 + Σₓ(k) / ϵ(k)
        mass_ratio = 1.0 .+ sigma_x_over_Ek
        # Get means and error bars from the mass ratio up to this order
        means_mass_ratio = Measurements.value.(mass_ratio)
        stdevs_mass_ratio = Measurements.uncertainty.(mass_ratio)

        ikF = findall(x->x==1.0, k_kf_grid)
        println("eff mass: ", means_mass_ratio[ikF])
        # Data gets noisy above 3rd loop order
        # marker = N > 2 ? "o-" : "-"
        marker = "o-"
        ax3.plot(
            k_kf_grid,
            means_mass_ratio,
            marker;
            markersize=2,
            color="C$(i-1)",
            label="\$N=$N\$ ($solver)",
        )
        ax3.fill_between(
            k_kf_grid,
            means_mass_ratio - stdevs_mass_ratio,
            means_mass_ratio + stdevs_mass_ratio;
            color="C$(i-1)",
            alpha=0.4,
        )
    end
    ax3.legend(; loc="lower right")
    ax3.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    ax3.set_ylim(0, 1.1)
    ax3.set_xlabel("\$k / k_F\$")
    ax3.set_ylabel("\$\\epsilon_{\\mathrm{momt.}}(k) / \\epsilon_k\$")
    # ax3.set_ylabel("\$m_e \\,/\\, m_{\\mathrm{momt.}}\$")
    xloc = 1.75
    yloc = 0.75
    ydiv = -0.095
    ax3.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax3.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax3.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    fig3.tight_layout()
    fig3.savefig(
        "results/fock/moment_energy_ratio_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver).pdf",
    )

    # # Mass ratio
    # fig4, ax4 = plt.subplots()
    # ax4.axhline(1; linestyle="--", color="gray")

    # fock_mass_ratio_exact(y) = (param.e0^2 * param.me / (2pi * param.kF)) * ((1 + y^2) * log(abs((1+y)/(1-y))) / y - 2) / y^2

    # ax4.plot(
    #     LinRange(0.0, 3.0, 500),
    #     fock_mass_ratio_exact(LinRange(0.0, 3.0, 500)),
    #     "k";
    #     label="\$N=1\$ (exact, \$T=0\$)",
    # ) 
    # for (i, N) in enumerate(min_order:max_order_plot)
    #     N == 1 && continue
    #     # Σₓ(k) / ϵ(k)
    #     sigma_x_over_Ek = eTF * sigma_x_over_eTF_total[N] ./ Ek
        
    #     # m_e / m_{momt}(k) = 1 + ∂Σₓ(k)/∂ϵ(k)


    #     mass_ratio = 1.0 .+ sigma_x_over_Ek


    #     # Get means and error bars from the mass ratio up to this order
    #     means_mass_ratio = Measurements.value.(mass_ratio)
    #     stdevs_mass_ratio = Measurements.uncertainty.(mass_ratio)

    #     ikF = findall(x->x==1.0, k_kf_grid)
    #     println("eff mass: ", means_mass_ratio[ikF])

    #     # Data gets noisy above 3rd loop order
    #     # marker = N > 2 ? "o-" : "-"
    #     marker = "o-"
    #     ax4.plot(
    #         k_kf_grid,
    #         means_mass_ratio,
    #         marker;
    #         markersize=2,
    #         color="C$(i-1)",
    #         label="\$N=$N\$ ($solver)",
    #     )
    #     ax4.fill_between(
    #         k_kf_grid,
    #         means_mass_ratio - stdevs_mass_ratio,
    #         means_mass_ratio + stdevs_mass_ratio;
    #         color="C$(i-1)",
    #         alpha=0.4,
    #     )
    # end
    # ax4.legend(; loc="lower right")
    # ax4.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    # ax4.set_ylim(0, 1.1)
    # ax4.set_xlabel("\$k / k_F\$")
    # ax4.set_ylabel("\$\\epsilon_{\\mathrm{momt.}}(k) / \\epsilon_k\$")
    # # ax4.set_ylabel("\$m_e \\,/\\, m_{\\mathrm{momt.}}\$")
    # xloc = 1.75
    # yloc = 0.7
    # ydiv = -0.095
    # ax4.text(
    #     xloc,
    #     yloc,
    #     "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
    #     fontsize=14,
    # )
    # ax4.text(
    #     xloc,
    #     yloc,
    #     "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
    #     fontsize=14,
    # )
    # ax4.text(
    #     xloc,
    #     yloc + ydiv,
    #     "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
    #     # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
    #     fontsize=14,
    # )
    # fig4.tight_layout()
    # fig4.savefig(
    #     "results/fock/moment_mass_ratio_N=$(max_order_plot)_rs=$(param.rs)_" *
    #     "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver).pdf",
    # )

    plt.close("all")
    return
end

main()
