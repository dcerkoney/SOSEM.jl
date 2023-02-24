using CodecZlib
using ElectronGas
using ElectronLiquid
using Interpolations
using JLD2
using Lehmann
using MCIntegration
using Measurements
using Parameters
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport matplotlib.pyplot as plt
@pyimport mpl_toolkits.axes_grid1.inset_locator as il

"""
Computes the exact value for the Yukawa-screened Fock self-energy Σ^λ_F(k).
"""
function sigma_lambda_fock_exact(k::Float64, param::UEG.ParaMC)
    # Dimensionless wavenumber at the Fermi surface (x = k / kF)
    x = k / param.kF
    # Dimensionless Yukawa mass squared (lambda = λ / kF²)
    lambda = param.mass2 / param.kF^2
    # Dimensionless screened Lindhard function
    F_x_lambda = UEG_MC.screened_lindhard(x; lambda=lambda)
    # δμ₁ cancels the real part of the Fock self-energy
    # at the Fermi surface for a Yukawa-screened UEG.
    return -(param.e0^2 * param.kF / (2 * pi^2 * param.ϵ0)) * F_x_lambda
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

    # Number of evals
    neval = 1e7

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = false

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # # UEG parameters
    # max_order = 3
    # loadparam = ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    # # Load the raw data
    # savename =
    #     "results/data/occupation_n=$(max_order)_rs=$(rs)_" *
    #     "beta_ef=$(beta)_lambda=$(mass2)_neval=$(neval)_$(solver)"
    # orders, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
    #     key = "$(UEG.short(loadparam))"
    #     return f[key]
    # end

    res_list = Result[]
    local param, kgrid
    loadparam =
        ParaMC(; order=2, rs=rs, beta=beta, mass2=mass2, isDynamic=false, isFock=false)
    for i in 1:3
        savename = "results/data/occupation_N=2_fock_term_$(i)_neval=$(neval)"
        param, kgrid, res = jldopen("$savename.jld2", "a+") do f
            key = "$(UEG.short(loadparam))"
            return f[key]
        end
        push!(res_list, res)
    end
    println(res_list)

    # Get dimensionless k-grid (k / kF) and index corresponding to the Fermi energy
    k_kf_grid = kgrid / param.kF
    println(k_kf_grid)
    println(length(k_kf_grid))

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    partitions_list = [[(2, 0, 0)], [(1, 1, 0)], [(0, 2, 0)]]
    data = UEG_MC.restodict(res_list, partitions_list)
    println(data)

    # TODO: Rerun with missing sign on (1, 1, 0) data fixed
    for P in keys(data)
        data[P] *= -1          # Remove extra overall -1
        data[(1, 1, 0)] *= -1  # Add missing (-1)^(n_intn)
    end

    # Reexpand merged data in powers of 
    ct_filename = "examples/counterterms/data_Z_$(ct_string).jld2"
    z, μ = UEG_MC.load_z_mu(param; ct_filename=ct_filename)
    δz, δμ = CounterTerm.sigmaCT(2, μ, z; isfock=false, verbose=1)
    # Test manual renormalization with exact lowest-order chemical potential
    # δμ1 = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
    δμ1 = SelfEnergy.Fock0_ZeroTemp(param.kF, param.basic)  # = ReΣ₁[λ](kF, 0)

    println(UEG.paraid(param))

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # Bare occupation fₖ on dense grid for plotting
    kgrid_fine = param.kF * LinRange(0.75, 1.25, 500)
    k_kf_grid_fine = LinRange(0.75, 1.25, 500)
    ϵk_fine = @. kgrid_fine^2 / (2 * param.me) - param.μ

    # Fermi function
    fe_fine = -Spectral.kernelFermiT.(-1e-8, ϵk_fine, param.β)
    # First derivative of the Fermi function f'(ϵₖ)
    fpe_fine = -Spectral.kernelFermiT_dω.(-1e-8, ϵk_fine, param.β)
    # Second derivative of the Fermi function f''(ϵₖ)
    fppe_fine = -Spectral.kernelFermiT_dω2.(-1e-8, ϵk_fine, param.β)
    # Third derivative of the Fermi function f'''(ϵₖ)
    fpppe_fine = -Spectral.kernelFermiT_dω3.(-1e-8, ϵk_fine, param.β)

    fppe_exact =
        @. (param.β^2 / 4) * sech(param.β * ϵk_fine / 2)^2 * tanh.(param.β * ϵk_fine / 2)
    abs_errs = @. abs((fppe_fine - fppe_exact) / fppe_exact)
    worst_abs_err = maximum(abs_errs)
    println(
        "Worst absolute percent error (exact vs. computed f''(ϵₖ)): $(100 * worst_abs_err)",
    )

    # Get ratio of exact to calculated result on coarse kgrid
    ϵk = @. kgrid^2 / (2 * param.me) - param.μ
    # fppe = -Spectral.kernelFermiT_dω2.(-1e-8, ϵk, param.β)
    fppe = @. (param.β^2 / 4) * sech(param.β * ϵk / 2)^2 * tanh.(param.β * ϵk / 2)

    # δn^{(1)}_F(kσ) = f'(ϵₖ) (Σ^λ_F(k) - Σ^λ_F(k_F)) = f'(ϵₖ) (Σ^λ_F(k) + δμ₁)
    # δn^{(2)}_F(kσ) = f''(ϵₖ) (Σ^λ_F(k) - Σ^λ_F(k_F))^2 = f''(ϵₖ) (Σ^λ_F(k) + δμ₁)^2
    # sigma_lambda_F_fine = sigma_lambda_fock_exact.(kgrid_fine, [param])
    sigma_lambda_F_fine = SelfEnergy.Fock0_ZeroTemp.(kgrid_fine, [param.basic])
    # sigma_lambda_F_fine = sigma_lambda_fock_exact.(kgrid_fine, [param])
    dn1F_exact_fine = @. -fpe_fine * (sigma_lambda_F_fine - δμ1)
    dn2F_exact_fine = @. -fppe_fine * (δμ1^2 / 2 + (sigma_lambda_F_fine - δμ1)^2 / 2)

    @assert δμ1 ≈ sigma_lambda_fock_exact(param.kF, param)

    println("f''(0): $(Spectral.kernelFermiT_dω2.(-1e-8, 0.0, param.β))")

    # Mean and error bars for the computed results
    dn2F_calc = data[(2, 0, 0)] + δμ[1] * data[(1, 1, 0)] + δμ[1]^2 * data[(0, 2, 0)]
    dn2F_calc_exact_mu = data[(2, 0, 0)] + δμ1 * data[(1, 1, 0)] + δμ1^2 * data[(0, 2, 0)]
    dn2F_means = Measurements.value.(dn2F_calc)
    dn2F_stdevs = Measurements.uncertainty.(dn2F_calc)
    dn2F_means_exact_mu = Measurements.value.(dn2F_calc_exact_mu)
    dn2F_stdevs_exact_mu = Measurements.uncertainty.(dn2F_calc_exact_mu)

    sigma_lambda_F = SelfEnergy.Fock0_ZeroTemp.(kgrid, [param.basic])
    # sigma_lambda_F = sigma_lambda_fock_exact.(kgrid, [param])
    dn2F_exact = @. -fppe * (δμ1^2 / 2 + (sigma_lambda_F - δμ1)^2 / 2)
    ratio = dn2F_exact ./ dn2F_means_exact_mu
    max_ratio = argmax(abs, ratio)
    println(ratio)
    println(max_ratio)

    term1_exact_fine = @. -fppe_fine * sigma_lambda_F_fine^2 / 2
    term2_exact_fine = @. fppe_fine * sigma_lambda_F_fine
    term3_exact_fine = -fppe_fine
    dn2F_exact_fine = term1_exact_fine + term2_exact_fine * δμ1 + term3_exact_fine * δμ1^2

    # Plot each term in the 2nd order Fock series benchmark
    fig, ax = plt.subplots()
    # Compare to automatically calculated results
    rs = 1.0
    beta = 40.0
    mass2 = 1.0
    solver = :vegasmc
    # Number of evals
    neval = 1e9
    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    max_order = 3
    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = false
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
    if isFock
        ct_string *= "_noFock"
    end
    # UEG parameters for MC integration
    loadparam = ParaMC(;
        order=max_order,
        rs=rs,
        beta=beta,
        mass2=mass2,
        isDynamic=false,
        isFock=isFock,
    )
    # Load the raw data
    savename =
        "results/data/occupation_n=$(max_order)_rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(mass2)_neval=$(neval)_$(solver)_$(ct_string)"
    orders, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(loadparam))"
        return f[key]
    end
    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data_auto = UEG_MC.restodict(res, partitions)
    for P in keys(data_auto)
        data_auto[P] *= -1  # Remove extra overall -1
    end
    # Mean and error bars for the computed results
    # dn2F_calc_auto =
    #     data_auto[(2, 0, 0)] + δμ[1] * data_auto[(1, 1, 0)] + δμ[1]^2 * data_auto[(0, 2, 0)]
    dn2F_calc_auto_exact_mu =
        data_auto[(2, 0, 0)] + δμ1 * data_auto[(1, 1, 0)] + δμ1^2 * data_auto[(0, 2, 0)]
    # dn2F_means_auto = Measurements.value.(dn2F_calc_auto)
    # dn2F_stdevs_auto = Measurements.uncertainty.(dn2F_calc_auto)
    dn2F_means_exact_mu_auto = Measurements.value.(dn2F_calc_auto_exact_mu)
    # dn2F_stdevs_exact_mu_auto = Measurements.uncertainty.(dn2F_calc_auto_exact_mu)
    for (i, (P,)) in enumerate(partitions_list)
        # Get means and error bars from the result up to this order
        means = Measurements.value.(data_auto[P])
        stdevs = Measurements.uncertainty.(data_auto[P])
        marker = "o-"
        ax.plot(
            k_kf_grid,
            means,
            marker;
            markersize=3,
            color="C$(i+2)",
            label="\$\\mathcal{P} = $P\$ ($solver, auto)",
        )
        ax.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            color="C$(i+2)",
            alpha=0.4,
        )
    end
    ax.plot(
        k_kf_grid_fine,
        term1_exact_fine,
        "-";
        label="\$-\\frac{1}{2} f^{\\prime\\prime}(\\xi_k) \\left(\\Sigma^\\lambda_F(k)\\right)^2\$ (exact)",
    )
    ax.plot(
        k_kf_grid_fine,
        term2_exact_fine,
        "-";
        label="\$f^{\\prime\\prime}(\\xi_k) \\Sigma^\\lambda_F(k)\$ (exact)",
    )
    ax.plot(
        k_kf_grid_fine,
        term3_exact_fine,
        "-";
        label="\$-f^{\\prime\\prime}(\\xi_k)\$ (exact)",
    )
    # Results with manual diagram generation
    for (i, (P,)) in enumerate(partitions_list)
        # Get means and error bars from the result up to this order
        means = Measurements.value.(data[P])
        # stdevs = Measurements.uncertainty.(data[P])
        marker = "o"
        ax.plot(
            k_kf_grid,
            means,
            marker;
            markersize=3,
            color="C$(i-1)",
            label="\$i=$i \\in $P\$ ($solver, manual)",
        )
        # ax.fill_between(
        #     k_kf_grid,
        #     means - stdevs,
        #     means + stdevs;
        #     color="C$(i-1)",
        #     alpha=0.4,
        # )
    end
    ax.legend(; loc="upper left")
    ax.set_ylim(nothing, 20)
    ax.set_xlim(0.75, 1.25)
    # ax.set_xlim(0.25, 1.75)
    # ax.set_ylim(-27, 27)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$-\\delta n^{(2)}_{i/\\mathcal{P}}({k\\sigma})\$")
    xloc = 1.025
    yloc = -5
    ydiv = -4
    ax.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(
        "results/occupation/occupation_shift_n=2_fock_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver)_partitions.pdf",
    )

    # Plot the Fock series contribution to the second-order occupation shift
    fig, ax = plt.subplots()
    ax.axvline(1.0; linestyle="--", linewidth=1, color="gray")
    # Full result from automatic diagram generation
    ax.plot(
        k_kf_grid,
        # dn2F_means,
        dn2F_means_exact_mu_auto,
        "o-";
        markersize=3,
        color="C0",
        label="\$-\\delta n^{(2)}_{(2,0,0)} - \\delta n^{(2)}_{(1,1,0)} - \\delta n^{(2)}_{(0,2,0)}\$ ($solver, auto)",
    )
    # Result from manual diagram generation
    marker = "o"
    ax.plot(
        k_kf_grid,
        # dn2F_means,
        dn2F_means_exact_mu,
        marker;
        markersize=3,
        color="C1",
        label="\$-\\delta n^{(2)}_{i=1} - \\delta n^{(2)}_{i=2} - \\delta n^{(2)}_{i=3}\$ ($solver, manual)",
    )
    # Exact result
    ax.plot(
        k_kf_grid_fine,
        dn2F_exact_fine,
        "C1";
        label="\$-f^{\\prime\\prime}(\\xi_k) \\left[\\frac{1}{2}\\left(\\Sigma^\\lambda_F(k_F)\\right)^2  + \\frac{1}{2}\\left(\\Sigma^\\lambda_F(k) - \\Sigma^\\lambda_F(k_F)\\right)^2\\right]\$ (exact)",
    )
    # ax.plot(
    #     k_kf_grid_fine,
    #     term1_exact_fine + term2_exact_fine * δμ1 + term3_exact_fine * δμ1^2,
    #     "k";
    #     label="exact",
    # )

    # ax.fill_between(
    #     k_kf_grid,
    #     # dn2F_means - dn2F_stdevs,
    #     # dn2F_means + dn2F_stdevs;
    #     dn2F_means_exact_mu - dn2F_stdevs_exact_mu,
    #     dn2F_means_exact_mu + dn2F_stdevs_exact_mu;
    #     color="r",
    #     alpha=0.4,
    # )
    ax.legend(; loc="upper left", framealpha=1)
    ax.set_xlim(0.75, 1.25)
    ax.set_ylim(nothing, 4)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$-\\delta n^{(2)}({k\\sigma})\$")
    xloc = 1.025
    yloc = -1
    ydiv = -0.75
    ax.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(
        "results/occupation/occupation_shift_n=2_fock_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver).pdf",
    )

    # # Plot the occupation shift to second order in the Fock series
    # dn0F_total = fe_fine
    # dn1F_total = dn0F_total + dn1F_exact_fine
    # dn2F_total = dn1F_total + dn2F_exact_fine
    # fig, ax = plt.subplots()
    # ax.axhspan(0, 1; alpha=0.2, facecolor="k", edgecolor=nothing)
    # ax.axvline(1.0; linestyle="--", linewidth=1, color="gray")
    # # Exact results
    # ax.plot(k_kf_grid_fine, dn0F_total; label="\$N=0\$")
    # # Add back exact screened Fock series contribution to full N=0 result
    # ax.plot(k_kf_grid_fine, dn2F_exact_fine + dn0F_total; label="\$N=0\$ plus \$n=2\$")
    # ax.legend(; loc="best")
    # ax.set_xlim(0.75, 1.25)
    # ax.set_ylim(-19, 19)
    # ax.set_xlabel("\$k / k_F\$")
    # ax.set_ylabel(
    #     "\$n_N(k,\\sigma)\$",
    # )
    # xloc = 1.03
    # yloc = -8.5
    # ydiv = -6
    # ax.text(
    #     xloc,
    #     yloc,
    #     "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta), \\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}}\$";
    #     fontsize=12,
    # )
    # fig.tight_layout()
    # fig.savefig("results/occupation/occupation_N=1_plus_N=2_fock.pdf")

    # # Plot the occupation Fock insertion series
    # fig, ax = plt.subplots()
    # # ax.axhspan(0, 1; alpha=0.2, facecolor="k", edgecolor=nothing)
    # ax.axhline(0.0; linestyle="-", linewidth=0.5, color="k")
    # ax.axhline(1.0; linestyle="-", linewidth=0.5, color="k")
    # ax.axvline(1.0; linestyle="--", linewidth=1, color="gray")
    # # Exact results
    # ax.plot(k_kf_grid_fine, dn0F_total; label="\$N=0\$")
    # ax.plot(k_kf_grid_fine, dn1F_total; label="\$N=1\$")
    # ax.plot(k_kf_grid_fine, dn2F_total; label="\$N=2\$")
    # # ax.plot(
    # #     k_kf_grid_fine,
    # #     dn3F_total;
    # #     label="\$N=3\$",
    # # )
    # ax.legend(; loc="best")
    # ax.set_xlim(0.75, 1.25)
    # # ax.set_ylim(-17, 22)
    # ax.set_xlabel("\$k / k_F\$")
    # ax.set_ylabel(
    #     "\$\\sum^N_{n=0} f^{(n)}(\\xi_k) (\\Sigma^\\lambda_F(k) - \\Sigma^\\lambda_F(k_F))^n \$",
    # )
    # xloc = 1.03
    # yloc = -8.5
    # ydiv = -6
    # ax.text(
    #     xloc,
    #     yloc,
    #     "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta), \\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}}\$";
    #     fontsize=12,
    # )
    # fig.tight_layout()
    # fig.savefig(
    #     "results/occupation/occupation_shift_N=2_fock_exact_rs=$(param.rs)_" *
    #     "beta_ef=$(param.beta)_lambda=$(param.mass2).pdf",
    # )

    plt.close("all")
    return
end

main()
