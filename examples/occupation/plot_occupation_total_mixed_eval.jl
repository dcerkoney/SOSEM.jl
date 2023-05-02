using CodecZlib
using DataFrames
using DelimitedFiles
using ElectronGas
using ElectronLiquid
using Interpolations
using JLD2
using Lehmann
using LsqFit
using Measurements
using Parameters
using Polynomials
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt
@pyimport mpl_toolkits.axes_grid1.inset_locator as il

"""Compute the coefficient of determination (r-squared) for a dataset."""
function rsquared(xs, ys, yhats)
    ybar = sum(yhats) / length(yhats)
    ss_res = sum((ys .- yhats) .^ 2)
    ss_tot = sum((ys .- ybar) .^ 2)
    return 1 - ss_res / ss_tot
end
function rsquared(xs, ys, yhats, fit::LsqFit.LsqFitResult)
    ybar = sum(yhats) / length(yhats)
    ss_res = sum(fit.resid .^ 2)
    ss_tot = sum((ys .- ybar) .^ 2)
    return 1 - ss_res / ss_tot
end

# TODO: Switch to scipy curve_fit / least_squares with bounds on (Z, m⋆).
#       Consider using Optim or scipy equivalent for LSQ fit to data with
#       error bars (see notes).
function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    # rs = 1.0
    rs = 2.0
    beta = 40.0
    # mass2 = 1.0
    mass2 = 0.4
    solver = :vegasmc

    # Number of evals
    # neval12 = rs == 1 ? 1e10 : 5e10  # neval12 for existing data has this discrepancy
    # neval12 = 1e10
    # neval12 = 5e10
    neval3 = 5e10
    neval4 = 5e10
    # neval = max(neval12, neval3, neval4)
    neval = max(neval3, neval4)

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order = 0
    max_order = 4
    min_order_plot = 0
    max_order_plot = 4

    # Save total results
    save = true

    # Include the LQS quasiparticle fit for (Z, m⋆) in the plot?
    plot_fit = false

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Remove Fock insertions?
    isFock = false

    # Ignore measured mu/lambda partitions?
    fix_mu = false
    fix_lambda = false
    fix_string = fix_mu || fix_lambda ? "_fix" : ""
    if fix_mu
        fix_string *= "_mu"
    end
    if fix_lambda
        fix_string *= "_lambda"
    end

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "with_ct" : ""
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
    if max_order > 3
        max_together = 3
    else
        max_together = max_order
    end
    # if max_order > 2
    #     max_together = 2
    # else
    #     max_together = max_order
    # end
    savename =
        "results/data/occupation_n=$(max_together)_rs=$(rs)_beta_ef=$(beta)_" *
        # "results/data/occupation_n=$(max_order)_rs=$(rs)_beta_ef=$(beta)_" *
        # "lambda=$(mass2)_neval=$(neval12)_$(solver)_$(ct_string)"
        "lambda=$(mass2)_neval=$(neval3)_$(solver)_$(ct_string)"
    print("Loading data from $savename...")
    orders, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(loadparam))"
        return f[key]
    end
    println("done!")
    # if max_order >= 3
    #     # 3rd order 
    #     savename =
    #         "results/data/occupation_n=3_rs=$(rs)_beta_ef=$(beta)_" *
    #         "lambda=$(mass2)_neval=$(neval3)_$(solver)_$(ct_string)"
    #     print("Loading 3rd order data from $savename...")
    #     orders3, param3, kgrid3, partitions3, res3 = jldopen("$savename.jld2", "a+") do f
    #         key = "$(UEG.short(loadparam))"
    #         return f[key]
    #     end
    #     println("done!")
    # end
    if max_order >= 4
        # 4th order 
        savename =
            "results/data/occupation_n=4_rs=$(rs)_beta_ef=$(beta)_" *
            "lambda=$(mass2)_neval=$(neval4)_$(solver)_$(ct_string)"
        print("Loading 4th order data from $savename...")
        orders4, param4, kgrid4, partitions4, res4 = jldopen("$savename.jld2", "a+") do f
            key = "$(UEG.short(loadparam))"
            return f[key]
        end
        println("done!")
    end

    # Get dimensionless k-grid (k / kF) and index corresponding to the Fermi energy
    k_kf_grid = kgrid / param.kF
    # if max_order >= 3
    #     k_kf_grid3 = kgrid3 / param.kF
    # end
    if max_order >= 4
        k_kf_grid4 = kgrid4 / param.kF
    end

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    # NOTE: Old data for orders 0, 1, and 2 at rs = 1 has Taylor factors already present in eval!
    # if loadparam.rs != 1.0
    #     for (k, v) in data
    #         data[k] = v / (factorial(k[2]) * factorial(k[3]))
    #     end
    # end
    for (k, v) in data
        data[k] = v / (factorial(k[2]) * factorial(k[3]))
    end
    # # Add 3rd order results to data dict
    # if max_order >= 3
    #     data3 = UEG_MC.restodict(res3, partitions3)
    #     for (k, v) in data3
    #         data3[k] = v / (factorial(k[2]) * factorial(k[3]))
    #     end
    #     merge!(data, data3)
    # end
    # Add 4th order results to data dict
    if max_order >= 4
        data4 = UEG_MC.restodict(res4, partitions4)
        for (k, v) in data4
            data4[k] = v / (factorial(k[2]) * factorial(k[3]))
            # NOTE: we delete the point at kF for consistency with old data
            # @assert length(data4[k]) == length(kgrid) + 1
            # deleteat!(data4[k], kgrid4 .== param.kF)
            @assert length(data4[k]) == length(kgrid)
        end
        # NOTE: we delete the point at kF for consistency with old data
        # @assert length(kgrid4) == length(kgrid) + 1
        # deleteat!(kgrid4, kgrid4 .== param.kF)
        merge!(data, data4)
        @assert length(kgrid4) == length(kgrid)
    end

    # Zero out partitions with mu renorm if present (fix mu)
    if renorm_mu == false || fix_mu
        for P in keys(data)
            if P[2] > 0
                println("Fixing mu without lambda renorm, ignoring n_k partition $P")
                data[P] = zero(data[P])
            end
        end
    end
    # Zero out partitions with lambda renorm if present (fix lambda)
    if renorm_lambda == false || fix_lambda
        for P in keys(data)
            if P[3] > 0
                println("No lambda renorm, ignoring n_k partition $P")
                data[P] = zero(data[P])
            end
        end
    end

    println(data)
    for P in keys(data)
        # Undo extra minus sign factor
        data[P] = -data[P]
    end

    # Zero out double-counted (Fock renormalized) partitions
    if isFock && min_order ≤ 1
        data[(1, 0, 0)] = zero(data[(max_order, 0, 0)])
    end

    merged_data = CounterTerm.mergeInteraction(data)

    # Get exact bare/Fock occupation
    if param.isFock
        fock =
            SelfEnergy.Fock0_ZeroTemp.(kgrid, [param.basic]) .-
            SelfEnergy.Fock0_ZeroTemp(param.kF, param.basic)
        ϵk = kgrid .^ 2 / (2 * param.me) .- param.μ + fock  # ϵ_HF = ϵ_0 + (Σ_F(k) - δμ₁)
    else
        ϵk = kgrid .^ 2 / (2 * param.me) .- param.μ         # ϵ_0
    end
    bare_occupation_exact = -Spectral.kernelFermiT.(-1e-8, ϵk, param.β)

    # Set bare result manually using exact Fermi function
    if haskey(merged_data, (0, 0)) == false
        # treat quadrature data as numerically exact
        merged_data[(0, 0)] = [measurement(bare_occupation_exact, 0.0)]
    end
    if min_order == 0
        stdscores = stdscore.(merged_data[(0, 0)], bare_occupation_exact)
        worst_score = argmax(abs, stdscores)
        println("Worst standard score for N=0 (measured): $worst_score")
        # println("Scores:\n$stdscores")
    end

    # Reexpand merged data in powers of μ
    ct_filename = "examples/counterterms/data_Z_$(ct_string_short).jld2"
    z, μ = UEG_MC.load_z_mu(param; ct_filename=ct_filename)
    # Add Taylor factors to CT data
    for (p, v) in z
        z[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    for (p, v) in μ
        μ[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    # Zero out partitions with mu renorm if present (fix mu)
    if renorm_mu == false || fix_mu
        for P in keys(μ)
            if P[2] > 0
                println("Fixing mu without lambda renorm, ignoring μ partition $P")
                μ[P] = zero(μ[P])
            end
        end
    end
    # Zero out partitions with lambda renorm if present (fix lambda)
    if renorm_lambda == false || fix_lambda
        for P in keys(μ)
            if P[3] > 0
                println("No lambda renorm, ignoring μ partition $P")
                μ[P] = zero(μ[P])
            end
        end
    end
    δz, δμ = CounterTerm.sigmaCT(max_order, μ, z; isfock=isFock, verbose=1)

    println("Computed δμ: ", δμ)
    # δμ[2] = measurement("-0.08196(8)")  # Replace δμ[2] with benchmark dMu2 value
    occupation = UEG_MC.chemicalpotential_renormalization_density(
        merged_data,
        δμ;
        min_order=0,
        max_order=max_order,
    )
    occupation_2_manual =
        merged_data[(2, 0)] +
        δμ[1] * merged_data[(1, 1)] +
        δμ[1]^2 * merged_data[(0, 2)] +
        δμ[2] * merged_data[(0, 1)]
    scores_2 = stdscore.(occupation[2] - occupation_2_manual, 0.0)
    println("Deviation: ", occupation[2] - occupation_2_manual)
    worst_score_2 = argmax(abs, scores_2)
    println("2nd order renorm vs manual worst score: $worst_score_2")

    if max_order ≥ 1 && renorm_mu == true && isFock == false && fix_mu == false
        # Test manual renormalization with exact lowest-order chemical potential
        δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
        # nₖ⁽¹⁾ = nₖ_{1,0} + δμ₁ nₖ_{0,1}
        occupation_1_manual = merged_data[(1, 0)] + δμ1_exact * merged_data[(0, 1)]
        stdscores = stdscore.(occupation[1], occupation_1_manual)
        worst_score = argmax(abs, stdscores)
        println("Exact δμ₁: ", δμ1_exact)
        println("Computed δμ₁: ", δμ[1])
        println(
            "Worst standard score for total result to 1st " *
            "order (auto vs exact+manual): $worst_score",
        )
        @assert worst_score ≤ 10
    end
    # Aggregate the full results for Σₓ up to order N
    occupation_total = UEG_MC.aggregate_orders(occupation)

    println(UEG.paraid(param))
    println(partitions)
    println(res)

    if save
        savename =
            "results/data/rs=$(param.rs)_beta_ef=$(param.beta)_" *
            "lambda=$(param.mass2)_$(intn_str)$(solver)_$(ct_string)"
        f = jldopen("$savename.jld2", "a+"; compress=true)
        # NOTE: no bare result for c1b observable (accounted for in c1b0)
        for N in min_order_plot:max_order
            if N == 4
                num_eval = neval4
            # elseif N == 3
            #     num_eval = neval3
            else
                num_eval = neval3
                # num_eval = neval12
            end
            # num_eval = neval
            # Skip exact (N = 0) result
            N == 0 && continue
            # Skip Fock result if HF renormalization was used
            isFock && N == 1 && continue
            # Update existing results if applicable
            if haskey(f, "occupation") &&
               haskey(f["occupation"], "N=$N") &&
               haskey(f["occupation/N=$N"], "neval=$num_eval")
                @warn("replacing existing data for N=$N, neval=$num_eval")
                delete!(f["occupation/N=$N"], "neval=$num_eval")
            end
            f["occupation/N=$N/neval=$num_eval/meas"] = occupation_total[N]
            f["occupation/N=$N/neval=$num_eval/param"] = param
            f["occupation/N=$N/neval=$num_eval/kgrid"] = kgrid
        end
    end

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # Bare/Fock occupation on dense grid for plotting
    kgrid_fine = param.kF * np.linspace(0.0, 3.0; num=600)
    k_kf_grid_fine = np.linspace(0.0, 3.0; num=600)
    ϵk_fine = @. kgrid_fine^2 / (2 * param.me) - param.μ
    fe_fine = -Spectral.kernelFermiT.(-1e-8, ϵk_fine, param.β)  # f(ϵk)

    if param.isFock
        fock =
            SelfEnergy.Fock0_ZeroTemp.(kgrid_fine, [param.basic]) .-
            SelfEnergy.Fock0_ZeroTemp(param.kF, param.basic)
        ϵk = kgrid_fine .^ 2 / (2 * param.me) .- param.μ + fock  # ϵ_HF = ϵ_0 + (Σ_F(k) - δμ₁)
    else
        ϵk = kgrid_fine .^ 2 / (2 * param.me) .- param.μ         # ϵ_0
    end
    bare_occupation_fine = -Spectral.kernelFermiT.(-1e-8, ϵk, param.β)

    # Plot the occupation number for each aggregate order
    fig, ax = plt.subplots()
    ax.axvline(1.0; linestyle="--", linewidth=1, color="gray")

    # Shade the thermal broadening region of the Fermi function
    # NOTE: the FWHM of the Fermi distribution is ~ 3.53 kB T
    # fermi_hwhm = 3.53 / (2 * param.β * param.kF)  # HWHM in units of kF
    # fermi_hwhm = 3.53 / (2 * param.beta)  # HWHM in units of kF

    # HWHM of -f'(ϵ) = ln((√2 + 1) / (√2 - 1)) / β (in units of EF; beta = β EF)
    fermi_hwhm_over_ef = log((sqrt(2) + 1) / (sqrt(2) - 1)) / param.beta

    # ϵ_± = ϵF ± Δ_HWHM
    e_ef_minus = 1 - fermi_hwhm_over_ef
    e_ef_plus  = 1 + fermi_hwhm_over_ef

    # k_± = √(2 m ϵ_±) = kF √(1 ± Δ_HWHM / ϵF)
    # k_kf_minus = 1 - fermi_hwhm_over_ef / 2  # ≈ √(1 - Δ_HWHM / ϵF)    (to linear order in Δ)
    # k_kf_plus  = 1 + fermi_hwhm_over_ef / 2  # ≈ √(1 + Δ_HWHM / ϵF)  (to linear order in Δ)
    k_kf_minus = sqrt(e_ef_minus)  # = √(1 - Δ_HWHM / ϵF)
    k_kf_plus  = sqrt(e_ef_plus)   # = √(1 + Δ_HWHM / ϵF)

    # Use maximum broadening (sqrt is a non-linear transformation, so abs(k - k_+) ≠ abs(k - k_-)!)
    dk_max = max(abs(1 - k_kf_minus), abs(k_kf_plus - 1))
    # k_kf_lesser  = 1 - dk_max
    # k_kf_greater = 1 + dk_max

    # TODO: make this a function of # points near kF instead, and make convergence plots of Z and m*
    #       vs N for different window sizes. Are the results heavily dependent on the window size?
    # W ≡ window_scale * Δ_FWHM
    window_scale = 1.0
    k_kf_lesser  = 1 - window_scale * dk_max
    k_kf_greater = 1 + window_scale * dk_max

    lwindow = param.kF * k_kf_lesser
    rwindow = param.kF * k_kf_greater
    ax.axvspan(
        lwindow / param.kF,
        rwindow / param.kF;
        # 1 - fermi_hwhm,
        # 1 + fermi_hwhm;
        color="0.9",
        label="\$\\mathrm{FWHM}_{k}(f_{k\\sigma})\$",
    )

    # if min_order_plot == 0
    # Include bare occupation fₖ in plot
    # ax.plot(k_kf_grid_fine, bare_occupation_fine, "k"; label="\$N=0\$ (exact)")
    # end

    ikFplus = searchsortedfirst(k_kf_grid, 1.0)  # Index where k ⪆ kF
    ikFminus = ikFplus - 1
    ic = 1
    colors = ["C0", "C1", "C2", "C3", "C4"]
    # colors = ["orchid", "cornflowerblue", "turquoise", "chartreuse"]
    colors2 = ["purple", "blue", "green", "yellow", "orange"]
    zfactors = []
    mstar_over_ms = []
    for (i, N) in enumerate(min_order:max_order_plot)
        # if plot_fit
        #     N < max_order_plot && continue
        # end
        # N == 0 && continue
        isFock && N == 1 && continue
        # Plot measured data
        means = Measurements.value.(occupation_total[N])
        stdevs = Measurements.uncertainty.(occupation_total[N])

        # qp fit at lowest and highest orders
        if plot_fit
            # lwindow = param.kF * sqrt(1 - 4 * fermi_hwhm / param.EF)
            # rwindow = param.kF * sqrt(1 + 4 * fermi_hwhm / param.EF)

            # Fitting constant C fixed by value at kF, C ≈ n(kF) - Z/2. 
            # Since k = kF is not on-grid, use the midpoint of the nearest two points
            # (linear interpolation, where the kgrid is assumed symmetrical about k = kF)
            means_kF = (means[ikFplus] + means[ikFminus]) / 2.0  # n(kF) ≈ (n(kF⁻) + n(kF⁺)) / 2
            # ξ⋆(k) & f(ξ⋆(k)) on-grid, ξk = ϵ⋆(k) - μ⋆
            # @. ξstar_k(k, p) = @. (k^2 - param.kF^2) / (2 * p[2])
            # @. fξstar_k(k, p) = @. -Spectral.kernelFermiT.(-1e-8, ξstar_k(k, p), param.β)
            # n_qp(Z, m⋆) = hat(n)(kF) - Z/2 + Zf(k^2 / 2m⋆ - kF^2 / 2m⋆), and p = (Z, m⋆)
            # @. model_Z_mstar(k, p) = means_kF + p[1] * (fξstar_k(k, p) - 1 / 2)
            @. function model_Z_mstar(k, p)
                return means_kF +
                       p[1] * (
                    -Spectral.kernelFermiT.(
                        -1e-8,
                        (k^2 - param.kF^2) / (2 * p[2]),
                        param.β,
                    ) - 1 / 2
                )
            end
            # @. function model_Z_mstar_v2(k, p)
            #     return (means_kF - p[1] / 2) / (1 + 1 / param.β^2) +
            #            p[1] *
            #            -Spectral.kernelFermiT.(
            #         -1e-8,
            #         (k^2 - param.kF^2) / (2 * p[2]),
            #         param.β,
            #     )
            # end
            # Fix m⋆ value to K.H. & K.C. data 
            if rs == 1.0 || rs == 2.0
                if rs == 1.0
                    mstar_khkc = param.me * 0.955
                elseif rs == 2.0
                    mstar_khkc = param.me * 0.943
                end
                @. ξstar_khkc(k) = @. (k^2 - param.kF^2) / (2 * mstar_khkc)
                @. function fξstar_khkc(k)
                    @. -Spectral.kernelFermiT.(-1e-8, ξstar_khkc(k), param.β)
                end
                # @. model_Z(k, p) = means_kF + p[1] * (fξstar_khkc(k) - 1 / 2)
                @. function model_Z(k, p)
                    return means_kF +
                           p[1] * (
                        -Spectral.kernelFermiT.(
                            -1e-8,
                            (k^2 - param.kF^2) / (2 * mstar_khkc),
                            param.β,
                        ) - 1 / 2
                    )
                end
            end
            # Initial parameters for curve fitting
            p0_Z_mstar = [1.0, 0.5]  # Z=1 and m=mₑ (in Rydbergs)
            p0_Z       = [1.0]        # m=mₑ

            # Gridded data for k in window near kF
            k_data = kgrid[lwindow .≤ kgrid .≤ rwindow]
            means_data = means[lwindow .≤ kgrid .≤ rwindow]

            # Least-squares quasiparticle fit
            fit_N = curve_fit(model_Z_mstar, k_data, means_data, p0_Z_mstar)
            qp_fit_N(k) = model_Z_mstar(k, fit_N.param)
            # fit_N = curve_fit(model_Z_mstar_v2, k_data, means_data, p0_Z_mstar)
            # qp_fit_N(k) = model_Z_mstar_v2(k, fit_N.param)
            # Coefficients of determination (r²)
            r2 = rsquared(k_data, means_data, qp_fit_N(k_data), fit_N)
            # Fermi-liquid parameters (Z, m⋆) (on the Fermi surface k = kF) from quasiparticle fit
            println("(N=$N): (Z, m⋆) ≈ $(fit_N.param), r2=$r2")
            push!(zfactors, fit_N.param[1])
            push!(mstar_over_ms, fit_N.param[2] / param.me)
            zfactor_estimate, mstar_estimate = fit_N.param
            # Estimate errors and covariance matrix for fit params Z and m⋆
            errors_est = estimate_errors(fit_N)
            covariance_est = estimate_covar(fit_N)
            println(
                "(N=$N) Estimated errors and covariance matrix for fit parameters (Z, m⋆):",
            )
            println(errors_est)
            println(covariance_est)
            # Another LSQ fit, but using K.H. & K.C. data for m⋆/m at rs = 1, 2
            if rs == 1.0 || rs == 2.0
                fit_N_khkc = curve_fit(model_Z, k_data, means_data, p0_Z)
                qp_fit_N_khkc(k) = model_Z(k, fit_N_khkc.param)
                # Coefficients of determination (r²)
                r2 = rsquared(k_data, means_data, qp_fit_N_khkc(k_data), fit_N_khkc)
                # Fermi-liquid parameters (Z, m⋆) (on the Fermi surface k = kF) from quasiparticle fit
                println(
                    "(N=$N): m⋆ ≡ m⋆_KHKC = $mstar_khkc, Z ≈ $(fit_N_khkc.param), r2=$r2",
                )
                zfactor_estimate_khkc = fit_N_khkc.param[1]
                # Estimate error for fit param Z
                error_est = estimate_errors(fit_N_khkc)
                println("(N=$N) Estimated error for fit parameter Z:")
                println(error_est)
            end
        end

        marker = "o-"
        ax.plot(
            k_kf_grid,
            means,
            marker;
            markersize=2,
            # color="C$ic",
            color="$(colors[ic])",
            label="\$N=$N\$ ($solver)",
            # label="\$N=$N\$ ($solver, SOSEM)",
        )
        ax.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            # color="C$ic",
            color="$(colors[ic])",
            alpha=0.4,
        )
        # Plot Fqp(Z, m⋆) best fit to fk and Z fk (lowest and max orders)
        if N == max_order_plot && plot_fit
            ax.plot(
                k_kf_grid_fine,
                # qp_fit_N_khkc(kgrid),
                qp_fit_N(kgrid_fine);
                # color="C$ic",
                # color="$(colors[ic])",
                color="k",
                # label="\$N=$N\$ (LSQ fit to \$F(Z, m^\\star = $mstar_khkc})\$)",
                label="\$N=$N\$ (LSQ fit to \$F(Z, m^\\star)\$)",
                # label="\$N=$N\$ ($solver, SOSEM)",
            )
            ax.text(
                0.25,
                0.8,
                # "(\$N=$N\$) \$Z \\approx $(round(zfactor_estimate; digits=4))\$";
                "\$Z \\approx $(round(zfactor_estimate; digits=4))\$";
                fontsize=14,
            )
            ax.text(
                0.25,
                # 0.6 - (N / 30.0),
                0.7,
                # "(\$N=$N\$) \$m^\\star \\approx $(round(mstar_estimate; digits=4))\$";
                "\$m^\\star / m \\approx $(round(mstar_estimate / param.me; digits=4))\$";
                fontsize=14,
            )
            ax.plot(
                k_kf_grid_fine,
                qp_fit_N_khkc(kgrid_fine);
                markersize=2,
                color="gray",
                label="\$N=$N\$ (LSQ fit to \$F_{KHKC}(Z)\$)",
            )
            ax.text(
                0.25,
                0.6,
                "\$Z_{KHKC} \\approx $(round(zfactor_estimate_khkc; digits=4))\$";
                fontsize=14,
            )
            ax.text(
                0.25,
                0.5,
                "\$m^\\star_{KHKC} / m = $(round(mstar_khkc / param.me; digits=4))\$";
                fontsize=14,
            )
        end

        # Extrapolate Z-factor to kF⁻ & kF⁺ at the maximum order
        # if N == max_order_plot || N == 0
        # # Grid data outside of thermal broadening window
        # k_kf_grid_lower = k_kf_grid[k_kf_grid .≤ k_minus / param.kF]
        # k_kf_grid_upper = k_kf_grid[k_kf_grid .≥ k_plus / param.kF]
        # zfactor_lower = means[k_kf_grid .≤ k_minus / param.kF]
        # zfactor_upper = means[k_kf_grid .≥ k_plus / param.kF]

        # # Interpolate Z-factor curves below and above kF
        # zfactor_lower_interp = linear_interpolation(
        #     k_kf_grid_lower,
        #     zfactor_lower;
        #     extrapolation_bc=Line(),
        # )
        # zfactor_upper_interp = linear_interpolation(
        #     k_kf_grid_upper,
        #     zfactor_upper;
        #     extrapolation_bc=Line(),
        # )
        # # TODO: need to do this the hard way for cubic interp
        # # zfactor_lower_interp = cubic_spline_interpolation(k_kf_grid_lower, zfactor_lower; extrapolation_bc=Line())
        # # zfactor_upper_interp = cubic_spline_interpolation(k_kf_grid_upper, zfactor_upper; extrapolation_bc=Line())

        # # Plot the interpolated data with T=0 extrapolation to kF⁻ & kF⁺
        # k_kf_lower_fine = LinRange(0.0, 1.0, 200)
        # k_kf_upper_fine = LinRange(1.0, 2.0, 200)
        # zfactor_lower_fine = zfactor_lower_interp.(k_kf_lower_fine)
        # zfactor_upper_fine = zfactor_upper_interp.(k_kf_upper_fine)
        # zfactor_estimate = zfactor_lower_interp(1) - zfactor_upper_interp(1)  # n(kF⁻) - n(kF⁺)
        # println("Z-factor estimate: $zfactor_estimate")
        # ax.plot(k_kf_lower_fine, zfactor_lower_fine; linestyle="-", color="k")
        # ax.plot(
        #     k_kf_upper_fine,
        #     zfactor_upper_fine;
        #     linestyle="-",
        #     color="k",
        #     label="\$N=$N\$ (\$T=0\$ extrapolation)",
        # )
        # ax.text(
        #     # 0.25,
        #     0.125,
        #     # 0.6,
        #     0.6 - (N / 30.0),
        #     # "\$Z \\approx $(round(zfactor_estimate; digits=4))\$";
        #     "\$Z \\approx $(round(zfactor_estimate; digits=4))\$ (\$N=$N\$)";
        #     fontsize=14,
        # )
        # end
        ic += 1
    end
    ax.legend(; loc="upper right")
    # ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    # ax.set_xlim(0.75, 1.25)
    ax.set_xlim(0, 2)
    # ax.set_ylim(nothing, 2)
    ax.set_xlabel("\$k / k_F\$")
    if isFock
        prefix = "n^{\\mathrm{F}}_{k\\sigma}"
    else
        prefix = "n_{k\\sigma}"
    end
    lcond = renorm_lambda == false || fix_lambda
    mcond = renorm_mu == false || fix_mu
    if lcond && mcond
        suffix = "(\\lambda, \\mu=\\mu_0)"
    elseif lcond && mcond == false
        suffix = "(\\lambda)"
    elseif lcond == false && mcond
        suffix = "(\\mu)"
    else
        suffix = ""
    end
    ax.set_ylabel("\$$prefix$suffix\$")
    if isFock
        xloc = 1.035
        yloc = 0.6
        ydiv = -0.125
    else
        # xloc = 1.5
        xloc = 1.15
        yloc = 0.5
        ydiv = -0.1
    end
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
    fig.tight_layout()
    fit_string = plot_fit ? "_fit" : ""
    fig.savefig(
        "results/occupation/" *
        "occupation_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_" *
        "$(solver)_$(ct_string)$(fix_string)$(fit_string).pdf",
    )

    if plot_fit
        fig, ax = plt.subplots()
        ax.plot(min_order:max_order_plot, mstar_over_ms, "o-"; markersize=3)
        ax.set_xticks(collect(min_order:max_order_plot))
        ax.set_xlabel("Perturbation order \$N\$")
        ax.set_ylabel("\$m^\\star / m\$")
        fig.tight_layout()
        fig.savefig("results/occupation/effective_mass_ratio_vs_N.pdf")

        fig, ax = plt.subplots()
        ax.plot(min_order:max_order_plot, zfactors, "o-"; markersize=3)
        ax.set_xticks(collect(min_order:max_order_plot))
        ax.set_xlabel("Perturbation order \$N\$")
        ax.set_ylabel("\$Z\$")
        fig.tight_layout()
        fig.savefig("results/occupation/z_factor_vs_N.pdf")
    end

    plt.close("all")
    return
end

main()
