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

function spline(x, y, e)
    # generate knots with spline without constraints
    w = 1.0 ./ e
    spl = interp.UnivariateSpline(x, y; w=w, k=3)
    __x = collect(LinRange(x[1], x[end], 1000))
    # __x = collect(LinRange(x[1], x[end], 100))
    yfit = spl(__x)
    return __x, yfit
end

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

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

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

    rs = 1.0
    # rs = 2.0
    beta = 40.0
    mass2 = 1.0
    # mass2 = 0.4
    solver = :vegasmc

    # Number of evals
    neval12 = rs == 1 ? 1e10 : 5e10  # neval12 for existing data has this discrepancy
    # neval12 = 1e10
    # neval12 = 5e10
    neval3 = 5e10
    neval4 = 5e10
    neval = max(neval12, neval3, neval4)

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    n_min = 0  # True minimal loop order for this observable
    min_order = 0
    max_order = 4
    min_order_plot = 0
    max_order_plot = 4

    # Save total results
    save = false
    # save = true

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

    if max_order > 2
        max_together = 2
    else
        max_together = max_order
    end

    # UEG parameters for MC integration
    para = ParaMC(; order=2, rs=rs, beta=beta, mass2=mass2, isDynamic=false, isFock=isFock)
    para3 = ParaMC(; order=3, rs=rs, beta=beta, mass2=mass2, isDynamic=false, isFock=isFock)
    para4 = ParaMC(; order=4, rs=rs, beta=beta, mass2=mass2, isDynamic=false, isFock=isFock)

    # Load the raw data
    local htf2, htf3, htf4
    savename =
        "results/data/occupation/occupation_n=$(max_together)_rs=$(rs)_beta_ef=$(beta)_" *
        # "results/data/occupation/occupation_n=$(max_order)_rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(mass2)_neval=$(neval12)_$(solver)_$(ct_string)"
    print("Loading data from $savename...")
    orders, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        htf2 = f["has_taylor_factors"]
        key = "$(UEG.short(para))"
        return f[key]
    end
    println("done!")
    if max_order >= 3
        # 3rd order 
        savename =
            "results/data/occupation/occupation_n=3_rs=$(rs)_beta_ef=$(beta)_" *
            "lambda=$(mass2)_neval=$(neval3)_$(solver)_$(ct_string)"
        print("Loading 3rd order data from $savename...")
        orders3, kgrid3, partitions3, res3 = jldopen("$savename.jld2", "a+") do f
            htf3 = f["has_taylor_factors"]
            key = "$(UEG.short(para3))"
            return f[key]
        end
        println("done!")
    end
    if max_order >= 4
        # 4th order 
        savename =
            "results/data/occupation/occupation_n=4_rs=$(rs)_beta_ef=$(beta)_" *
            "lambda=$(mass2)_neval=$(neval4)_$(solver)_$(ct_string)"
        print("Loading 4th order data from $savename...")
        orders4, kgrid4, partitions4, res4 = jldopen("$savename.jld2", "a+") do f
            htf4 = f["has_taylor_factors"]
            key = "$(UEG.short(para4))"
            return f[key]
        end
        println("done!")
    end

    # Get dimensionless k-grid (k / kF) and index corresponding to the Fermi energy
    k_kf_grid = kgrid / para.kF
    if max_order >= 3
        k_kf_grid3 = kgrid3 / para.kF
    end
    if max_order >= 4
        k_kf_grid4 = kgrid4 / para.kF
    end

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    # NOTE: Old data for orders 0, 1, and 2 at rs = 1 has Taylor factors already present in eval!
    if htf2 == false
        for (k, v) in data
            data[k] = v / (factorial(k[2]) * factorial(k[3]))
        end
    end
    # Add 3rd order results to data dict
    if max_order >= 3
        data3 = UEG_MC.restodict(res3, partitions3)
        if htf3 == false
            for (k, v) in data3
                data3[k] = v / (factorial(k[2]) * factorial(k[3]))
            end
        end
        merge!(data, data3)
    end
    # Add 4th order results to data dict
    if max_order >= 4
        data4 = UEG_MC.restodict(res4, partitions4)
        for (k, v) in data4
            if htf4 == false
                data4[k] = v / (factorial(k[2]) * factorial(k[3]))
            end
            # NOTE: we delete the point at kF for consistency with old data
            @assert length(data4[k]) == length(kgrid) + 1
            deleteat!(data4[k], kgrid4 .== para.kF)
            @assert length(data4[k]) == length(kgrid)
        end
        # NOTE: we delete the point at kF for consistency with old data
        @assert length(kgrid4) == length(kgrid) + 1
        deleteat!(kgrid4, kgrid4 .== para.kF)
        merge!(data, data4)
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
    if para.isFock
        fock =
            SelfEnergy.Fock0_ZeroTemp.(kgrid, [para.basic]) .-
            SelfEnergy.Fock0_ZeroTemp(para.kF, para.basic)
        ϵk = kgrid .^ 2 / (2 * para.me) .- para.μ + fock  # ϵ_HF = ϵ_0 + (Σ_F(k) - δμ₁)
    else
        ϵk = kgrid .^ 2 / (2 * para.me) .- para.μ         # ϵ_0
    end
    bare_occupation_exact = -Spectral.kernelFermiT.(-1e-8, ϵk, para.β)

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
    ct_filename = "examples/counterterms/data/data_Z.jld2"
    z, μ, has_taylor_factors = UEG_MC.load_z_mu_old(para4; ct_filename=ct_filename)
    # Add Taylor factors to CT data
    if has_taylor_factors == false
        for (p, v) in z
            z[p] = v / (factorial(p[2]) * factorial(p[3]))
        end
        for (p, v) in μ
            μ[p] = v / (factorial(p[2]) * factorial(p[3]))
        end
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
    _, δμ, _ = CounterTerm.sigmaCT(max_order - n_min, μ, z; isfock=isFock, verbose=1)
    # δμ[2] = measurement("-0.08196(8)")  # replaces δμ[2] with benchmark dMu2 value
    println("Computed δμ: ", δμ)
    occupation =
        UEG_MC.chemicalpotential_renormalization_green(merged_data, δμ; max_order=max_order)

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
        δμ1_exact = UEG_MC.delta_mu1(para)  # = ReΣ₁[λ](kF, 0)
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

    println(UEG.paraid(para))
    println(partitions)
    println(res)

    if save
        savename =
            "results/data/processed/rs=$(para.rs)/rs=$(para.rs)_beta_ef=$(para.beta)_" *
            "lambda=$(para.mass2)_$(intn_str)$(solver)_$(ct_string)_archive1"
        # "lambda=$(para.mass2)_$(intn_str)$(solver)_$(ct_string)"
        f = jldopen("$savename.jld2", "a+"; compress=true)
        # NOTE: no bare result for c1b observable (accounted for in c1b0)
        for N in min_order_plot:max_order
            if N == 4
                num_eval = neval4
            elseif N == 3
                num_eval = neval3
            else
                num_eval = neval12
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
            f["occupation/N=$N/neval=$num_eval/para"] = para
            f["occupation/N=$N/neval=$num_eval/kgrid"] = kgrid
        end
    end

    # Bare/Fock occupation on dense grid for plotting
    kgrid_fine = para.kF * np.linspace(0.0, 3.0; num=600)
    k_kf_grid_fine = np.linspace(0.0, 3.0; num=600)
    ϵk_fine = @. kgrid_fine^2 / (2 * para.me) - para.μ
    fe_fine = -Spectral.kernelFermiT.(-1e-8, ϵk_fine, para.β)  # f(ϵk)

    if para.isFock
        fock =
            SelfEnergy.Fock0_ZeroTemp.(kgrid_fine, [para.basic]) .-
            SelfEnergy.Fock0_ZeroTemp(para.kF, para.basic)
        ϵk = kgrid_fine .^ 2 / (2 * para.me) .- para.μ + fock  # ϵ_HF = ϵ_0 + (Σ_F(k) - δμ₁)
    else
        ϵk = kgrid_fine .^ 2 / (2 * para.me) .- para.μ         # ϵ_0
    end
    bare_occupation_fine = -Spectral.kernelFermiT.(-1e-8, ϵk, para.β)

    # Plot the occupation number for each aggregate order
    axvline(1.0; linestyle="--", linewidth=1, color="0.7")

    # Shade the thermal broadening region of the Fermi function
    # NOTE: the FWHM of the Fermi distribution is ~ 3.53 kB T
    # fermi_hwhm = 3.53 / (2 * para.β * para.kF)  # HWHM in units of kF
    # fermi_hwhm = 3.53 / (2 * para.beta)  # HWHM in units of kF

    fermi_hwhm = log((sqrt(2) + 1) / (sqrt(2) - 1)) / para.β  # HWHM of -f'(ϵ) = ln((√2 + 1) / (√2 - 1)) / β
    # We obtain k_± via ϵ_± = ϵF ± Δ_HWHM = k^2_± / 2m
    k_minus = para.kF * sqrt(1 - fermi_hwhm / para.EF)  # k₋ = kF √(1 - Δ_HWHM / ϵF)
    k_plus = para.kF * sqrt(1 + fermi_hwhm / para.EF)   # k₊ = kF √(1 + Δ_HWHM / ϵF)
    # TODO: Why is ~4 HWHM necessary here? Check for errors!
    # k_minus = para.kF * sqrt(1 - 4fermi_hwhm / para.EF)  # k₋ = kF √(1 - Δ_HWHM / ϵF)  
    # k_plus = para.kF * sqrt(1 + 4fermi_hwhm / para.EF)   # k₊ = kF √(1 + Δ_HWHM / ϵF)  
    lwindow = para.kF * sqrt(1 - fermi_hwhm / para.EF)
    rwindow = para.kF * sqrt(1 + fermi_hwhm / para.EF)
    axvspan(
        lwindow / para.kF,
        rwindow / para.kF;
        # 1 - fermi_hwhm,
        # 1 + fermi_hwhm;
        color="0.925",
        # label="\$\\mathrm{FWHM}_{k}(f_{k\\sigma})\$",
    )

    # if min_order_plot == 0
    # Include bare occupation fₖ in plot
    # plot(k_kf_grid_fine, bare_occupation_fine, "k"; label="\$N=0\$ (exact)")
    # end

    ikFplus = searchsortedfirst(k_kf_grid, 1.0)  # Index where k ⪆ kF
    ikFminus = ikFplus - 1
    ic = 1
    for (i, N) in enumerate(min_order:max_order_plot)
        # N < max_order_plot && continue
        # N == 0 && continue
        isFock && N == 1 && continue
        # Plot measured data
        means = Measurements.value.(occupation_total[N])
        stdevs = Measurements.uncertainty.(occupation_total[N])

        # qp fit at lowest and highest orders
        if N == max_order_plot
            lwindow = para.kF * sqrt(1 - 4 * fermi_hwhm / para.EF)
            rwindow = para.kF * sqrt(1 + 4 * fermi_hwhm / para.EF)
            # Fitting constant C fixed by value at kF, C ≈ n(kF) - Z/2. 
            # Since k = kF is not on-grid, use the midpoint of the nearest two points
            # (linear interpolation, where the kgrid is assumed symmetrical about k = kF)
            means_kF = (means[ikFplus] + means[ikFminus]) / 2.0  # n(kF) ≈ (n(kF⁻) + n(kF⁺)) / 2
            # ξ⋆(k) & f(ξ⋆(k)) on-grid, ξk = ϵ⋆(k) - μ⋆
            # @. ξstar_k(k, p) = @. (k^2 - para.kF^2) / (2 * p[2])
            # @. fξstar_k(k, p) = @. -Spectral.kernelFermiT.(-1e-8, ξstar_k(k, p), para.β)
            # n_qp(Z, m⋆) = hat(n)(kF) - Z/2 + Zf(k^2 / 2m⋆ - kF^2 / 2m⋆), and p = (Z, m⋆)
            # @. model_Z_mstar(k, p) = means_kF + p[1] * (fξstar_k(k, p) - 1 / 2)
            @. function model_Z_mstar(k, p)
                return means_kF +
                       p[1] * (
                    -Spectral.kernelFermiT.(-1e-8, (k^2 - para.kF^2) / (2 * p[2]), para.β) -
                    1 / 2
                )
            end
            # @. function model_Z_mstar_v2(k, p)
            #     return (means_kF - p[1] / 2) / (1 + 1 / para.β^2) +
            #            p[1] *
            #            -Spectral.kernelFermiT.(
            #         -1e-8,
            #         (k^2 - para.kF^2) / (2 * p[2]),
            #         para.β,
            #     )
            # end
            # Fix m⋆ value to K.H. & K.C. data 
            if rs == 1.0 || rs == 2.0
                if rs == 1.0
                    mstar_khkc = para.me * 0.955
                elseif rs == 2.0
                    mstar_khkc = para.me * 0.943
                end
                @. ξstar_khkc(k) = @. (k^2 - para.kF^2) / (2 * mstar_khkc)
                @. function fξstar_khkc(k)
                    @. -Spectral.kernelFermiT.(-1e-8, ξstar_khkc(k), para.β)
                end
                # @. model_Z(k, p) = means_kF + p[1] * (fξstar_khkc(k) - 1 / 2)
                @. function model_Z(k, p)
                    return means_kF +
                           p[1] * (
                        -Spectral.kernelFermiT.(
                            -1e-8,
                            (k^2 - para.kF^2) / (2 * mstar_khkc),
                            para.β,
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
                # Estimate error for fit para Z
                error_est = estimate_errors(fit_N_khkc)
                println("(N=$N) Estimated error for fit parameter Z:")
                println(error_est)
            end
        end
        _x, _y = spline(k_kf_grid, means, stdevs)
        plot(
            _x,
            _y;
            color=color[ic],
            linestyle=N > 0 ? "--" : "-",
            zorder=10 * ic + 3,
            label=N == 0 ? "\$N = 0\$" : nothing,
        )
        if N > 0
            errorbar(
                k_kf_grid,
                means;
                yerr=stdevs,
                color=color[ic],
                capsize=2,
                markersize=2,
                fmt="o",
                markerfacecolor="none",
                label="\$N = $N\$",
                zorder=10 * ic + 3,
            )
        end

        # plot(
        #     k_kf_grid,
        #     means,
        #     marker;
        #     markersize=2,
        #     # color="C$ic",
        #     color="$(color[ic])",
        #     label="\$N=$N\$ ($solver)",
        #     # label="\$N=$N\$ ($solver, SOSEM)",
        # )
        # fill_between(
        #     k_kf_grid,
        #     means - stdevs,
        #     means + stdevs;
        #     # color="C$ic",
        #     color="$(color[ic])",
        #     alpha=0.4,
        # )

        # # Plot Fqp(Z, m⋆) best fit to fk and Z fk (lowest and max orders)
        # if N == max_order_plot
        #     plot(
        #         k_kf_grid_fine,
        #         # qp_fit_N_khkc(kgrid),
        #         qp_fit_N(kgrid_fine);
        #         # color="C$ic",
        #         # color="$(color[ic])",
        #         color="k",
        #         # label="\$N=$N\$ (LSQ fit to \$F(Z, m^\\star = $mstar_khkc})\$)",
        #         label="\$N=$N\$ (LSQ fit to \$F(Z, m^\\star)\$)",
        #         # label="\$N=$N\$ ($solver, SOSEM)",
        #     )
        #     text(
        #         0.25,
        #         0.8,
        #         # "(\$N=$N\$) \$Z \\approx $(round(zfactor_estimate; digits=4))\$";
        #         "\$Z \\approx $(round(zfactor_estimate; digits=4))\$";
        #         fontsize=14,
        #     )
        #     text(
        #         0.25,
        #         # 0.6 - (N / 30.0),
        #         0.7,
        #         # "(\$N=$N\$) \$m^\\star \\approx $(round(mstar_estimate; digits=4))\$";
        #         "\$m^\\star / m \\approx $(round(mstar_estimate / para.me; digits=4))\$";
        #         fontsize=14,
        #     )
        #     plot(
        #         k_kf_grid_fine,
        #         qp_fit_N_khkc(kgrid_fine);
        #         markersize=2,
        #         color="gray",
        #         label="\$N=$N\$ (LSQ fit to \$F_{KHKC}(Z)\$)",
        #     )
        #     text(
        #         0.25,
        #         0.6,
        #         "\$Z_{KHKC} \\approx $(round(zfactor_estimate_khkc; digits=4))\$";
        #         fontsize=14,
        #     )
        #     text(
        #         0.25,
        #         0.5,
        #         "\$m^\\star_{KHKC} / m = $(round(mstar_khkc / para.me; digits=4))\$";
        #         fontsize=14,
        #     )
        # end

        # Extrapolate Z-factor to kF⁻ & kF⁺ at the maximum order
        # if N == max_order_plot || N == 0
        # # Grid data outside of thermal broadening window
        # k_kf_grid_lower = k_kf_grid[k_kf_grid .≤ k_minus / para.kF]
        # k_kf_grid_upper = k_kf_grid[k_kf_grid .≥ k_plus / para.kF]
        # zfactor_lower = means[k_kf_grid .≤ k_minus / para.kF]
        # zfactor_upper = means[k_kf_grid .≥ k_plus / para.kF]

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
        # plot(k_kf_lower_fine, zfactor_lower_fine; linestyle="-", color="k")
        # plot(
        #     k_kf_upper_fine,
        #     zfactor_upper_fine;
        #     linestyle="-",
        #     color="k",
        #     label="\$N=$N\$ (\$T=0\$ extrapolation)",
        # )
        # text(
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
    legend(; loc="lower left")
    # xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    xlim(0.8, 1.2)
    # xlim(0, 2)
    # ylim(nothing, 2)
    xlabel("\$k / k_F\$")
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
    ylabel("\$$prefix$suffix\$")
    if isFock
        xloc = 1.035
        yloc = 0.6
        ydiv = -0.125
    else
        xloc = 1.035
        yloc = 0.8
        ydiv = -0.125
    end
    text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=12,
    )
    text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
        fontsize=12,
    )
    PyPlot.tight_layout()
    savefig(
        "results/occupation/" *
        "occupation_N=$(max_order_plot)_rs=$(para.rs)_" *
        # "beta_ef=$(para.beta)_lambda=$(para.mass2)_neval=$(neval)_$(solver)_$(ct_string)$(fix_string)_fit.pdf",
        "beta_ef=$(para.beta)_lambda=$(para.mass2)_neval=$(neval)_$(solver)_$(ct_string)$(fix_string).pdf",
    )

    plt.close("all")
    return
end

main()
