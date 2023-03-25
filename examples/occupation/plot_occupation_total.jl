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

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    rs = 5.0
    beta = 40.0
    mass2 = 0.1375
    solver = :vegasmc

    # Number of evals
    # neval = 5e10
    neval12 = 5e10
    neval3 = 5e10
    neval = max(neval12, neval3)

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order = 0
    max_order = 2
    min_order_plot = 0
    max_order_plot = 2

    # Save total results
    save = true

    # Add a zoom-in inset to plot?
    inset = false

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Remove Fock insertions?
    isFock = false

    # Plot benchmark results?
    benchmark = false

    # Compare with results from EFT_UEG?
    compare_eft = false

    # Set green4 to zero for benchmarking?
    no_green4 = false
    no_green4_str = no_green4 ? "_no_green4" : ""

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
    if max_order == 3
        max_together = 2
    else
        max_together = max_order
    end
    savename =
        "results/data/occupation_n=$(max_together)_rs=$(rs)_beta_ef=$(beta)_" *
        # "results/data/occupation_n=$(max_order)_rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(mass2)_neval=$(neval)_$(solver)_$(ct_string)"
    orders, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(loadparam))"
        return f[key]
    end
    if max_order == 3
        # 3rd order 
        savename =
            "results/data/occupation_n=$(max_order)_rs=$(rs)_beta_ef=$(beta)_" *
            "lambda=$(mass2)_neval=$(neval3)_$(solver)_$(ct_string)"
        orders5, param5, kgrid5, partitions5, res5 = jldopen("$savename.jld2", "a+") do f
            key = "$(UEG.short(loadparam))"
            return f[key]
        end
    end

    if compare_eft
        # Load the EFT_UEG data (NOTE: EFT data is already in Dict format!)
        savename = "results/occupation/data_K$(no_green4_str)"
        param_eft, tgrid_eft, kgrid_eft, data_eft = jldopen("$savename.jld2", "a+") do f
            key = "$(UEG.short(loadparam))"
            return f[key]
        end
        println(param_eft)
        println(data_eft)
    end

    # Read in benchmark data
    benchmark_dfs = DataFrame[]
    for order in 2:3
        data, header = readdlm(
            "results/occupation/benchmark/Nk_beta40.0_rs1.0_ms1.0_o$(order).txt",
            ' ';
            header=true,
        )
        push!(benchmark_dfs, DataFrame(data, vec(header)))
    end
    bm_kgrid, bm_occupation_0, bm_occupation_0_err = eachcol(benchmark_dfs[1])
    bm_kgrid, bm_occupation_2, bm_occupation_2_err = eachcol(benchmark_dfs[2])
    bm_k_kf_grid = bm_kgrid / param.kF

    # Get dimensionless k-grid (k / kF) and index corresponding to the Fermi energy
    k_kf_grid = kgrid / param.kF
    if max_order == 3
        k_kf_grid5 = kgrid5 / param.kF
    end

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    for (k, v) in data
        data[k] = v / (factorial(k[2]) * factorial(k[3]))
    end
    # NOTE: Old data for orders 0, 1, and 2 may have Taylor factors already present in eval!
    # for (k, v) in data
    #     data[k] = v / (factorial(k[2]) * factorial(k[3]))
    # end
    # Add 5th order results to data dict
    if max_order == 3
        data3 = UEG_MC.restodict(res5, partitions5)
        for (k, v) in data3
            data3[k] = v / (factorial(k[2]) * factorial(k[3]))
        end
        merge!(data, data3)
    end
    # data_eft = sort(data_eft)

    # Zero out partitions with mu renorm if present (fix mu)
    if renorm_mu == false || fix_mu
        for P in keys(data)
            if P[2] > 0
                println("Fixing mu without lambda renorm, ignoring n_k partition $P")
                data[P] = zero(data[P])
            end
        end
        if compare_eft
            for P in keys(data_eft)
                if P[2] > 0
                    println("Fixing mu without lambda renorm, ignoring n_k partition $P")
                    data_eft[P] = zero(data_eft[P])
                end
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
        if compare_eft
            for P in keys(data_eft)
                if P[3] > 0
                    println("No lambda renorm, ignoring n_k partition $P")
                    data_eft[P] = zero(data_eft[P])
                end
            end
        end
    end

    println(data)
    # TODO: Fix minus sign bug in integration scripts, (-1) → (-1)^(n_g)
    #       Here, we add back in the missing factor (-1)^(n_g - 1) in post-processing,
    #       where n_g = 2 * n_loop + n_μ + 1 for a given partition P = (n_loop, n_μ, n_λ)
    for P in keys(data)
        # Undo extra minus sign factor
        data[P] = -data[P]
        # n_g = 2P[1] + P[2] + 1
        # data[P] = (-1)^(n_g - 1) * data[P]
        # if sum(P) ≤ 1
        #     println(P)
        #     data[P] = -data[P]
        # end
    end
    # data[(0, 1, 0)] = -data[(0, 1, 0)]
    # data[(0, 1, 0)] = -data[(0, 1, 0)]

    # Zero out double-counted (Fock renormalized) partitions
    if isFock && min_order ≤ 1
        data[(1, 0, 0)] = zero(data[(max_order, 0, 0)])
        # data[(0, 1, 0)] = zero(data[(max_order, 0, 0)])  # Combines with dMu2, nonzero!
    end

    merged_data = CounterTerm.mergeInteraction(data)
    if compare_eft
        merged_data_eft = CounterTerm.mergeInteraction(data_eft)
    end

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
    if compare_eft
        if haskey(merged_data_eft, (0, 0)) == false
            # treat quadrature data as numerically exact
            merged_data_eft[(0, 0)] = [measurement(bare_occupation_exact, 0.0)]
        end
    end
    if min_order == 0
        stdscores = stdscore.(merged_data[(0, 0)], bare_occupation_exact)
        worst_score = argmax(abs, stdscores)
        println("Worst standard score for N=0 (measured): $worst_score")
        # println("Scores:\n$stdscores")
        if isFock
            bm_occupation_2_meas = measurement.(bm_occupation_0, bm_occupation_0_err)
            stdscores = stdscore.(bm_occupation_2_meas, bare_occupation_exact)
            worst_score = argmax(abs, stdscores)
            println("Worst standard score for N=0 (benchmark): $worst_score")
            # println("Scores:\n$stdscores")
            @assert worst_score ≤ 10
        end
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

    if compare_eft
        for (k, v) in merged_data_eft
            println(v)
            merged_data_eft[k] = vec(v)
        end
    end

    println("Computed δμ: ", δμ)
    # δμ[2] = measurement("-0.08196(8)")  # Use benchmark dMu2 value
    occupation = UEG_MC.chemicalpotential_renormalization_green(
        merged_data,
        δμ;
        min_order=0,
        max_order=max_order,
    )
    if compare_eft
        occupation_eft = UEG_MC.chemicalpotential_renormalization_green(
            merged_data_eft,
            δμ;
            min_order=0,
            max_order=max_order,
        )
    end
    # occupation = UEG_MC.RenormMeasType()
    # occupation[0] = merged_data[(0, 0)]
    # occupation[2] = merged_data[(2, 0)] + δμ[2] * merged_data[(0, 1)]
    # occupation[3] = zero(occupation[2])
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
    if compare_eft
        occupation_total_eft = UEG_MC.aggregate_orders(occupation_eft)
    end

    # occupation = Dict()
    # for (k, v) in merged_data
    #     if k[2] > 0
    #         println("Skipping partition $k with mu CTs")
    #         continue
    #     end
    #     println("md key: $k => $(sum(k))")
    #     occupation[sum(k)] = v
    # end
    # println(collect(keys(occupation)))

    # occupation_total = Dict()
    # occupation_total[0] = occupation[0]
    # occupation_total[1] = occupation_total[0] + occupation[1]
    # occupation_total[2] = occupation_total[1] + occupation[2]
    # occupation_total[3] = occupation_total[2] + occupation[3]
    # println(collect(keys(occupation_total)))

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
            num_eval = N == 3 ? neval3 : neval12
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

    # Bare occupation fₖ on dense grid for plotting
    kgrid_fine = param.kF * LinRange(0.75, 1.25, 500)
    k_kf_grid_fine = LinRange(0.75, 1.25, 500)
    ϵk_fine = @. kgrid_fine^2 / (2 * param.me) - param.μ
    fe_fine = -Spectral.kernelFermiT.(-1e-8, ϵk_fine, param.β)
    # First derivative of the Fermi function f'(ϵₖ)
    fpe_fine = -Spectral.kernelFermiT_dω.(-1e-8, ϵk_fine, param.β)
    fpe_fine_exact = @. param.β * fe_fine * (fe_fine - 1)
    fppe_fine_exact =
        @. (param.β^2 / 4) * sech(param.β * ϵk_fine / 2)^2 * tanh.(param.β * ϵk_fine / 2)

    # Fermi function
    fe_fine = -Spectral.kernelFermiT.(-1e-8, ϵk_fine, param.β)
    # First derivative of the Fermi function f'(ϵₖ)
    fpe_fine = -Spectral.kernelFermiT_dω.(-1e-8, ϵk_fine, param.β)
    # Second derivative of the Fermi function f''(ϵₖ)
    fppe_fine = -Spectral.kernelFermiT_dω2.(-1e-8, ϵk_fine, param.β)
    # Third derivative of the Fermi function f'''(ϵₖ)
    fpppe_fine = -Spectral.kernelFermiT_dω3.(-1e-8, ϵk_fine, param.β)

    # g₁(k, 0⁻) (= -δn₁(kσ)) = f'(ξₖ) (Σ^λ_F(k_F) - Σ^λ_F(k))
    first_order_exact = @. fpe_fine * (
        SelfEnergy.Fock0_ZeroTemp(param.kF, [param.basic]) -
        SelfEnergy.Fock0_ZeroTemp(kgrid_fine, [param.basic])
    )

    # (1, 0, 0) = -f'(ξₖ) Σ^λ_F(k)
    exact_gsigmag = @. -fpe_fine * SelfEnergy.Fock0_ZeroTemp(kgrid_fine, [param.basic])

    if max_order_plot == 1
        # Test gg diagram (0, 1, 0)
        fig, ax = plt.subplots()
        # ax.axhline(0.0; linewidth=1, color="k")
        ax.plot(k_kf_grid_fine, fpe_fine, "-"; color="k", label="\$f'(\\xi_k)\$")
        means = Measurements.value.(data[(0, 1, 0)])
        stdevs = Measurements.uncertainty.(data[(0, 1, 0)])
        ax.plot(
            k_kf_grid,
            means,
            "o-";
            markersize=2,
            color="C0",
            label="\$(0, 1, 0)\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid,
            (means - stdevs),
            (means + stdevs);
            color="C0",
            alpha=0.4,
        )
        ax.set_xlabel("\$k / k_F\$")
        ax.set_xlim(0.75, 1.25)
        # ax.set_ylim(nothing, 4)
        ax.legend(; loc="best")
        fig.tight_layout()
        fig.savefig("results/occupation/minus_nk_gg.pdf")

        # Test g(-Σ)g diagram (1, 0, 0)
        fig, ax = plt.subplots()
        # ax.axhline(0.0; linewidth=1, color="k")
        fpnes = [fpe_fine, fppe_fine, fpppe_fine]
        ax.plot(
            k_kf_grid_fine,
            exact_gsigmag,
            "-";
            color="k",
            label="\$-f'(\\xi_k) \\Sigma^\\lambda_F(k)\$",
        )
        means = Measurements.value.(data[(1, 0, 0)])
        stdevs = Measurements.uncertainty.(data[(1, 0, 0)])
        ax.plot(
            k_kf_grid,
            means,
            "o-";
            markersize=2,
            color="C0",
            label="\$(1, 0, 0)\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid,
            (means - stdevs),
            (means + stdevs);
            color="C0",
            alpha=0.4,
        )
        ax.set_xlabel("\$k / k_F\$")
        ax.set_xlim(0.75, 1.25)
        # ax.set_ylim(nothing, 4)
        ax.legend(; loc="best")
        fig.tight_layout()
        fig.savefig("results/occupation/minus_nk_gsigmag.pdf")

        # Test first order result
        fig, ax = plt.subplots()
        ax.plot(
            k_kf_grid_fine,
            first_order_exact,
            "-";
            color="C0",
            label="\$f'(\\xi_k) (\\Sigma^\\lambda_F(k_F) - \\Sigma^\\lambda_F(k))\$",
        )
        means = Measurements.value.(occupation[1])
        stdevs = Measurements.uncertainty.(occupation[1])
        ax.plot(k_kf_grid, means, "o-"; markersize=2, color="C1", label="\$N=1\$ ($solver)")
        ax.fill_between(
            k_kf_grid,
            (means - stdevs),
            (means + stdevs);
            color="C1",
            alpha=0.4,
        )
        ax.set_ylabel("\$g_1(k, \\tau=0^-) = -\\delta n^1_{k\\sigma}\$")
        ax.set_xlabel("\$k / k_F\$")
        ax.set_xlim(0.75, 1.25)
        # ax.set_ylim(nothing, 4)
        ax.legend(; loc="best")
        fig.tight_layout()
        fig.savefig("results/occupation/minus_delta_nk1.pdf")
    end

    # fig, ax = plt.subplots()
    # # ax.axhline(0.0; linewidth=1, color="k")
    # fpnes = [fpe_fine, fppe_fine, fpppe_fine]
    # colors = ["sienna", "darkgreen", "darkred"]
    # signs = [-1.0, 1.0, -1.0]
    # signstrs = ["(-1) \\times", "", "(-1) \\times"]
    # for N in 1:3
    #     ax.plot(
    #         k_kf_grid_fine,
    #         fpnes[N],
    #         "-";
    #         color="$(colors[N])",
    #         label="\$f$(repeat("'", N))(\\xi_k)\$",
    #     )
    #     # if N == 1
    #     #     ax.plot(
    #     #         k_kf_grid_fine,
    #     #         fpe_fine_exact,
    #     #         "--";
    #     #         color="k",
    #     #         label="\$f'(\\xi_k)\$ (exact)",
    #     #     )
    #     # elseif N == 2
    #     #     ax.plot(
    #     #         k_kf_grid_fine,
    #     #         fppe_fine_exact,
    #     #         "--";
    #     #         color="gray",
    #     #         label="\$f''(\\xi_k)\$ (exact)",
    #     #     )
    #     # end
    #     means = Measurements.value.(data[(0, N, 0)])
    #     stdevs = Measurements.uncertainty.(data[(0, N, 0)])
    #     ax.plot(
    #         k_kf_grid,
    #         signs[N] * means,
    #         "o-";
    #         markersize=2,
    #         color="C$N",
    #         label="\$$(signstrs[N])(0, $N, 0)\$ ($solver)",
    #     )
    #     ax.fill_between(
    #         k_kf_grid,
    #         signs[N] * (means - stdevs),
    #         signs[N] * (means + stdevs);
    #         color="C$N",
    #         alpha=0.4,
    #     )
    # end
    # ax.set_xlabel("\$k / k_F\$")
    # ax.set_xlim(0.75, 1.25)
    # # ax.set_ylim(nothing, 4)
    # ax.legend(; loc="best")
    # fig.tight_layout()
    # fig.savefig("results/occupation/green_derivatives.pdf")

    # Bare/Fock occupation on dense grid for plotting
    kgrid_fine = param.kF * np.linspace(0.0, 3.0; num=600)
    k_kf_grid_fine = np.linspace(0.0, 3.0; num=600)
    if param.isFock
        fock =
            SelfEnergy.Fock0_ZeroTemp.(kgrid_fine, [param.basic]) .-
            SelfEnergy.Fock0_ZeroTemp(param.kF, param.basic)
        ϵk = kgrid_fine .^ 2 / (2 * param.me) .- param.μ + fock  # ϵ_HF = ϵ_0 + (Σ_F(k) - δμ₁)
    else
        ϵk = kgrid_fine .^ 2 / (2 * param.me) .- param.μ         # ϵ_0
    end
    bare_occupation_fine = -Spectral.kernelFermiT.(-1e-8, ϵk, param.β)

    if benchmark && isFock
        # Get standard scores vs benchmark
        stdscores = stdscore.(occupation_total[2], bm_occupation_2)
        worst_score = argmax(abs, stdscores)
        println(stdscores)
        println(
            "Worst standard score for N=2 with Fock renorm " *
            "(measured vs benchmark mean): $worst_score",
        )
    end

    # Plot the occupation number for each aggregate order
    fig, ax = plt.subplots()
    ax.axvline(1.0; linestyle="--", linewidth=1, color="gray")
    if min_order_plot == 0
        # Include bare occupation fₖ in plot
        ax.plot(k_kf_grid_fine, bare_occupation_fine, "k"; label="\$N=0\$ (exact)")
    end
    ic = 1
    colors = ["C0", "C1", "C2", "C3"]
    # colors = ["orchid", "cornflowerblue", "turquoise", "chartreuse"]
    colors2 = ["purple", "blue", "green", "yellow"]
    for (i, N) in enumerate(min_order:max_order_plot)
        # N == 0 && continue
        isFock && N == 1 && continue
        # Plot benchmark data
        if benchmark && isFock && max_order_plot ≥ 0 && i == 1
            ax.plot(
                bm_k_kf_grid,
                bm_occupation_0,
                "--";
                markersize=2,
                color="C$ic",
                label="\$N=0\$ (benchmark)",
            )
            ax.fill_between(
                bm_k_kf_grid,
                bm_occupation_0 - bm_occupation_0_err,
                bm_occupation_0 + bm_occupation_0_err;
                color="C$ic",
                alpha=0.4,
            )
            ic += 1
        end
        if benchmark && isFock && max_order_plot ≥ 2 && i == 3
            ax.plot(
                bm_k_kf_grid,
                bm_occupation_2,
                "--";
                markersize=2,
                color="C$ic",
                label="\$N=2\$ (benchmark)",
            )
            ax.fill_between(
                bm_k_kf_grid,
                bm_occupation_2 - bm_occupation_2_err,
                bm_occupation_2 + bm_occupation_2_err;
                color="C$ic",
                alpha=0.4,
            )
            ic += 1
        end
        # Plot measured data
        means = Measurements.value.(occupation_total[N])
        stdevs = Measurements.uncertainty.(occupation_total[N])
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
        if compare_eft
            means = Measurements.value.(occupation_total_eft[N])
            stdevs = Measurements.uncertainty.(occupation_total_eft[N])
            marker = "o-"
            ax.plot(
                kgrid_eft / param.kF,
                means,
                marker;
                markersize=2,
                # color="C$ic",
                color="$(colors2[ic])",
                label="\$N=$N\$ ($solver, ElectronLiquid)",
            )
            ax.fill_between(
                kgrid_eft / param.kF,
                means - stdevs,
                means + stdevs;
                # color="C$ic",
                color="$(colors2[ic])",
                alpha=0.4,
            )
        end
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
        xloc = 1.05
        yloc = 0.5
        ydiv = -0.15
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
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    fig.tight_layout()
    if inset
        # Reset color index
        ic = 1
        # Plot inset
        ax_inset =
            il.inset_axes(ax; width="38%", height="28.5%", loc="lower left", borderpad=0)
        # Compare result to bare occupation fₖ
        ax_inset.plot(k_kf_grid_fine, bare_occupation_fine, "k"; label="\$N=0\$ (exact)")
        ax_inset.axvline(1.0; linestyle="--", linewidth=1, color="gray")
        # ax_inset.axhspan(0, 1; alpha=0.2, facecolor="k", edgecolor=nothing)
        ax_inset.axhline(0.0; linestyle="-", linewidth=0.5, color="k")
        ax_inset.axhline(1.0; linestyle="-", linewidth=0.5, color="k")
        for (i, N) in enumerate(min_order:max_order_plot)
            # for (i, N) in enumerate(min_order:1)
            # N == 0 && continue
            isFock && N == 1 && continue
            # Get means and error bars from the result up to this order
            means = Measurements.value.(occupation_total[N])
            stdevs = Measurements.uncertainty.(occupation_total[N])
            marker = "o-"
            ax_inset.plot(
                k_kf_grid,
                means,
                marker;
                markersize=2,
                color="$(colors[ic])",
                label="\$N=$N\$ ($solver)",
            )
            ax_inset.fill_between(
                k_kf_grid,
                means - stdevs,
                means + stdevs;
                color="$(colors[ic])",
                alpha=0.4,
            )
            ic += 1
        end
        xpad = 0.04
        ypad = 0.2
        ax_inset.set_xlim(0.95 - xpad, 1.05 + xpad)
        ax_inset.set_ylim(-ypad, 1 + ypad)
        ax_inset.set_xlabel("\$k / k_F\$"; labelpad=7)
        ax_inset.set_ylabel("\$n_{k\\sigma}\$")
        ax_inset.xaxis.set_label_position("top")
        ax_inset.yaxis.set_label_position("right")
        ax_inset.xaxis.tick_top()
        ax_inset.yaxis.tick_right()
    end
    dirstring = isFock ? "results/occupation/benchmark/" : "results/occupation/"
    fig.savefig(
        dirstring *
        "occupation_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_neval=$(neval)_$(solver)_$(ct_string)$(fix_string).pdf",
    )

    plt.close("all")
    return
end

main()
