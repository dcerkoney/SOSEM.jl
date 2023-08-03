using CodecZlib
using ElectronLiquid
using ElectronGas
using FeynmanDiagram
using Interpolations
using JLD2
using MCIntegration
using Measurements
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

# We plot the k=0 value
const extK_load = 0.0

# NOTE: Call from main project directory as: julia examples/c1b/plot_c1b_total.jl

# Converts JLD2 data from old to new ParaMC format on load by adding the `initialized` field (see: https://juliaio.github.io/JLD2.jl/stable/advanced/)
# NOTE: Requires the type name `ElectronLiquid.UEG.ParaMC` to be explicitly specified
function JLD2.rconvert(::Type{ElectronLiquid.UEG.ParaMC}, nt::NamedTuple)
    return ElectronLiquid.UEG.ParaMC(; nt..., initialized=false)
end

function load_old_data(filename)
    # Upgrade objects with breaking changes
    typemap = Dict("ElectronLiquid.UEG.ParaMC" => JLD2.Upgrade(ElectronLiquid.UEG.ParaMC))
    return load(filename; typemap=typemap)
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
    mass2 = 1.0
    # rs = 2.0
    # mass2 = 0.4
    beta = 40.0
    solver = :vegasmc
    # expand_bare_interactions = 1          # single V[V_λ] scheme
    # expand_bare_interactions = 0          # bare V, V (non-reexpanded) scheme
    # intn_schemes = [1]  # compare schemes
    # intn_scheme_strings = ["\$V, V[V_\\lambda]\$"]
    intn_schemes = [0, 1]  # compare schemes
    intn_scheme_strings = ["\$V, V\$", "\$V, V[V_\\lambda]\$"]

    # neval = 1e10
    neval = 1e9

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    n_min = 3  # True minimal loop order for this observable
    min_order = 3
    # max_order = 5
    max_order = 6
    min_order_plot = 2
    max_order_plot = 5
    # max_order_plot = 6
    @assert max_order ≥ 3

    # Load data from multiple fixed-order runs
    # fixed_orders = collect(min_order:max_order)

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Save total results?
    save = false
    # save = true

    # Include RPA(+FL) results?
    plot_rpa_fl = true

    param = UEG.ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    if plot_rpa_fl
        # Non-dimensionalize bare and RPA+FL non-local moments
        rs_lo = rs
        sosem_lo = np.load("results/data/python/soms_rs=$(rs_lo)_beta_ef=40.0.npz")
        # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
        param_lo = Parameter.atomicUnit(0, rs_lo)    # (dimensionless T, rs)
        eTF_lo = param_lo.qTF^2 / (2 * param_lo.me)

        # # Interpolate bare results and downsample to coarse k_kf_grid_vegas
        k_kf_grid_vegas = np.load("results/kgrids/kgrid_vegas_dimless_n=77_small.npy")
        @assert k_kf_grid_vegas[1] == 0.0

        # RPA(+FL) corrections to LO for class (b) moment at k=0
        delta_c1b_rpa = sosem_lo.get("delta_rpa_b_vegas_N=1e+07.npy")[1] / eTF_lo^2
        delta_c1b_rpa_err = sosem_lo.get("delta_rpa_b_err_vegas_N=1e+07.npy")[1] / eTF_lo^2
        delta_c1b_rpa_fl = sosem_lo.get("delta_rpa+fl_b_vegas_N=1e+07.npy")[1] / eTF_lo^2
        delta_c1b_rpa_fl_err =
            sosem_lo.get("delta_rpa+fl_b_err_vegas_N=1e+07.npy")[1] / eTF_lo^2
    end

    # Plot results vs order N
    orders = min_order:max_order_plot
    plot_orders = min_order_plot:max_order_plot
    # Compare with RPA & RPA+FL results
    if plot_rpa_fl && min_order_plot == 2
        # Non-dimensionalize bare and RPA+FL non-local moments
        rs_lo = rs
        sosem_lo = np.load("results/data/python/soms_rs=$(rs_lo)_beta_ef=40.0.npz")
        # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
        param_lo = Parameter.atomicUnit(0, rs_lo)    # (dimensionless T, rs)
        eTF_lo = param_lo.qTF^2 / (2 * param_lo.me)

        # Bare and RPA(+FL) results (stored in Hartree a.u.)
        k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
        c1nl_lo =
            (sosem_lo.get("bare_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d")) /
            eTF_lo^2
        c1nl_rpa =
            (sosem_lo.get("rpa_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d")) /
            eTF_lo^2
        c1nl_rpa_fl =
            (sosem_lo.get("rpa+fl_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d")) /
            eTF_lo^2
        # RPA(+FL) means are error bars
        c1nl_rpa_means, c1nl_rpa_stdevs =
            Measurements.value.(c1nl_rpa), Measurements.uncertainty.(c1nl_rpa)
        c1nl_rpa_fl_means, c1nl_rpa_fl_stdevs =
            Measurements.value.(c1nl_rpa_fl), Measurements.uncertainty.(c1nl_rpa_fl)

        c1nl_rpa_mean = c1nl_rpa_means[1]
        c1nl_rpa_fl_mean = c1nl_rpa_fl_means[1]
        c1nl_rpa_err = c1nl_rpa_stdevs[1]
        c1nl_rpa_fl_err = c1nl_rpa_fl_stdevs[1]
        ex_plot_orders = [1.8, 5.2]
        plot(ex_plot_orders, -0.5 * one.(ex_plot_orders), "k"; linestyle="-.", label="LO")
        plot(
            ex_plot_orders,
            c1nl_rpa_mean * one.(ex_plot_orders),
            "k";
            linestyle="--",
            label="RPA",
        )
        # fill_between(
        #     orders,
        #     (c1nl_rpa_mean - c1nl_rpa_err) * one.(orders),
        #     (c1nl_rpa_mean + c1nl_rpa_err) * one.(orders);
        #     color="k",
        #     alpha=0.3,
        # )
        plot(ex_plot_orders, c1nl_rpa_fl_mean * one.(ex_plot_orders), "k"; label="RPA\$+\$FL")
        # fill_between(
        #     orders,
        #     (c1nl_rpa_fl_mean - c1nl_rpa_fl_err) * one.(orders),
        #     (c1nl_rpa_fl_mean + c1nl_rpa_fl_err) * one.(orders);
        #     color="k",
        #     alpha=0.3,
        # )
        # plot(orders, delta_c1b_rpa * one.(orders), "k"; linestyle="--", label="RPA")
        # fill_between(
        #     orders,
        #     (delta_c1b_rpa - delta_c1b_rpa_err) * one.(orders),
        #     (delta_c1b_rpa + delta_c1b_rpa_err) * one.(orders);
        #     color="k",
        #     alpha=0.3,
        # )
        # plot(orders, delta_c1b_rpa_fl * one.(orders), "k"; label="RPA\$+\$FL")
        # fill_between(
        #     orders,
        #     (delta_c1b_rpa_fl - delta_c1b_rpa_fl_err) * one.(orders),
        #     (delta_c1b_rpa_fl + delta_c1b_rpa_fl_err) * one.(orders);
        #     color="k",
        #     alpha=0.3,
        # )
    end
    for (expand_bare_interactions, intn_scheme_str) in
        zip(intn_schemes, intn_scheme_strings)

        # Distinguish results with fixed vs re-expanded bare interactions
        intn_str = ""
        if expand_bare_interactions == 2
            intn_str = "no_bare_"
        elseif expand_bare_interactions == 1
            intn_str = "one_bare_"
        end

        # Use LaTex fonts for plots
        plt.rc("text"; usetex=true)
        plt.rc("font"; family="serif")

        # Load the order 3-4 results from JLD2 (and μ data from csv, if applicable)
        # if max_order == 5
        #     max_together = 4
        # else
        #     max_together = max_order
        # end
        local htf
        param =
            UEG.ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)
        savename =
            "results/data/c1bL/c1bL_k=$(extK_load)_n=$(max_order)_rs=$(rs)_" *
            "beta_ef=$(beta)_lambda=$(mass2)_" *
            "neval=$(neval)_$(intn_str)$(solver)$(ct_string)_new"
        # "neval=$(neval)_$(intn_str)$(solver)$(ct_string)"
        print("Loading data from $savename.jld2...")
        settings, extK, partitions, data = jldopen("$savename.jld2", "a+") do f
            htf = f["has_taylor_factors"]
            key = "$(UEG.short(param))"
            return f[key]
        end
        @assert extK == 0.0
        println("done!")

        println()
        println("$expand_bare_interactions, $intn_str")
        println(data)
        println()

        # # Load the fixed order 5 result from JLD2
        # local kgrid5, res5, partitions5
        # if max_order == 5
        #     savename5 =
        #         "results/data/c1bL/c1bL_n=$(max_order)_rs=$(rs)_" *
        #         "beta_ef=$(beta)_lambda=$(mass2)_" *
        #         "neval=$(neval5)_$(intn_str)$(solver)$(ct_string)"
        #     settings5, param5, kgrid5, partitions5, res5 = jldopen("$savename5.jld2", "a+") do f
        #         key = "$(UEG.short(param))"
        #         return f[key]
        #     end
        # end

        # Convert results to a Dict of measurements at each order with interaction counterterms merged
        # data = UEG_MC.restodict(res, partitions)
        if htf == false
            for (k, v) in data
                data[k] = v / (factorial(k[2]) * factorial(k[3]))
            end
        end
        # # Add 5th order results to data dict
        # if max_order == 5
        #     data5 = UEG_MC.restodict(res5, partitions5)
        #     for (k, v) in data5
        #         data5[k] = v / (factorial(k[2]) * factorial(k[3]))
        #     end
        #     merge!(data, data5)
        # end
        merged_data = CounterTerm.mergeInteraction(data)
        println([k for (k, _) in merged_data])

        # # Reexpand merged data in powers of μ
        # ct_filename = "examples/counterterms/data/data_Z$(ct_string).jld2"
        # z, μ = UEG_MC.load_z_mu(param; ct_filename=ct_filename)
        # # Add Taylor factors to CT data
        # for (p, v) in z
        #     z[p] = v / (factorial(p[2]) * factorial(p[3]))
        # end
        # for (p, v) in μ
        #     μ[p] = v / (factorial(p[2]) * factorial(p[3]))
        # end
        # δz, δμ = CounterTerm.sigmaCT(max_order - n_min, μ, z; verbose=1)

        # Reexpand merged data in powers of μ
        # δμ = load_mu_counterterm(
        #     param;
        #     max_order=max_order - n_min,
        #     parafilename="examples/counterterms/data/para.csv",
        #     ct_filename="examples/counterterms/data/data_Z$(ct_string).jld2",
        #     verbose=1,
        # )

        # Reexpand merged data in powers of μ
        ct_filename = "examples/counterterms/data/data_Z.jld2"
        zpara = param
        zpara.order = 4
        z, μ, has_taylor_factors = UEG_MC.load_z_mu_old(zpara; ct_filename=ct_filename)
        # Add Taylor factors to CT data
        if has_taylor_factors == false
            for (p, v) in z
                z[p] = v / (factorial(p[2]) * factorial(p[3]))
            end
            for (p, v) in μ
                μ[p] = v / (factorial(p[2]) * factorial(p[3]))
            end
        end
        _, δμ, _ = CounterTerm.sigmaCT(max_order - n_min, μ, z; verbose=1)
        println("Computed δμ: ", δμ)

        println(merged_data)
        println(δμ)
        c1bL = UEG_MC.chemicalpotential_renormalization_sosem(
            merged_data,
            δμ;
            lowest_order=n_min,  # there is no second order for this observable
            min_order=min_order,
            max_order=max(max_order, max_order_plot),
        )
        # Test manual renormalization with exact lowest-order chemical potential
        if max_order >= 4
            # NOTE: For this observable, there is no second-order
            δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
            # C⁽¹ᵇ⁾₄ = 2(C⁽¹ᵇ⁾ᴸ_{4,0} + δμ₁ C⁽¹ᵇ⁾ᴸ_{3,1})
            c1b4_manual =
                2 * (
                    merged_data[(3, 0)] +
                    merged_data[(4, 0)] +
                    δμ1_exact * merged_data[(3, 1)]
                )
            c1b4L = 2 * (c1bL[3] + c1bL[4])
            stdscores = stdscore.(c1b4L, c1b4_manual)
            worst_score = argmax(abs, stdscores)
            println("Exact δμ₁: ", δμ1_exact)
            println("Computed δμ₁: ", δμ[1])
            println(
                "Worst standard score for total result to 4th " *
                "order (auto vs exact+manual): $worst_score",
            )
        end

        # Aggregate the full results for C⁽¹ᶜ⁾ up to order N
        c1bL_total = UEG_MC.aggregate_orders(c1bL)

        # partitions = collect(Iterators.flatten(partitions_list))

        println(settings)
        println(UEG.paraid(param))
        println(partitions)
        println(data)
        # println(res_list)
        # println(partitions_list)

        if save
            savename =
                "results/data/processed/rs=$(param.rs)/rs=$(param.rs)_beta_ef=$(param.beta)_" *
                "lambda=$(param.mass2)_$(solver)$(ct_string)_archive1_processed_data"
            # "lambda=$(param.mass2)_$(intn_str)$(solver)$(ct_string)_archive1_processed_data"
            # "lambda=$(param.mass2)_$(intn_str)$(solver)$(ct_string)"
            f = jldopen("$savename.jld2", "a+"; compress=true)
            # NOTE: no bare result for c1b observable (accounted for in c1b0)
            for N in min_order_plot:max_order
                # Add RPA & RPA+FL results to data group
                if N == 2
                    if plot_rpa_fl
                        if haskey(f, "c1b_k=0")
                            if haskey(f["c1b_k=0"], "RPA") &&
                               haskey(f["c1b_k=0/RPA"], "neval=$(1e7)")
                                @warn("replacing existing data for RPA, neval=$(1e7)")
                                delete!(f["c1b_k=0/RPA"], "neval=$(1e7)")
                            end
                            if haskey(f["c1b_k=0"], "RPA+FL") &&
                               haskey(f["c1b_k=0/RPA+FL"], "neval=$(1e7)")
                                @warn("replacing existing data for RPA+FL, neval=$(1e7)")
                                delete!(f["c1b_k=0/RPA+FL"], "neval=$(1e7)")
                            end
                        end
                        # RPA
                        meas_rpa = measurement.(delta_c1b_rpa, delta_c1b_rpa_err)
                        # meas_rpa = measurement.(c1b_rpa, c1b_rpa_err)
                        f["c1b_k=0/RPA/neval=$(1e7)/meas"] = meas_rpa
                        f["c1b_k=0/RPA/neval=$(1e7)/param"] = param
                        # RPA+FL
                        meas_rpa_fl = measurement.(delta_c1b_rpa_fl, delta_c1b_rpa_fl_err)
                        # meas_rpa_fl = measurement.(c1b_rpa_fl, c1b_rpa_fl_err)
                        f["c1b_k=0/RPA+FL/neval=$(1e7)/meas"] = meas_rpa_fl
                        f["c1b_k=0/RPA+FL/neval=$(1e7)/param"] = param
                    end
                else
                    # num_eval = N == 5 ? neval5 : neval34
                    num_eval = neval
                    if haskey(f, "c1b_k=0") &&
                       haskey(f["c1b_k=0"], "N=$N") &&
                       haskey(f["c1b_k=0/N=$N"], "neval=$num_eval")
                        @warn("replacing existing data for N=$N, neval=$num_eval")
                        delete!(f["c1b_k=0/N=$N"], "neval=$num_eval")
                    end
                    # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
                    #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
                    f["c1b_k=0/N=$N/neval=$num_eval/meas"] = 2 * c1bL_total[N]
                    f["c1b_k=0/N=$N/neval=$num_eval/settings"] = settings
                    f["c1b_k=0/N=$N/neval=$num_eval/param"] = param
                end
            end
        end

        # Get means and error bars from the results vs order
        # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
        #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
        means =
            [2 * Measurements.value.(c1bL_total[N][1]) for N in min_order:max_order_plot]
        stdevs = [
            2 * Measurements.uncertainty.(c1bL_total[N][1]) for
            N in min_order:max_order_plot
        ]
        c1b_k0 = [2 * c1bL_total[N][1] for N in min_order:max_order_plot]

        # Check convergence
        if expand_bare_interactions == 1
            m_prev = means[1]
            println("Percent difference btw. successive orders:")
            for (N, m) in enumerate(means[2:end])
                this_order = N + min_order
                prev_order = this_order - 1
                println(
                    "N=$(prev_order),$(this_order):\t",
                    100 * abs((m - m_prev) / m_prev),
                )
                m_prev = m
            end
        end

        # N = 3, 4
        neval34 = 5e10
        # N = 5
        neval5_c1b0 = 5e10
        neval5_c1b = 5e10
        neval5_c1c = 5e10
        neval5_c1d = 5e10
        # Filename for new JLD2 format
        filename =
            "results/data/processed/rs=$(rs)/rs=$(rs)_beta_ef=$(beta)_" *
            "lambda=$(mass2)_$(solver)_with_ct_mu_lambda_archive1_processed_data"
        # "lambda=$(mass2)_$(intn_str)$(solver)_with_ct_mu_lambda_archive1_processed_data"
        c1nl_N_totals = []
        c1nl_N_means = []
        c1nl_N_stdevs = []
        for (i, N) in enumerate(plot_orders)
            # Load the data for each observable
            d = load_old_data("$filename.jld2")
            local c1nl_N_total
            if N == 5
                param = d["c1d/N=5/neval=$neval5_c1d/param"]
                kgrid = d["c1d/N=5/neval=$neval5_c1d/kgrid"]
                c1nl_N_total =
                    c1b_k0[i - 1] +
                    d["c1b0/N=5/neval=$neval5_c1b0/meas"][1] +
                    d["c1c/N=5/neval=$neval5_c1c/meas"][1] +
                    d["c1d/N=5/neval=$neval5_c1d/meas"][1]
            elseif N == 2
                param = d["c1d/N=$N/neval=$neval34/param"]
                kgrid = d["c1d/N=$N/neval=$neval34/kgrid"]
                c1nl_N_total =
                    d["c1b0/N=$N/neval=$neval34/meas"][1] +
                    d["c1c/N=$N/neval=$neval34/meas"][1] +
                    d["c1d/N=$N/neval=$neval34/meas"][1]
            else
                param = d["c1d/N=$N/neval=$neval34/param"]
                kgrid = d["c1d/N=$N/neval=$neval34/kgrid"]
                c1nl_N_total =
                    c1b_k0[i - 1] +
                    d["c1b0/N=$N/neval=$neval34/meas"][1] +
                    d["c1c/N=$N/neval=$neval34/meas"][1] +
                    d["c1d/N=$N/neval=$neval34/meas"][1]
            end
            # Get means and error bars from the result up to this order
            push!(c1nl_N_totals, c1nl_N_total)
            push!(c1nl_N_means, c1nl_N_total.val)
            push!(c1nl_N_stdevs, c1nl_N_total.err)
        end

        println("\nTotal non-local moment ($intn_scheme_str) vs order N:\n")
        for (i, N) in enumerate(plot_orders)
            println(" • N = $N:\t$(c1nl_N_totals[i])")
        end
        println(c1nl_N_totals)

        # Plot c1nl(k=0) vs N
        # Data gets noisy above 3rd loop order
        marker = "o-"
        ic = expand_bare_interactions  # 0, 1
        errorbar(
            plot_orders,
            c1nl_N_means;
            yerr=c1nl_N_stdevs,
            color=color[ic + 1],
            capsize=2,
            markersize=2,
            fmt="o-",
            markerfacecolor="none",
            label="$(intn_scheme_str)",
        )
        # plot(
        #     plot_orders,
        #     c1nl_N_means,
        #     marker;
        #     markersize=4,
        #     color=color[ic+1],
        #     label="$(intn_scheme_str)",
        #     # label="$(intn_scheme_str) ($solver)",
        # )
        # # plot(orders, means, marker; markersize=4, color=color[ic+1], label="RPT ($solver)")
        # fill_between(
        #     plot_orders,
        #     c1nl_N_means - c1nl_N_stdevs,
        #     c1nl_N_means + c1nl_N_stdevs;
        #     color=color[ic+1],
        #     alpha=0.4,
        # )

        #  # Plot c1b(k=0) vs N
        # # Data gets noisy above 3rd loop order
        # marker = "o-"
        # ic = expand_bare_interactions  # 0, 1
        # plot(
        #     orders,
        #     means,
        #     marker;
        #     markersize=4,
        #     color=color[ic+1],
        #     label="$(intn_scheme_str) ($solver)",
        # )
        # # plot(orders, means, marker; markersize=4, color=color[ic+1], label="RPT ($solver)")
        # fill_between(orders, means - stdevs, means + stdevs; color=color[ic+1], alpha=0.4)
    end
    # legend(; loc="best", ncol=2)
    legend(; loc=(0.24, 0.61), ncol=2)
    xticks(plot_orders)
    xlim(minimum(ex_plot_orders), maximum(ex_plot_orders))
    # ylim(-1.1, nothing)
    # ylim(nothing, 0.0)
    # legend(; loc="best", framealpha=1.0)
    # xticks(orders)
    # xlim(minimum(orders), maximum(orders))
    xlabel("Perturbation order \$N\$")
    ylabel("\$C^{(1)nl}(k=0) \\,/\\, {\\epsilon}^{2}_{\\mathrm{TF}}\$")
    # ylabel("\$C^{(1b)}(k=0) \\,/\\, {\\epsilon}^{2}_{\\mathrm{TF}}\$")
    xloc = 3.25
    if rs == 1.0
        xloc = 2.0
        yloc = -0.685
        ydiv = -0.025
        # yloc = -0.525
        # ydiv = -0.025
        # yloc = -0.1675
        # ydiv = -0.0125
    elseif rs == 2.0
        yloc = -0.25
        ydiv = -0.025
    else
        yloc = Inf
        ydiv = 0.0
    end
    # xloc = 1.6
    # yloc = -0.085
    # ydiv = -0.025
    text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\\, \\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
        fontsize=12,
    )
    # text(
    #     xloc,
    #     yloc + ydiv,
    #     "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)}\$";
    #     # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
    #     fontsize=12,
    # )
    # text(
    #     xloc,
    #     yloc + 2 * ydiv,
    #     "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
    #     fontsize=12,
    # )
    # if expand_bare_interactions == 0
    #     plt.title("Using fixed bare Coulomb interactions \$V_1\$, \$V_2\$")
    # elseif expand_bare_interactions == 1
    #     plt.title(
    #         "Using single re-expanded Coulomb interaction \$V_1[V_\\lambda]\$, \$V_2\$",
    #     )
    # elseif expand_bare_interactions == 2
    #     plt.title(
    #         "Using re-expanded Coulomb interactions \$V_1[V_\\lambda]\$, \$V_2[V_\\lambda]\$",
    #     )
    # end
    plt.tight_layout()
    savefig(
        # "results/c1b/c1b_k=0_vs_N_rs=$(rs)_" *
        "results/c1b/c1nl_k=0_vs_N_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(solver)$(ct_string)_comparison.pdf",
        # "neval=$(neval)_$(intn_str)$(solver)$(ct_string)_total.pdf",
    )
    plt.close("all")
    return
end

main()
