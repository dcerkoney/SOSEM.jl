using ElectronLiquid
using ElectronGas
using FeynmanDiagram
using JLD2
using MCIntegration
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# Type signature for partitions
const PartitionType = Vector{Tuple{Int,Int,Int}}

# NOTE: Call from main project directory as: julia examples/c1b0/plot_c1b0_ct_check.jl

"""Convert a list of MCIntegration results for partitions {P} to a Dict of measurements."""
function restodict(res::Result, partitions::PartitionType)
    data = Dict()
    for (i, p) in enumerate(partitions)
        data[p] = measurement.(res.mean[i], res.stdev[i])
    end
    return data
end
function restodict(res_list::Vector{Result}, partitions_list::Vector{PartitionType})
    @assert length(res_list) == length(partitions_list)
    data = Dict()
    for o in eachindex(partitions_list)
        for (i, p) in enumerate(partitions_list[o])
            data[p] = measurement.(res_list[o].mean[i], res_list[o].stdev[i])
        end
    end
    return data
end

function merge_fixed_order_data(
    filenames,
    plotsettings::DiagGen.Settings,
    plotparams::UEG.ParaMC,
)
    # TODO: Refactor---what is the cleanest way to merge the data? 
    #       Should we compose the MCIntegration.Result structs?

    # ParaMC fields for which we require equality between the different JLD2 data
    required_param = [:rs, :beta, :mass2, :isDynamic]

    # Settings fields for which we require equality between the different JLD2 data
    required_settings = [:filter, :interaction, :expand_bare_interactions]

    # Merge JDL2 data from different fixed-order calculations
    local settings, params, kgrid
    res_list = Vector{Result}()
    partitions_list = Vector{PartitionType}()
    for (i, savename) in enumerate(filenames)
        _settings, _params, _kgrid, _partitions, _res = jldopen("$savename.jld2", "a+") do f
            key = "$(UEG.short(plotparams))"
            return f[key]
        end
        # TODO: relax requirement that kgrid is the same for all data
        if i == 1
            kgrid = _kgrid
        else
            @assert kgrid ≈ _kgrid
        end
        # Make sure that required settings are the same for all data
        for field in required_settings
            data_setting = getfield(_settings, field)
            plot_setting = getfield(plotsettings, field)
            if data_setting != plot_setting
                @warn (
                    "Skipping file '$savename': DiagGen settings" *
                    " differ from current plot settings:"
                ) maxlog = 1
                @warn (
                    " • Data setting: $data_setting" * "\n • Plot setting: $plot_setting"
                )
            end
        end
        # Make sure that required parameters are the same for all data
        for field in required_param
            data_param = getfield(_params, field)
            plot_param = getfield(plotparams, field)
            if data_param != plot_param
                @warn (
                    "Skipping file '$savename': MC params differ from current plot params:"
                ) maxlog = 1
                @warn (" • Data param: $data_param" * "\n • Plot param: $plot_param")
            end
        end
        # Pick param & settings from max order
        if i == length(filenames)
            params = _params
            settings = _settings
        end
        # Add the current results and partitions to the lists
        push!(res_list, _res)
        push!(partitions_list, _partitions)
    end
    return (settings, params, kgrid, partitions_list, res_list)
end

function load_z_mu(
    param::UEG.ParaMC,
    parafilename="para.csv",
    ct_filename="examples/counterterms/data_Z.jld2",
)
    # Load μ from csv
    local ct_data
    filefound = false
    f = jldopen(ct_filename, "r")
    for key in keys(f)
        if UEG.paraid(f[key][1]) == UEG.paraid(param)
            ct_data = f[key]
            filefound = true
        end
    end
    if !filefound
        throw(KeyError(UEG.paraid(param)))
    end

    df = CounterTerm.fromFile(parafilename)
    para, _, _, data = ct_data
    printstyled(UEG.short(para); color=:yellow)
    println()

    function zfactor(data, β)
        return @. (imag(data[2, 1]) - imag(data[1, 1])) / (2π / β)
    end

    function mu(data)
        return real(data[1, 1])
    end

    for p in sort([k for k in keys(data)])
        println("$p: μ = $(mu(data[p]))   z = $(zfactor(data[p], para.β))")
    end

    μ = Dict()
    for (p, val) in data
        μ[p] = mu(val)
    end
    z = Dict()
    for (p, val) in data
        z[p] = zfactor(val, para.β)
    end

    return z, μ
end

function main()
    rs = 1.0
    beta = 200.0
    mass2 = 2.0
    # mass2 = 0.1
    solver = :vegasmc
    expand_bare_interactions = false

    neval = 5e8
    max_order = 4
    min_order_plot = 4
    max_order_plot = 4
    @assert max_order ≥ 3

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true
    @assert renorm_mu

    # Include unscreened bare result
    plot_bare = false

    plotparams =
        UEG.ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    plotsettings = DiagGen.Settings(;
        observable=DiagGen.c1bL0,
        n_order=max_order,
        expand_bare_interactions=expand_bare_interactions,
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
    )

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions
        intn_str = "no_bare_"
    end

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # colors = ["orchid", "cornflowerblue", "turquoise", "chartreuse", "greenyellow"]
    # markers = ["-", "-", "-", "-", "-"]

    # Load the results from JLD2
    # savename =
    #     "results/data/c1bL0_n=$(max_order)_rs=$(rs)_" *
    #     "beta_ef=$(beta)_lambda=$(mass2)_" *
    #     "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)"
    # settings, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
    #     key = "$(UEG.short(plotparams))"
    #     return f[key]
    # end

    # Load the results from multiple JLD2 files
    data_fixed_orders = [3, 4]
    filenames = [
        "results/data/c1bL0_n=$(order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)" for
        order in data_fixed_orders
    ]
    settings, param, kgrid, partitions_list, res_list =
        merge_fixed_order_data(filenames, plotsettings, plotparams)

    # Convert fixed-order data to dictionary
    data = restodict(res_list, partitions_list)

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    println(settings)
    println(UEG.paraid(param))
    println(res_list)
    println(partitions_list)

    # Plot the results
    fig, ax = plt.subplots()

    # Non-dimensionalize bare and RPA+FL non-local moments
    rs_quad = 1.0
    sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=200.0.npz")
    # np.load("results/data/soms_rs=$(Float64(param.rs))_beta_ef=$(param.beta).npz")
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    param_quad = Parameter.atomicUnit(0, rs_quad)    # (dimensionless T, rs)
    eTF_quad = param_quad.qTF^2 / (2 * param_quad.me)
    c1b_lo_quad_dimless = sosem_quad.get("bare_b") / eTF_quad^2
    # RPA+FL results for class (b) moment
    c1b_rpa_fl = sosem_quad.get("rpa+fl_b") / eTF_quad^2
    c1b_rpa_fl_err = sosem_quad.get("rpa+fl_b_err") / eTF_quad^2
    if plot_bare
        ax.plot(k_kf_grid_quad, c1b_lo_quad_dimless, "k"; linestyle="--", label="LO (quad)")
        ax.plot(k_kf_grid_quad, c1b_rpa_fl, "k"; label="RPA\$+\$FL (vegas)")
        ax.fill_between(
            k_kf_grid_quad,
            c1b_rpa_fl - c1b_rpa_fl_err,
            c1b_rpa_fl + c1b_rpa_fl_err;
            color="k",
            alpha=0.3,
        )
    end

    # Next available color for plotting
    next_color = 4
    for partitions in partitions_list
        for p in partitions
            if !(min_order_plot <= sum(p) <= max_order_plot)
                continue
            end
            # Get means and error bars from the result up to this order
            # NOTE: Since C⁽¹ᵇ⁾ᴸ = C⁽¹ᵇ⁾ᴿ for the UEG, the
            #       full class (b) moment is C⁽¹ᵇ⁾ = 2C⁽¹ᵇ⁾ᴸ.
            means = 2 * Measurements.value.(data[p])
            stdevs = 2 * Measurements.uncertainty.(data[p])
            # Data gets noisy above 1st Green's function counterterm order
            # marker =
            #     (p[2] > 1 || (p[1] > 3 && p[2] > 0)) ?
            #     "o-" : "-"
            marker = "-"
            ax.plot(
                k_kf_grid,
                means,
                marker;
                markersize=2,
                color=next_color == 3 ? "C0" : "C$next_color",
                # color="C$next_color",
                label="\$\\widetilde{C}^{(1b0)}_{$p}\$",
                # label="\$\\widetilde{C}^{(1b0)}_{$p}\$ ($solver)",
                # label="\$\\mathcal{P}=$p\$ ($solver)",
            )
            ax.fill_between(
                k_kf_grid,
                means - stdevs,
                means + stdevs;
                color=next_color == 3 ? "C0" : "C$next_color",
                # color="C$next_color",
                alpha=0.4,
            )
            next_color -= 1
        end
    end

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])

    # Reexpand merged data in powers of μ
    z, μ = load_z_mu(param)
    δz, δμ = CounterTerm.sigmaCT(max_order - 2, μ, z; verbose=1)
    println("Computed δμ: ", δμ)
    c1b0 = UEG_MC.chemicalpotential_renormalization(
        merged_data,
        δμ;
        min_order=min_order_plot,
        max_order=max_order_plot,
    )

    # Check the counterterm cancellation to leading order in δμ
    if min_order_plot ≤ 3
        # Test manual renormalization with exact lowest-order chemical potential;
        # the first-order counterterm is: δμ1= ReΣ₁[λ](kF, 0)
        δμ1_exact = UEG_MC.delta_mu1(param)
        # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1} (exact δμ₁)
        c1b03_exact = merged_data[(3, 0)] + δμ1_exact * merged_data[(2, 1)]
        c1b03_means_exact = Measurements.value.(c1b03_exact)
        c1b03_errs_exact = Measurements.uncertainty.(c1b03_exact)
        println("Largest magnitude of C^{(1b0)}_{n=3}(k): $(maximum(abs.(c1b03_exact)))")
        # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1} (calc δμ₁)
        c1b03 = c1b0[2]  # c1b0 = [c1b02, c1b03, ...]
        c1b03_means = Measurements.value.(c1b03)
        c1b03_errs = Measurements.uncertainty.(c1b03)
        stdscores = stdscore.(c1b03, c1b03_exact)
        worst_score = argmax(abs, stdscores)
        println("Exact δμ₁: ", δμ1_exact)
        println("Computed δμ₁: ", δμ[1])
        println(
            "Worst standard score for total result to 3rd " *
            "order (auto vs exact+manual): $worst_score",
        )

        c1b03_kind = ["exact", "calc."]
        c1b03_kind_means = [c1b03_means_exact, c1b03_means]
        c1b03_kind_errs = [c1b03_errs_exact, c1b03_errs]
        for (kind, means, errs) in zip(c1b03_kind, c1b03_kind_means, c1b03_kind_errs)
            ax.plot(
                k_kf_grid,
                means,
                "-";
                # "o-";
                markersize=2,
                color="C$next_color",
                label="\$\\widetilde{C}^{(1b0)}_{n=3} = \\widetilde{C}^{(1b0)}_{(3,0)} " *
                      " + \\delta\\mu_1 \\widetilde{C}^{(1b0)}_{(2,1)}\$ ($kind \$\\delta\\mu\$, $solver)",
            )
            ax.fill_between(
                k_kf_grid,
                means - errs,
                means + errs;
                color="C$next_color",
                alpha=0.4,
            )
            next_color += 1
        end
    end

    # Plot the counterterm cancellation at next-leading order in δμ
    if max_order_plot ≥ 4
        c1b04_means = Measurements.value.(c1b0[3])
        c1b04_errs = Measurements.uncertainty.(c1b0[3])
        ax.plot(
            k_kf_grid,
            c1b04_means,
            "-";
            # "o-";
            # markersize=3,
            linewidth=2,
            color=next_color == 0 ? "r" : "C$next_color",
            label="\$\\widetilde{C}^{(1b0)}_{n=4}\$",
        )
        ax.fill_between(
            k_kf_grid,
            c1b04_means - c1b04_errs,
            c1b04_means + c1b04_errs;
            color=next_color == 0 ? "r" : "C$next_color",
            alpha=0.4,
        )
        ax.text(
            0.1025,
            -0.75,
            "\$\\widetilde{C}^{(1b0)}_{n=4} = \\widetilde{C}^{(1b0)}_{(4,0)} + " *
            "\\delta\\mu_1 \\widetilde{C}^{(1b0)}_{(3,1)}\$";
            fontsize=12,
        )
        ax.text(
            0.31,
            -1.025,
            "\$ + \\delta\\mu^2_1 \\widetilde{C}^{(1b0)}_{(2,2)} + " *
            "\\delta\\mu_2 \\widetilde{C}^{(1b0)}_{(2,1)}\$";
            fontsize=12,
        )
    end

    # Plot labels and legend
    ax.legend(; loc="lower right", ncol=2)
    ax.set_xlim(minimum(k_kf_grid), min(2.0, maximum(k_kf_grid)))
    # ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    # ax.set_ylim(-0.1, 0.0025)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$\\widetilde{C}^{(1b0)}_{(\\,\\cdot\\,)}(k) " *
        " \\equiv C^{(1b0)}_{(\\,\\cdot\\,)}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    # # (nmax = 3)
    # xloc = 1.75
    # yloc = -0.15
    # ydiv = -0.05
    # (nmax = 4)
    xloc = 1.125
    yloc = 1.5
    ydiv = -0.3
    ax.text(
        xloc,
        yloc,
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = 200,\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = 2\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + 2 * ydiv,
        "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=12,
    )
    plt.title("Using fixed bare Coulomb interactions \$V_1\$, \$V_2\$")
    # plt.title(
    #     "Using re-expanded Coulomb interactions \$V_1[V_\\lambda]\$, \$V_2[V_\\lambda]\$",
    # )
    plt.tight_layout()
    fig.savefig(
        "results/c1b0/c1b0_n=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)_ct_cancellation.pdf",
    )
    plt.close("all")
    return
end

main()
