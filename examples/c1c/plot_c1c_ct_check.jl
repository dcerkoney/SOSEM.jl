using ElectronLiquid
using ElectronGas
using JLD2
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1c/plot_c1c_ct_check.jl

"""Convert a list of MCIntegration results for partitions {P} to a Dict of measurements."""
function restodict(res, partitions)
    data = Dict()
    for (i, p) in enumerate(partitions)
        data[p] = measurement.(res.mean[i], res.stdev[i])
    end
    return data
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
    min_order_plot = 3
    max_order_plot = 3
    @assert max_order ≥ 3

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true
    @assert renorm_mu

    # Include unscreened bare result
    plot_bare = false

    plotparam =
        UEG.ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

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
    savename =
        "results/data/c1c_n=$(max_order)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)"
    settings, param, kgrid, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(plotparam))"
        return f[key]
    end
    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    println(settings)
    println(UEG.paraid(param))
    println(partitions)
    println(res)

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
    c1c_quad_dimless = sosem_quad.get("bare_c") / eTF_quad^2
    if plot_bare
        ax.plot(
            k_kf_grid_quad,
            c1c_quad_dimless,
            "k";
            label="\$\\mathcal{P}=$((2,0,0))\$ (quad)",
        )
    end
    for o in eachindex(partitions)
        if !(min_order_plot <= sum(partitions[o]) <= max_order_plot)
            continue
        end
        # Get means and error bars from the result for this partition
        local means, stdevs
        if res.config.N == 1
            # res gets automatically flattened for a single-partition measurement
            means, stdevs = res.mean, res.stdev
        else
            means, stdevs = res.mean[o], res.stdev[o]
        end
        # Data gets noisy above 1st Green's function counterterm order
        marker =
            (partitions[o][2] > 1 || (partitions[o][1] > 3 && partitions[o][2] > 0)) ?
            "o-" : "-"
        # marker = "o-"
        ax.plot(
            k_kf_grid,
            means,
            marker;
            markersize=2,
            color="C$(o - 1)",
            label="\$\\widetilde{C}^{(1c)}_{$(partitions[o])}\$ ($solver)",
            # label="\$\\mathcal{P}=$(partitions[o])\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            color="C$(o - 1)",
            alpha=0.4,
        )
    end

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = restodict(res, partitions)
    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])

    # Reexpand merged data in powers of μ
    z, μ = load_z_mu(param)
    δz, δμ = CounterTerm.sigmaCT(max_order - 2, μ, z; verbose=1)
    println("Computed δμ: ", δμ)
    c1c = UEG_MC.chemicalpotential_renormalization(max_order_plot, merged_data, δμ)

    # Test manual renormalization with exact lowest-order chemical potential;
    # the first-order counterterm is: δμ1= ReΣ₁[λ](kF, 0)
    δμ1_exact = UEG_MC.delta_mu1(param)
    # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1} (exact δμ₁)
    c1c3_exact = merged_data[(3, 0)] + δμ1_exact * merged_data[(2, 1)]
    c1c3_means_exact = Measurements.value.(c1c3_exact)
    c1c3_errs_exact = Measurements.uncertainty.(c1c3_exact)
    println("Largest magnitude of C^{(1c)}_{n=3}(k): $(maximum(abs.(c1c3_exact)))")
    # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1} (calc δμ₁)
    c1c3 = c1c[2]  # c1c = [c1c2, c1c3, ...]
    c1c3_means = Measurements.value.(c1c3)
    c1c3_errs = Measurements.uncertainty.(c1c3)
    stdscores = stdscore.(c1c3, c1c3_exact)
    worst_score = argmax(abs, stdscores)
    println("Exact δμ₁: ", δμ1_exact)
    println("Computed δμ₁: ", δμ[1])
    println(
        "Worst standard score for total result to 3rd " *
        "order (auto vs exact+manual): $worst_score",
    )

    # Check the counterterm cancellation to leading order in δμ
    c1c3_kind = ["exact", "calc."]
    c1c3_kind_means = [c1c3_means_exact, c1c3_means]
    c1c3_kind_errs = [c1c3_errs_exact, c1c3_errs]
    next_color = length(partitions)  # next available color for plotting
    for (kind, means, errs) in zip(c1c3_kind, c1c3_kind_means, c1c3_kind_errs)
        ax.plot(
            k_kf_grid,
            means,
            "-";
            # "o-";
            markersize=2,
            color="C$next_color",
            label="\$\\widetilde{C}^{(1c)}_{n=3} = \\widetilde{C}^{(1c)}_{(3,0)} " *
                  " + \\delta\\mu_1 \\widetilde{C}^{(1c)}_{(2,1)}\$ ($kind \$\\delta\\mu_1\$, $solver)",
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

    if max_order ≥ 4 && max_order_plot ≥ 4
        # Plot the counterterm cancellation at next-leading order in δμ
        c1c4_means = Measurements.value.(c1c[3])
        c1c4_errs = Measurements.uncertainty.(c1c[3])
        ax.plot(
            k_kf_grid,
            c1c4_means,
            "-";
            # "o-";
            markersize=2,
            color="C$next_color",
            label="\$\\widetilde{C}^{(1c)}_{n=4} = \\widetilde{C}^{(1c)}_{(4,0)} + " *
                  "\\delta\\mu_1 \\widetilde{C}^{(1c)}_{(3,1)}\$ + " *
                  "\\delta\\mu^2_1 \\widetilde{C}^{(1c)}_{(2,2)}\$ + " *
                  "\\delta\\mu_2 \\widetilde{C}^{(1c)}_{(2,1)}\$ " *
                  "($kind \$\\delta\\mu_1\$, $solver)",
        )
        ax.fill_between(
            k_kf_grid,
            c1c4_means - c1c4_errs,
            c1c4_means + c1c4_errs;
            color="C$next_color",
            alpha=0.4,
        )
    end

    # Plot labels and legend
    ax.legend(; loc="lower right")
    ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    # ax.set_ylim(-0.1, 0.0025)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$\\widetilde{C}^{(1c)}_{(\\,\\cdot\\,)}(k) " *
        " \\equiv C^{(1c)}_{(\\,\\cdot\\,)}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    xloc = 1.75
    yloc = -0.15
    ydiv = -0.05
    # xloc = 1.75
    # yloc = -0.02
    # yloc = -0.055
    # ydiv = -0.01
    # ydiv = -0.009
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
        "results/c1c/c1c_n=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)_ct_cancellation.pdf",
    )
    plt.close("all")
    return
end

main()
