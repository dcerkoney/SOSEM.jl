using ElectronLiquid
using ElectronGas
using JLD2
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1c/plot_c1c.jl

"""Convert a list of MCIntegration results for partitions {P} to a Dict of measurements."""
function restodict(res, partitions)
    data = Dict()
    for (i, p) in enumerate(partitions)
        data[p] = measurement.(res.mean[i], res.stdev[i])
    end
    return data
end

"""
Aggregate the measurements for C⁽¹ᶜ⁾ up to order N for nmin ≤ N ≤ nmax.
Assumes the input data has already been merged by interaction order and, 
if applicable, reexpanded in μ.
"""
function aggregate_results_c1cN(merged_data; nmax, nmin=2)
    c1cN = Dict()
    for n in nmin:nmax
        c1cN[n] = zero(merged_data[(n, 0)])
        println(n)
        for (p, meas) in merged_data
            # if p[1] <= n
            if sum(p) <= n
                println("adding partition $p to $n-order aggregate")
                c1cN[n] += meas
            end
        end
    end
    return c1cN
end

function main()
    rs = 2.0
    beta = 200.0
    mass2 = 2.0
    solver = :vegasmc
    expand_bare_interactions = true

    neval = 1e7
    min_order = 2
    max_order = 4
    max_order_plot = 3

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu_ex = true
    renorm_mu = true
    renorm_lambda = true

    plotparam =
        UEG.ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions
        intn_str = "no_bare_"
    end

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "with_ct" : ""
    ex_string = (renorm_mu && renorm_mu_ex) ? "_mu_ex" : ""
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

    # Load the results from JLD2 (and μ data from csv, if applicable)
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

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = restodict(res, partitions)
    merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])

    parafilename = "para.csv"
    ct_filename = "examples/counterterms/data_Z.jld2"

    # Get total data
    local c1cN
    if max_order_plot == 3 && renorm_mu_ex
        eTF = param.qTF^2 / (2 * param.me)
        μ0 = param.EF
        # δμ1 = -eTF / 2  # = Σₓ(kF)  (μHF = EF + Σₓ(kF))
        # δμ1 = eTF / 2  # = Σₓ(kF)  (μHF = EF + Σₓ(kF))
        δμ1 = μ0 + eTF / 2  # = Σₓ(kF)  (μHF = EF + Σₓ(kF))
        # μ1 = μ0 + δμ1
        # C⁽¹⁾_3 = C⁽¹⁾_{3,0} + δμ1 C⁽¹⁾_{2,1}
        c1c2 = merged_data[(2, 0)]
        c1c3 = merged_data[(3, 0)] + δμ1 * merged_data[(2, 1)]
        c1cN = [c1c2, c1c3]
    elseif renorm_mu
        # Load μ from csv
        local ct_data
        f = jldopen(ct_filename, "r")
        for key in keys(f)
            if UEG.paraid(f[key][1]) == UEG.paraid(param)
                ct_data = f[key]
            end
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

        # Reexpand merged data in powers of μ, if applicable
        _, δμ, _ = CounterTerm.sigmaCT(max_order_plot - 2, μ)
        c1cN = SOSEM.chemicalpotential_renormalization(max_order_plot, merged_data, δμ)
    else
        # Aggregate the full results for C⁽¹ᶜ⁾
        c1cN = aggregate_results_c1cN(merged_data; nmax=max_order_plot, nmin=min_order)
    end

    println(settings)
    println(UEG.paraid(param))
    println(partitions)
    println(res)
    println(c1cN)

    # Plot the results
    fig, ax = plt.subplots()

    # Non-dimensionalize bare and RPA+FL non-local moments
    rs_quad = 2.0
    sosem_quad = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=200.0.npz")
    # np.load("results/data/soms_rs=$(Float64(param.rs))_beta_ef=$(param.beta).npz")
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    param_quad = Parameter.atomicUnit(0, rs_quad)    # (dimensionless T, rs)
    eTF_quad = param_quad.qTF^2 / (2 * param_quad.me)

    data = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=200.0.npz")

    # Bare results (stored in Hartree a.u.)
    c1c_bare_quad = data.get("bare_c") / eTF_quad^2
    ax.plot(
        k_kf_grid_quad,
        c1c_bare_quad,
        "k";
        label="\$LO = \\mathrm{RPA}+\\mathrm{FL}\$ (quad)",
    )

    # No additional RPA+FL results for class (c) moment!

    # Plot for each aggregate order
    for (i, N) in enumerate(min_order:max_order_plot)
        # Get means and error bars from the result up to this order
        means = Measurements.value.(c1cN[i])
        stdevs = Measurements.uncertainty.(c1cN[i])
        # Data gets noisy above 3rd loop order
        marker = N > 3 ? "o-" : "-"
        ax.plot(
            k_kf_grid,
            means,
            marker;
            markersize=2,
            color="C$(N - 2)",
            label="\$N=$N\$ ($solver)",
        )
        ax.fill_between(
            k_kf_grid,
            means - stdevs,
            means + stdevs;
            color="C$(N - 2)",
            alpha=0.4,
        )
    end
    ax.legend(; loc="lower right")
    ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$C^{(1c)}_{N}(\\mathbf{k}) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    # xloc = 0.5
    # yloc = -0.075
    # ydiv = -0.009
    xloc = 1.75
    yloc = -0.3
    ydiv = -0.085
    ax.text(
        xloc,
        yloc,
        "\$r_s = 2,\\, \\beta \\hspace{0.1em} \\epsilon_F = 200,\$";
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
    # plt.title("Using fixed bare Coulomb interactions \$V_1\$, \$V_2\$")
    plt.title(
        "Using re-expanded Coulomb interactions \$V_1[V_\\lambda]\$, \$V_2[V_\\lambda]\$",
    )
    plt.tight_layout()
    fig.savefig(
        "results/c1c/c1c_N=$(param.order)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)$(ex_string)_total.pdf",
    )
    plt.close("all")
    return
end

main()
