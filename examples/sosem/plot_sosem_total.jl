using ElectronLiquid
using ElectronGas
using JLD2
using Measurements
using PyCall
using SOSEM: UEG_MC

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory as: julia examples/c1c/plot_c1c_total.jl

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
function aggregate_results_c1cN(c1c; nmax, nmin=2, renorm_mu=false)
    c1c_total = Dict()
    if renorm_mu
        # merged data is an ordered vector of data at each order nmin ≤ n ≤ nmax
        c1c_total =
            Dict(zip(nmin:nmax, accumulate(+, c1c[i] for i in eachindex(nmin:nmax))))
    else
        # merged data is a Dict of interaction-merged partitions P
        for n in nmin:nmax
            c1c_total[n] = zero(c1c[(n, 0)])
            println(n)
            for (p, meas) in c1c
                # if p[1] <= n
                if sum(p) <= n
                    println("adding partition $p to $n-order aggregate")
                    c1c_total[n] += meas
                end
            end
        end
    end
    return c1c_total
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
end

function main()
    rs = 1.0
    beta = 200.0
    mass2 = 2.0
    solver = :vegasmc
    expand_bare_interactions = false

    neval = 1e8
    min_order = 2
    max_order = 4
    max_order_plot = 4

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Manually perform chemical potential renormalization
    renorm_mu_lo_ex = false  # at lowest order
    renorm_mu_nlo_ex = false  # at next-lowest order

    plotparam =
        UEG.ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions
        intn_str = "no_bare_"
    end

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "with_ct" : ""
    renorm_string = (renorm_mu && renorm_mu_lo_ex) ? "_lo_mu_manual" : ""
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

    # Get total data
    if renorm_mu
        if renorm_mu_lo_ex && max_order_plot == 3
            δμ1 = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
            # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1}
            c1c2 = merged_data[(2, 0)]
            c1c3 = merged_data[(3, 0)] + δμ1 * merged_data[(2, 1)]
            c1c = [c1c2, c1c3]
        else
            # Reexpand merged data in powers of μ
            z, μ = load_z_mu(param)
            δz, δμ = CounterTerm.sigmaCT(max_order - 2, μ, z; verbose=1)
            println("Computed δμ: ", δμ)
            c1c = UEG_MC.chemicalpotential_renormalization(max_order_plot, merged_data, δμ)
            # Test manual renormalization with exact lowest-order chemical potential
            if !renorm_mu_lo_ex && max_order >= 3
                δμ1_exact = UEG_MC.delta_mu1(param)  # = ReΣ₁[λ](kF, 0)
                # C⁽¹⁾₃ = C⁽¹⁾_{3,0} + δμ₁ C⁽¹⁾_{2,1}
                c1c3_manual =
                    merged_data[(2, 0)] +
                    merged_data[(3, 0)] +
                    δμ1_exact * merged_data[(2, 1)]
                c1c3 = c1c[1] + c1c[2]
                stdscores = stdscore.(c1c3, c1c3_manual)
                worst_score = argmax(abs, stdscores)
                println("Exact δμ₁: ", δμ1_exact)
                println("Computed δμ₁: ", δμ[1])
                println(
                    "Worst standard score for total result to 3rd " *
                    "order (auto vs exact+manual): $worst_score",
                )
            end
        end
    else
        c1c = merged_data
    end

    # Aggregate the full results for C⁽¹ᶜ⁾ up to order N
    if renorm_mu_lo_ex
        c1c_total = Dict(2 => c1c2, 3 => c1c2 + c1c3)
    else
        c1c_total = aggregate_results_c1cN(
            c1c;
            nmin=min_order,
            nmax=max_order_plot,
            renorm_mu=renorm_mu,
        )
    end

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

    data = np.load("results/data/soms_rs=$(rs_quad)_beta_ef=200.0.npz")

    # Bare results (stored in Hartree a.u.)
    c1c_bare_quad = data.get("bare_c") / eTF_quad^2
    ax.plot(
        k_kf_grid_quad,
        c1c_bare_quad,
        "k";
        label="LO (quad)",
        # label="\$LO = \\mathrm{RPA}+\\mathrm{FL}\$ (quad)",
    )
    # No additional RPA+FL results for class (c) moment!

    # Plot for each aggregate order
    for N in min_order:max_order_plot
        # Get means and error bars from the result up to this order
        means = Measurements.value.(c1c_total[N])
        stdevs = Measurements.uncertainty.(c1c_total[N])
        # Data gets noisy above 3rd loop order
        # marker = "o-"
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
        if !renorm_mu_lo_ex && max_order <= 3 && N == 3
            ax.plot(
                k_kf_grid,
                Measurements.value.(c1c3_manual);
                color="r",
                linestyle="--",
                label="\$N=3\$ (manual, vegasmc)",
            )
        end
    end
    ax.legend(; loc="lower right")
    ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel(
        "\$C^{(1c)}_{N}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    # xloc = 0.5
    # yloc = -0.075
    # ydiv = -0.009
    xloc = 1.75
    yloc = -0.5
    ydiv = -0.095
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
        "results/c1c/c1c_N=$(max_order_plot)_rs=$(param.rs)_" *
        "beta_ef=$(param.beta)_lambda=$(param.mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)_$(ct_string)" *
        "$(renorm_string)_total.pdf",
    )
    plt.close("all")
    return
end

main()
