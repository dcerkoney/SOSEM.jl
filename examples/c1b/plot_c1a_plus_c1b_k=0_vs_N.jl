using CodecZlib
using ElectronLiquid
using ElectronGas
using FeynmanDiagram
using Interpolations
using JLD2
using MCIntegration
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# We plot the k=0 value
const extK_load = 0.0

# NOTE: Call from main project directory as: julia examples/c1b/plot_c1b_total.jl

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
    # expand_bare_interactions = 0          # bare V, V (non-reexpanded) scheme
    expand_bare_interactions = 1          # single V[V_λ] scheme

    # neval = 1e10
    neval = 1e9

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    n_min = 2  # True minimal loop order for this observable
    min_order = 2
    max_order = 5
    min_order_plot = 2
    max_order_plot = 5
    @assert max_order ≥ 3

    # Load data from multiple fixed-order runs
    # fixed_orders = collect(min_order:max_order)

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Save total results?
    save = true

    # Include RPA(+FL) results?
    plot_rpa_fl = true

    plotparam =
        UEG.ParaMC(; order=max_order, rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions == 2
        intn_str = "no_bare_"
    elseif expand_bare_interactions == 1
        intn_str = "one_bare_"
    end

    filename_c1a =
        "results/data/rs=$(plotparam.rs)_beta_ef=$(plotparam.beta)_" *
        "lambda=$(plotparam.mass2)_$(solver)$(ct_string)"
    filename_c1b_unif =
        "results/data/rs=$(plotparam.rs)_beta_ef=$(plotparam.beta)_" *
        "lambda=$(plotparam.mass2)_$(intn_str)$(solver)$(ct_string)"

    # Load the data for each observable
    local param
    c1as = []
    c1b_unifs = []
    for (i, N) in enumerate(min_order_plot:max_order_plot)
        fa = jldopen("$filename_c1a.jld2", "r")
        c1a = fa["c1l/N=$N/neval=1.0e10/meas"]
        close(fa)  # close file a
        fb = jldopen("$filename_c1b_unif.jld2", "r")
        if N == 2
            c1b_unif = zero(Measurement)
        else
            c1b_unif = fb["c1b_k=0/N=$N/neval=1.0e9/meas"]
            param = fb["c1b_k=0/N=$N/neval=1.0e9/param"]
        end
        close(fb)  # close file b
        push!(c1as, c1a[1])
        push!(c1b_unifs, c1b_unif[1])
    end

    # Get the total result c1a + c1b_unif
    c1a_plus_c1b_unif = c1as + c1b_unifs
    means, stdevs =
        Measurements.value.(c1a_plus_c1b_unif), Measurements.uncertainty.(c1a_plus_c1b_unif)

    println("c1as:\n$c1as")
    println("c1b_unifs:\n$c1b_unifs")

    orders = min_order_plot:max_order_plot

    # Check convergence with order
    m_prev = means[1]
    println("Percent difference btw. successive orders:")
    for (i, m) in enumerate(means[2:end])
        N = i + n_min
        println("N=$N:\t", 100 * abs((m - m_prev) / m_prev))
        m_prev = m
    end

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # Plot results vs order N
    fig, ax = plt.subplots()
    # Data gets noisy above 3rd loop order
    marker = "o-"
    ax.plot(orders, means, marker; markersize=4, color="C0", label="RPT ($solver)")
    ax.fill_between(orders, means - stdevs, means + stdevs; color="C0", alpha=0.4)
    # ax.legend(; loc="best")
    ax.set_xticks(orders)
    ax.set_xlim(minimum(orders), maximum(orders))
    ax.set_xlabel("Perturbation order \$N\$")
    ax.set_ylabel(
        "\$\\left(C^{(1a)} + C^{(1b)}(k=0)\\right) \\,/\\, {\\epsilon}^{2}_{\\mathrm{TF}}\$",
    )
    xloc = 3.65
    if rs == 1.0
        # yloc = 0.93
        # ydiv = -0.0075
        yloc = 0.94
        ydiv = -0.01
    else
        yloc = Inf
        ydiv = 0.0
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
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(neval)},\$";
        fontsize=14,
    )
    ax.text(
        xloc,
        yloc + 2 * ydiv,
        "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
        fontsize=12,
    )
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
    fig.savefig(
        "results/c1b/c1a_plus_c1b_k=0_vs_N_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        "neval=$(neval)_$(intn_str)$(solver)$(ct_string).pdf",
    )
    plt.close("all")
    return
end

main()
