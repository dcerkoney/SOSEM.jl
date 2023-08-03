using CodecZlib
using ElectronGas
using ElectronLiquid
using Interpolations
using JLD2
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

const c1l_N_vs_rs = Dict([
    # 0.5 => ,
    1.0 => [
        0.913 ± 0.0017,   # 𝓞(1)   exact
        1.0638 ± 0.0019,  # 𝓞(ξ),  rpt   
        1.1151 ± 0.0028,  # 𝓞(ξ²), rpt   
        1.1365 ± 0.0073,  # 𝓞(ξ³), rpt   
        # 1.102 ± 0.097,  # 𝓞(ξ⁴), rpt
    ],
    # 1.5 => ,
    # 2.0 => ,
])

# (V, V)
const c1nl_vv_unif_N_vs_rs = Dict([
    # 0.5 => ,
    1.0 => [
        -0.5 ± 0.0,          # 𝓞(1)   exact
        -0.65255 ± 0.00096,  # 𝓞(ξ),  rpt, blended data
        -0.6502 ± 0.0027,    # 𝓞(ξ²), rpt, blended data
        -0.676 ± 0.0074,     # 𝓞(ξ³), rpt, blended data
        #
        # # k-dependent run at k = 0
        # -0.6497 ± 0.0013,
        # -0.6476 ± 0.0037,
        # -0.673 ± 0.01,
        #
        # # k = 0 run
        # -0.6554 ± 0.0014
        # -0.6528 ± 0.0038
        # -0.679 ± 0.011
    ],
    # 1.5 => ,
    # 2.0 => ,
])

# (V, V_λ)
const c1nl_vvlambda_unif_N_vs_rs = Dict([
    # 0.5 => ,
    1.0 => [
        -0.5 ± 0.0,        # 𝓞(1)   exact
        -0.5967 ± 0.0013,  # 𝓞(ξ),  rpt, k = 0 run
        -0.6286 ± 0.0037,  # 𝓞(ξ²), rpt, k = 0 run
        -0.64 ± 0.01,      # 𝓞(ξ³), rpt, k = 0 run
        #
    ],
    # 1.5 => ,
    # 2.0 => ,
])

function spline(x, y, e)
    # generate knots with spline without constraints
    w = 1.0 ./ e
    spl = interp.UnivariateSpline(x, y; w=w, k=3)
    __x = collect(LinRange(x[1], x[end], 1000))
    yfit = spl(__x)
    return __x, yfit
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
        # nl
        cdict["orange"],   # rpa(+fl)
        cdict["magenta"],  # rpt (V, V)
        cdict["red"],      # rpt (V, V_λ)
        # l
        cdict["blue"],     # rpa(+fl)
        cdict["cyan"],     # rpt
    ]
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")

    # Use LaTex fonts for plots
    rcParams["font.size"] = 16
    rcParams["mathtext.fontset"] = "cm"
    # rcParams["font.family"] = "Times New Roman"

    rpt_rslist               = [1.0]
    rpt_l_som                = [c1l_N_vs_rs[rs][end] for rs in rpt_rslist]
    rpt_unif_nl_som_vv       = [c1nl_vv_unif_N_vs_rs[rs][end] for rs in rpt_rslist]
    rpt_unif_nl_som_vvlambda = [c1nl_vvlambda_unif_N_vs_rs[rs][end] for rs in rpt_rslist]

    rs_list = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0]
    unif_rpa_nl_som = []
    unif_rpa_fl_nl_som = []
    local_rpa_som = []
    local_rpa_fl_som = []
    for rs in rs_list
        println("rs = $rs")
        # Non-dimensionalize bare and RPA+FL non-local moments
        rs_lo = rs
        # FIXME: No time to re-run the RPA results at beta = 40 / ef, but the old results a
        #        beta = 100 are essentially at the same temperature, so reusing the data here
        # FIXME: Rerun the RPA results (n_vegas = 1e6) with n_vegas = 1e7 to match the RPA+FL data
        # FIXME: Rerun RPA+FL at rs = 0.1 with n_vegas = 1e7
        sosem_lo_rpa = np.load("results/data/python/soms_rs=$(rs_lo)_beta=100.0.npz")
        sosem_lo_rpa_fl = np.load("results/data/python/soms_rs=$(rs_lo)_beta_ef=40.0.npz")
        # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
        param_lo = Parameter.atomicUnit(0, rs_lo)    # (dimensionless T, rs)
        eTF_lo = param_lo.qTF^2 / (2 * param_lo.me)
        # RPA(+FL) results (stored in Hartree a.u.)
        c1l_rpa = abs(sosem_lo_rpa.get("rpa_a_T=0")[1]) / eTF_lo^2
        c1l_rpa_fl = abs(sosem_lo_rpa_fl.get("rpa+fl_a_T=0")[1]) / eTF_lo^2
        c1nl_rpa =
            (sosem_lo_rpa.get("rpa_b") + sosem_lo_rpa.get("bare_c") + sosem_lo_rpa.get(
                "bare_d",
            ))[1] / eTF_lo^2
        c1nl_rpa_fl =
            (sosem_lo_rpa_fl.get("rpa+fl_b") + sosem_lo_rpa_fl.get("bare_c") + sosem_lo_rpa_fl.get(
                "bare_d",
            ))[1] / eTF_lo^2
        push!(unif_rpa_nl_som, c1nl_rpa)
        push!(unif_rpa_fl_nl_som, c1nl_rpa_fl)
        push!(local_rpa_som, c1l_rpa)
        push!(local_rpa_fl_som, c1l_rpa_fl)
    end

    # Plot the local RPA moment and full uniform moments vs rs for comparison
    fig1 = figure(; figsize=(6, 4))
    axhline(; y=-0.5, linestyle="-", color="k", linewidth=1)
    # axhline(; y=-0.5, linestyle="-", color="k", linewidth=1, label="\$\\mathrm{LO}_1\$")
    plot(
        rs_list,
        unif_rpa_nl_som,
        "o--";
        markersize=2,
        label="\$\\mathrm{RPA}_1\$",
        color=color[1],
    )
    plot(
        rs_list,
        unif_rpa_fl_nl_som,
        "o-";
        markersize=2,
        label="\$\\mathrm{RPA+FL}_1\$",
        color=color[1],
    )

    println(rpt_l_som)
    println(rpt_unif_nl_som_vv)
    println(rpt_unif_nl_som_vvlambda)

    # plot rpt uniform nonlocal moment
    rpt_unif_nl_vv_means = Measurements.value.(rpt_unif_nl_som_vv)
    rpt_unif_nl_vv_errs = Measurements.uncertainty.(rpt_unif_nl_som_vv)
    errorbar(
        rpt_rslist,
        rpt_unif_nl_vv_means;
        yerr=rpt_unif_nl_vv_errs,
        color=color[2],
        capsize=2,
        markersize=2,
        fmt="o",
        markerfacecolor="none",
        label="\$\\mathrm{RPT}_1\$ \$(V, V)\$",
        # zorder=100,
    )
    rpt_unif_nl_vvlambda_means = Measurements.value.(rpt_unif_nl_som_vvlambda)
    rpt_unif_nl_vvlambda_errs = Measurements.uncertainty.(rpt_unif_nl_som_vvlambda)
    errorbar(
        rpt_rslist,
        rpt_unif_nl_vvlambda_means;
        yerr=rpt_unif_nl_vvlambda_errs,
        color=color[3],
        capsize=2,
        markersize=2,
        fmt="o",
        markerfacecolor="none",
        label="\$\\mathrm{RPT}_1\$ \$(V, V_\\lambda)\$",
        # zorder=101,
    )

    plot(
        rs_list,
        local_rpa_som,
        "o--";
        markersize=2,
        label="\$\\mathrm{RPA}_2\$",
        color=color[4],
    )
    plot(
        rs_list,
        local_rpa_fl_som,
        "o-";
        markersize=2,
        label="\$\\mathrm{RPA+FL}_2\$",
        color=color[4],
    )

    # plot rpt local moment
    rpt_l_means = Measurements.value.(rpt_l_som)
    rpt_l_errs = Measurements.uncertainty.(rpt_l_som)
    errorbar(
        rpt_rslist,
        rpt_l_means;
        yerr=rpt_l_errs,
        color=color[5],
        capsize=2,
        markersize=2,
        fmt="o",
        markerfacecolor="none",
        label="\$\\mathrm{RPT}_2\$",
        # zorder=102,
    )

    xlim(0.0, 5.1)
    # ylim(-1.7, 1.7)
    xticks(rs_list)
    legend(; loc=(0.05, 0.2425), ncol=2)
    xlabel("\$r_s\$")
    ylabel("\$C^{(1)} \\,/\\, \\epsilon^2_{\\mathrm{TF}} \$")
    savefig("results/c1l_vs_c1nl/c1l_c1nl_comparison_vs_rs.pdf")
    plt.tight_layout()
    plt.close("all")

    # Plot the local RPA moment and full uniform moments vs rs for comparison
    fig2 = figure(; figsize=(6, 4))
    axhline(1.0; linestyle="-", color="k", linewidth=1)
    ratio_rpa = abs.(unif_rpa_nl_som ./ local_rpa_som)
    ratio_rpa_fl = abs.(unif_rpa_fl_nl_som ./ local_rpa_fl_som)
    ratio_rpt_vv = abs.(rpt_unif_nl_som_vv ./ rpt_l_som)
    ratio_rpt_vvlambda = abs.(rpt_unif_nl_som_vvlambda ./ rpt_l_som)
    plot(
        rs_list,
        ratio_rpa,
        "o--";
        markersize=2,
        label="\$\\mathrm{RPA}\$",
        color=cdict["orange"],
    )
    plot(
        rs_list,
        ratio_rpa_fl,
        "o-";
        markersize=2,
        label="\$\\mathrm{RPA+FL}\$",
        color=cdict["orange"],
    )
    errorbar(
        rpt_rslist,
        Measurements.value.(ratio_rpt_vv);
        yerr=Measurements.uncertainty.(ratio_rpt_vv),
        color=cdict["blue"],
        capsize=2,
        markersize=2,
        fmt="o",
        markerfacecolor="none",
        label="\$\\mathrm{RPT}\$ \$(V, V)\$",
        # zorder=100,
    )
    errorbar(
        rpt_rslist,
        Measurements.value.(ratio_rpt_vvlambda);
        yerr=Measurements.uncertainty.(ratio_rpt_vvlambda),
        color=cdict["cyan"],
        capsize=2,
        markersize=2,
        fmt="o",
        markerfacecolor="none",
        label="\$\\mathrm{RPT}\$ \$(V, V_\\lambda)\$",
        # zorder=101,
    )
    xlim(0.0, 5.1)
    ylim(0.25, 1.75)
    # ylim(; bottom=0.0)
    xticks(rs_list)
    legend(; loc="upper left")
    # legend(; loc="best", ncol=2)
    xlabel("\$r_s\$")
    ylabel("\$\\left|C^{(1)nl}_{k=0} \\,\\Big/\\, C^{(1)l}\\right|\$")
    # ylabel("\$C^{(1)} \,\left/\, (k_F a_0)^2 \right.\$")
    savefig("results/c1l_vs_c1nl/c1l_c1nl_ratio_vs_rs.pdf")
    plt.tight_layout()
    plt.close("all")

    return
end

main()