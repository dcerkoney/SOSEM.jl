using CodecZlib
using DelimitedFiles
using ElectronLiquid
using ElectronGas
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

# NOTE: Call from main project directory as: julia examples/c1d/plot_c1d_total.jl

const vzn_dir = "results/vzn_paper"

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

function load_csv(filename)
    # assumes csv format: (x, y)
    d = readdlm(filename, ',')
    @assert ndims(d) == 2
    xdata = d[:, 1]
    ydata = d[:, 2]
    return xdata, ydata
end

function average(filename)
    # assumes csv format: (x, y)
    d = readdlm(filename, ',')
    @assert ndims(d) == 2
    ydata = d[:, 2]
    return sum(ydata) / length(ydata)
end

# function spline(x, y, e)
#     # generate knots with spline without constraints
#     w = 1.0 ./ e
#     spl = interp.UnivariateSpline(x, y; w=w, k=3)
#     __x = collect(LinRange(x[1], x[end], 100))
#     yfit = spl(__x)
#     return __x, yfit
# end

function spline(
    x,
    y,
    e;
    navg_low=2,
    ncut_low=5,
    kderiv_end=0.06434,
    left_bc=true,
    right_bc=true,
)
    # yfit = signal.savgol_filter(y, 5, 3)
    w = 1.0 ./ e
    @assert ncut_low > navg_low
    @assert x[ncut_low] > 0.01
    _x, _y = deepcopy(x[(ncut_low):end]), deepcopy(y[(ncut_low):end])
    _w = 1.0 ./ e[(ncut_low):end]

    #enforce left boundary condition: the derivative at k=0 is zero
    if left_bc
        rescale = 10000
        pushfirst!(_x, 0.01)
        pushfirst!(_x, 0.0)
        yavr = sum(y[1:navg_low] .* w[1:navg_low]) / sum(w[1:navg_low])
        pushfirst!(_y, yavr)
        pushfirst!(_y, yavr)
        pushfirst!(_w, _w[1] * rescale)
        pushfirst!(_w, _w[1] * rescale)
    end

    #enforce right boundary condition: the derivative at max k is kderiv_end
    if right_bc
        dx = 0.01
        rescale = 1000
        xmax = x[end]
        _w[end] *= rescale
        # push!(_x, xmax)
        # push!(_y, _y[end])
        # push!(_w, _w[end] * rescale)
        push!(_x, xmax + dx)
        push!(_y, _y[end] + kderiv_end * dx)
        push!(_w, _w[end] * rescale)
    end

    # generate knots with spline without constraints
    spl = interp.UnivariateSpline(_x, _y; w=_w, k=3)
    __x = collect(LinRange(0.0, x[end], 100))
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

    test_signflip = false
    # test_signflip = true

    rs = 1.0
    beta = 40.0
    mass2 = 1.0
    solver = :vegasmc
    expand_bare_interactions = 0

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    n_min = 2  # True minimal loop order for this observable
    min_order_plot = 2
    # max_order_plot = 3
    max_order_plot = 5

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions == 2
        intn_str = "no_bare_"
    elseif expand_bare_interactions == 1
        intn_str = "one_bare_"
    end
    V_str = expand_bare_interactions == 0 ? "V" : "V_\\lambda"

    # Setup plot styles
    style = PyPlot.matplotlib."style"
    style.use(["science", "std-colors"])
    color = [
        cdict["orange"],
        cdict["blue"],
        cdict["cyan"],
        cdict["magenta"],
        cdict["red"],
        cdict["teal"],
    ]
    # color = [cdict["blue"], cdict["orange"], "green", cdict["red"], "black"]
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")

    # Use LaTex fonts for plots
    rcParams["font.size"] = 16
    rcParams["mathtext.fontset"] = "cm"
    # rcParams["font.family"] = "Times New Roman"

    figure(; figsize=(6, 4))

    # colors = ["orchid", "cornflowerblue", "turquoise", "chartreuse", "greenyellow"]
    # markers = ["-", "-", "-", "-", "-"]

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
        (sosem_lo.get("rpa_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d")) / eTF_lo^2
    c1nl_rpa_fl =
        (sosem_lo.get("rpa+fl_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d")) /
        eTF_lo^2
    # RPA(+FL) means are error bars
    c1nl_rpa_means, c1nl_rpa_stdevs =
        Measurements.value.(c1nl_rpa), Measurements.uncertainty.(c1nl_rpa)
    c1nl_rpa_fl_means, c1nl_rpa_fl_stdevs =
        Measurements.value.(c1nl_rpa_fl), Measurements.uncertainty.(c1nl_rpa_fl)

    # N = 3, 4
    neval34 = 5e10

    # N = 5
    neval5_c1b0 = 5e10
    neval5_c1b = 5e10
    neval5_c1c = 5e10
    neval5_c1d = 5e10
    neval5 = min(neval5_c1b0, neval5_c1b, neval5_c1c, neval5_c1d)
    max_neval5 = max(neval5_c1b0, neval5_c1b, neval5_c1c, neval5_c1d)

    # Filename for new JLD2 format
    filename =
        "results/data/processed/rs=$(rs)/rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(mass2)_$(intn_str)$(solver)_with_ct_mu_lambda_archive1_processed_data"
    # "lambda=$(mass2)_$(intn_str)$(solver)_with_ct_mu_lambda_archive1"
    #     "lambda=$(mass2)_$(intn_str)$(solver)_with_ct_mu_lambda"
    linestyles = ["--", "-"]

    # UEG param at rs = 5 for VZN SOSEM plots
    rs_vzn = 5.0
    vzn_param = UEG.ParaMC(; rs=5.0, beta=40.0, isDynamic=false)

    # Load QMC local moment
    c1l_qmc_over_EF2 = average("$vzn_dir/c1_local_qmc.csv")
    println("C⁽¹⁾ˡ (QMC): $c1l_qmc_over_EF2")

    # Load full SOSEM data in HF and OB-QMC approximations
    k_kf_grid_hf, c1_hf_over_EF2 = load_csv("$vzn_dir/c1_hf.csv")
    k_kf_grid_qmc, c1_qmc_over_EF2 = load_csv("$vzn_dir/c1_ob-qmc.csv")
    println("C⁽¹⁾ (HF)\n: $c1_hf_over_EF2")
    println("C⁽¹⁾ (QMC)\n: $c1_qmc_over_EF2")

    # Subtract local contribution to obtain HF/QMC non-local moments
    # NOTE: VZN define C⁽¹⁾(HF) as the sum of the HF non-local moment,
    #       and the OB-QMC local moment (since C⁽¹⁾ˡ(HF) is divergent)
    c1nl_qmc_over_EF2 = c1_qmc_over_EF2 .- c1l_qmc_over_EF2
    c1nl_hf_over_EF2 = c1_hf_over_EF2 .- c1l_qmc_over_EF2

    println("C⁽¹⁾ⁿˡ (HF)\n: $c1nl_hf_over_EF2")
    println("C⁽¹⁾ⁿˡ (QMC)\n: $c1nl_qmc_over_EF2")

    hf_deriv_large_k =
        (c1nl_lo[end] - c1nl_lo[end - 1]) / (k_kf_grid_quad[end] - k_kf_grid_quad[end - 1])
    rpa_deriv_large_k =
        (c1nl_rpa_means[end] - c1nl_rpa_means[end - 1]) /
        (k_kf_grid_quad[end] - k_kf_grid_quad[end - 1])
    rpa_fl_deriv_large_k =
        (c1nl_rpa_fl_means[end] - c1nl_rpa_fl_means[end - 1]) /
        (k_kf_grid_quad[end] - k_kf_grid_quad[end - 1])
    println("\nC⁽¹⁾ⁿˡ (HF) large-k derivative\n: $hf_deriv_large_k")
    println("C⁽¹⁾ⁿˡ (RPA) large-k derivative\n: $rpa_deriv_large_k")
    println("C⁽¹⁾ⁿˡ (RPA+FL) large-k derivative\n: $rpa_fl_deriv_large_k")

    hf_deriv_small_k = (c1nl_lo[2] - c1nl_lo[1]) / (k_kf_grid_quad[2] - k_kf_grid_quad[1])
    rpa_deriv_small_k =
        (c1nl_rpa_means[2] - c1nl_rpa_means[1]) / (k_kf_grid_quad[2] - k_kf_grid_quad[1])
    rpa_fl_deriv_small_k =
        (c1nl_rpa_fl_means[2] - c1nl_rpa_fl_means[1]) /
        (k_kf_grid_quad[2] - k_kf_grid_quad[1])
    println("\nC⁽¹⁾ⁿˡ (HF) small-k derivative\n: $hf_deriv_small_k")
    println("C⁽¹⁾ⁿˡ (RPA) small-k derivative\n: $rpa_deriv_small_k")
    println("C⁽¹⁾ⁿˡ (RPA-FL) small-k derivative\n: $rpa_fl_deriv_small_k")

    # Change from units of eF^2 to eTF^2
    eTF = vzn_param.qTF^2 / (2 * vzn_param.me)
    c1nl_qmc_over_eTF2 = c1nl_qmc_over_EF2 * (vzn_param.EF / eTF)^2
    c1nl_hf_over_eTF2 = c1nl_hf_over_EF2 * (vzn_param.EF / eTF)^2

    # Plot the results for each order ξ and compare to RPA(+FL)
    sign_c1b = test_signflip ? -1 : 1
    factor_c1b0 = test_signflip ? 2 : 1
    for (i, N) in enumerate(min_order_plot:max_order_plot)
        # Load the data for each observable
        d = load_old_data("$filename.jld2")
        if N == 5
            param = d["c1d/N=5/neval=$neval5_c1d/param"]
            kgrid = d["c1d/N=5/neval=$neval5_c1d/kgrid"]
            c1nl_N_total =
                d["c1b0/N=5/neval=$neval5_c1b0/meas"] +
                sign_c1b * d["c1b/N=5/neval=$neval5_c1b/meas"] +
                d["c1c/N=5/neval=$neval5_c1c/meas"] +
                d["c1d/N=5/neval=$neval5_c1d/meas"]
        elseif N == 2
            param = d["c1d/N=$N/neval=$neval34/param"]
            kgrid = d["c1d/N=$N/neval=$neval34/kgrid"]
            c1nl_N_total =
                d["c1b0/N=$N/neval=$neval34/meas"] +
                d["c1c/N=$N/neval=$neval34/meas"] +
                d["c1d/N=$N/neval=$neval34/meas"]
        else
            param = d["c1d/N=$N/neval=$neval34/param"]
            kgrid = d["c1d/N=$N/neval=$neval34/kgrid"]
            c1nl_N_total =
                d["c1b0/N=$N/neval=$neval34/meas"] +
                sign_c1b * d["c1b/N=$N/neval=$neval34/meas"] +
                d["c1c/N=$N/neval=$neval34/meas"] +
                d["c1d/N=$N/neval=$neval34/meas"]
        end

        # Get dimensionless k-grid (k / kF)
        k_kf_grid = kgrid / param.kF

        # Get means and error bars from the result up to this order
        c1nl_N_means, c1nl_N_stdevs =
            Measurements.value.(c1nl_N_total), Measurements.uncertainty.(c1nl_N_total)
        @assert length(k_kf_grid) == length(c1nl_N_means) == length(c1nl_N_stdevs)

        if N == 4
            if N == 1
                c1l_N = measurement("0.913 ± 0.0017")
            elseif N == 2
                c1l_N = measurement("1.0639 ± 0.0019")
            elseif N == 3
                c1l_N = measurement("1.1154 ± 0.0028")
            elseif N == 4
                c1l_N = measurement("1.1369 ± 0.0073")
            end
            c1_nl_l_ratio = abs(c1nl_N_total[1] / c1l_N)
            # println("C⁽¹⁾ⁿˡ(k = 0) (N=$N)\n: $(c1nl_N_total[1])")
            # println("C⁽¹⁾ˡ(k = 0) (N=$N)\n: $c1l_N")
            # println("|C⁽¹⁾ⁿˡ(k = 0) / C⁽¹⁾ˡ| (N=$N)\n: $c1_nl_l_ratio")
        end

        println(" • N = $N:\t$(c1nl_N_total[1])")
        if i == 1
            plot(
                k_kf_grid_quad,
                c1nl_rpa_means,
                "gray";
                linestyle="-",
                label="RPA",
                zorder=0,
            )
            plot(k_kf_grid_quad, c1nl_rpa_fl_means, "k"; label="RPA\$+\$FL", zorder=1)
            plot(
                k_kf_grid_quad,
                c1nl_lo,
                color[i];
                linestyle="-",
                label="\$N = $N\$",
                zorder=2,
            )
            # plot(k_kf_grid_quad, c1nl_lo, color[i]; linestyle="-", label="HF")
        else
            errorbar(
                k_kf_grid,
                c1nl_N_means;
                yerr=c1nl_N_stdevs,
                color=color[i],
                capsize=2,
                markersize=2,
                fmt="o",
                markerfacecolor="none",
                label="\$N = $N\$",
                zorder=10 * i + 3,
            )
            # _x, _y = spline_dk0_zero(k_kf_grid, c1nl_N_means, c1nl_N_stdevs; klow=0.5)
            kderiv_end = (rpa_deriv_large_k + rpa_fl_deriv_large_k) / 2
            _x, _y = spline(
                k_kf_grid,
                c1nl_N_means,
                c1nl_N_stdevs;
                navg_low=N == 5 ? 3 : 1,
                ncut_low=N == 5 ? 7 : 2,
                # kderiv_end=rpa_fl_deriv_large_k,
                # kderiv_end=rpa_deriv_large_k,
                kderiv_end=kderiv_end,
                left_bc=true,
                right_bc=true,
            )
            plot(_x, _y; color=color[i], linestyle="--")
        end
        xlim(minimum(k_kf_grid), maximum(k_kf_grid))
    end
    # # Compare with VZN data at rs = 5
    # plot(
    #     k_kf_grid_hf,
    #     c1nl_hf_over_eTF2,
    #     "--";
    #     color="r",
    #     markersize=2,
    #     label="HF (\$r_s = 5\$)",
    # )
    # plot(
    #     k_kf_grid_qmc,
    #     c1nl_qmc_over_eTF2,
    #     "-";
    #     color="r",
    #     markersize=2,
    #     label="QMC (\$r_s = 5\$)",
    # )
    # xlim(minimum(k_kf_grid), 2.0)
    if test_signflip == false
        ylim(; top=-0.195)
    end
    legend(; loc="lower right")
    # legend(; loc="best")
    xlabel("\$k / k_F\$")
    ylabel(
        "\$C^{(1)nl}_{V,$V_str}(k) \\,/\\, {\\epsilon}^{\\hspace{0.1em}2}_{\\mathrm{TF}}\$",
    )
    xloc = 0.2
    yloc = -0.25
    ydiv = -0.05
    text(
        xloc,
        yloc,
        "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
        fontsize=12,
    )
    text(
        xloc,
        yloc + ydiv,
        "\$\\lambda = $(mass2)\\epsilon_{\\mathrm{Ry}},\\, N_{\\mathrm{eval}} = \\mathrm{$(max_neval5)}\$";
        # "\$\\lambda = \\frac{\\epsilon_{\\mathrm{Ry}}}{10},\\, N_{\\mathrm{eval}} = \\mathrm{$(max_neval5)},\$";
        fontsize=12,
    )
    # text(
    #     xloc,
    #     yloc + 2 * ydiv,
    #     "\${\\epsilon}_{\\mathrm{TF}}\\equiv\\frac{\\hbar^2 q^2_{\\mathrm{TF}}}{2 m_e}=2\\pi\\mathcal{N}_F\$ (a.u.)";
    #     fontsize=10,
    # )
    # plt.title("Using fixed bare Coulomb interactions \$V_1\$, \$V_2\$")
    # plt.title(
    #     "Using re-expanded Coulomb interactions \$V_1[V_\\lambda]\$, \$V_2[V_\\lambda]\$",
    # )
    PyPlot.tight_layout()
    signflip_str = test_signflip ? "_signflip" : ""
    savefig(
        # "results/c1nl/c1nl_N=4_rs=$(rs)_" *
        "results/c1nl/c1nl_N=$(max_order_plot)_rs=$(rs)_" *
        "beta_ef=$(beta)_lambda=$(mass2)_" *
        # "neval=$(max_neval5)_$(intn_str)$(solver)_total.pdf",
        "neval=$(max_neval5)_$(intn_str)$(solver)_total$(signflip_str).pdf",
        # "neval=$(max_neval5)_$(intn_str)$(solver)_total_conjectured.pdf",
    )
    plt.close("all")
    return
end

main()
