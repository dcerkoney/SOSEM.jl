using CodecZlib
using DelimitedFiles
using ElectronLiquid
using ElectronGas
using GreenFunc
using Interpolations
using JLD2
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory

const vzn_dir = "results/vzn_paper"

function load_csv(filename)
    # assumes csv format: (x, y)
    d = readdlm(filename, ',')
    @assert ndims(d) == 2
    xdata = d[:, 1]
    ydata = d[:, 2]
    return xdata, ydata
end

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
    expand_bare_interactions = false

    neval_c1l = 1e10
    neval_c1b0 = 5e10
    neval_c1b = 5e10
    neval_c1c = 5e10
    neval_c1d = 5e10
    # neval_c1b0 = 3e10
    # neval_c1b = 1e10
    # neval_c1c = 1e10
    # neval_c1d = 1e10
    neval = max(neval_c1b0, neval_c1b, neval_c1c, neval_c1d)

    # Use SOSEM data to max order N calculated in RPT
    N = 5

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # colors = ["orchid", "cornflowerblue", "turquoise", "chartreuse", "greenyellow"]
    # markers = ["-", "-", "-", "-", "-"]

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions
        intn_str = "no_bare_"
    end

    # Filename for new JLD2 format
    filename =
        "results/data/rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(mass2)_$(intn_str)$(solver)_with_ct_mu_lambda"

    # Load SOSEM data
    local param, kgrid, k_kf_grid, c1l_N_mean, c1l_N_total, c1nl_N_total
    # Load the data for each observable
    f = jldopen("$filename.jld2", "r")
    param = f["c1d/N=$N/neval=$neval_c1d/param"]
    kgrid = f["c1d/N=$N/neval=$neval_c1d/kgrid"]
    c1l_N_total = f["c1l/N=$N/neval=$neval_c1l/meas"]
    if N == 2
        c1nl_N_total =
            f["c1b0/N=$N/neval=$neval_c1b0/meas"] +
            f["c1c/N=$N/neval=$neval_c1c/meas"] +
            f["c1d/N=$N/neval=$neval_c1d/meas"]
    else
        c1nl_N_total =
            f["c1b0/N=$N/neval=$neval_c1b0/meas"] +
            f["c1b/N=$N/neval=$neval_c1b/meas"] +
            f["c1c/N=$N/neval=$neval_c1c/meas"] +
            f["c1d/N=$N/neval=$neval_c1d/meas"]
    end
    close(f)  # close file

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param.kF

    # Get means and error bars from the result up to this order
    c1l_N_mean, c1l_N_stdev =
        Measurements.value.(c1l_N_total), Measurements.uncertainty.(c1l_N_total)
    c1nl_N_means, c1nl_N_stdevs =
        Measurements.value.(c1nl_N_total), Measurements.uncertainty.(c1nl_N_total)
    @assert length(k_kf_grid) == length(c1nl_N_means) == length(c1nl_N_stdevs)

    println(param)
    println(kgrid)
    println(c1l_N_mean)
    println(c1l_N_stdev)
    println(c1nl_N_means)
    println(c1nl_N_stdevs)

    # Get the G0W0 self-energy and corresponding DLR grid from the ElectronGas package
    sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(param.basic; minK = 1e-8 * param.kF, maxK = 30 * param.kF)
    # sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(param.basic)
    sigma_dlr_dynamic = to_dlr(sigma_tau_dynamic)
    dlr = sigma_dlr_dynamic.mesh[1].dlr
    kgrid_g0w0 = sigma_dlr_dynamic.mesh[2]
    println(kgrid_g0w0)

    # Grid indices where k ≈ kF
    ikF = searchsortedfirst(k_kf_grid, 1.0)
    i2kF = searchsortedfirst(k_kf_grid, 2.0)
    ikF_g0w0 = searchsortedfirst(kgrid_g0w0, param.kF)
    i2kF_g0w0 = searchsortedfirst(kgrid_g0w0, 2param.kF)
    println(k_kf_grid[ikF - 1], " ", k_kf_grid[ikF])
    println(k_kf_grid[i2kF - 1], " ", k_kf_grid[i2kF])
    println(kgrid_g0w0[ikF - 1] / param.kF, " ", kgrid_g0w0[ikF] / param.kF)
    println(kgrid_g0w0[i2kF - 1] / param.kF, " ", kgrid_g0w0[i2kF] / param.kF)
    # Get the static and dynamic components of the self-energy
    sigma_g0w0_static = sigma_tau_instant[1, :]
    # sigma_g0w0_wn_dynamic = (param.β / 2π) * to_imfreq(sigma_dlr_dynamic)
    # sigma_g0w0_wn_dynamic = (2π / param.β) * to_imfreq(sigma_dlr_dynamic)
    # sigma_g0w0_wn_dynamic = ((2π)^3 / param.β) * to_imfreq(sigma_dlr_dynamic)
    # sigma_g0w0_wn_dynamic = (2π)^3 * to_imfreq(sigma_dlr_dynamic)
    # sigma_g0w0_wn_dynamic = param.β * to_imfreq(sigma_dlr_dynamic)
    sigma_g0w0_wn_dynamic = to_imfreq(sigma_dlr_dynamic)

    # The static part is real, so we can discard it for the present tail fit
    # println(imag(sigma_g0w0_static)[abs.(imag(sigma_g0w0_static)) .≥ 1e-8])
    # @assert all(abs.(imag(sigma_g0w0_static)) .≤ 1e-8)
    # @assert all(abs.(imag(sigma_g0w0_static)) .≤ 1e-10)

    # Positive Matsubara frequencies
    wns = dlr.ωn[dlr.ωn .≥ 0]

    # Self-energies at k = 0 and k = kF
    sigma_g0w0_wn_dynamic_k0 = sigma_g0w0_wn_dynamic[:, 1][dlr.ωn .≥ 0]
    sigma_g0w0_wn_dynamic_kF = sigma_g0w0_wn_dynamic[:, ikF_g0w0][dlr.ωn .≥ 0]
    sigma_g0w0_wn_dynamic_2kF = sigma_g0w0_wn_dynamic[:, i2kF_g0w0][dlr.ωn .≥ 0]

    # Put back dimensions in C⁽¹⁾
    eTF2 = param.qTF^4 / (2 * param.me)^2

    @assert kgrid[1] == 0.0
    @assert kgrid[ikF] ≈ param.kF

    # The high-frequency tail of ImΣ(iωₙ) is -C⁽¹⁾ / ωₙ
    tail_fit_k0 = -eTF2 * (c1l_N_mean + c1nl_N_means[1]) ./ wns
    tail_fit_kF = -eTF2 * (c1l_N_mean + c1nl_N_means[ikF]) ./ wns
    tail_fit_2kF = -eTF2 * (c1l_N_mean + c1nl_N_means[i2kF]) ./ wns
    tail_fit_k0_err = eTF2 * (c1l_N_stdev + c1nl_N_stdevs[1]) ./ wns
    tail_fit_kF_err = eTF2 * (c1l_N_stdev + c1nl_N_stdevs[ikF]) ./ wns
    tail_fit_2kF_err = eTF2 * (c1l_N_stdev + c1nl_N_stdevs[i2kF]) ./ wns

    # Get RPA value of the local moment at this rs
    k_kf_grid_rpa, c1l_rpa_over_rs2 = load_csv("$vzn_dir/c1l_over_rs2_rpa.csv")
    P = sortperm(k_kf_grid_rpa)
    c1l_rpa_over_rs2_interp =
        linear_interpolation(k_kf_grid_rpa[P], c1l_rpa_over_rs2[P]; extrapolation_bc=Line())
    eTF = param.qTF^2 / (2 * param.me)
    c1l_rpa = c1l_rpa_over_rs2_interp(param.rs) * param.rs^2
    c1l_rpa_over_eTF2 = c1l_rpa * (param.EF / eTF)^2
    println("C⁽¹⁾ˡ (RPA, rs = $(param.rs)): $c1l_rpa")
    println("C⁽¹⁾ˡ / eTF² (RPA, rs = $(param.rs)): $c1l_rpa_over_eTF2")

    # # Non-dimensionalize bare and RPA+FL non-local moments
    sosem_lo = np.load("results/data/soms_rs=$(rs)_beta_ef=40.0.npz")
    # rs_lo = rs
    # sosem_lo = np.load("results/data/soms_rs=$(rs_lo)_beta_ef=40.0.npz")
    # # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    # param_lo = Parameter.atomicUnit(0, rs_lo)    # (dimensionless T, rs)
    # eTF_lo = param_lo.qTF^2 / (2 * param_lo.me)

    # Bare and RPA(+FL) results (stored in Hartree a.u.)
    k_kf_grid_quad = np.linspace(0.0, 3.0; num=600)
    ikF_quad = searchsortedfirst(k_kf_grid_quad, 1.0)
    i2kF_quad = searchsortedfirst(k_kf_grid_quad, 2.0)
    c1nl_lo =
        (sosem_lo.get("bare_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d")) / eTF^2
    c1nl_rpa =
        (sosem_lo.get("rpa_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d")) / eTF^2
    c1nl_rpa_fl =
        (sosem_lo.get("rpa+fl_b") + sosem_lo.get("bare_c") + sosem_lo.get("bare_d")) / eTF^2
    # RPA(+FL) means are error bars
    c1nl_rpa_means, c1nl_rpa_stdevs =
        Measurements.value.(c1nl_rpa), Measurements.uncertainty.(c1nl_rpa)

    # Tail fit using RPA results
    rpa_tail_fit_k0 = -eTF^2 * (c1l_rpa_over_eTF2 + c1nl_rpa_means[1]) ./ wns / 2
    rpa_tail_fit_k0_err = -eTF^2 * c1nl_rpa_stdevs[1] ./ wns / 2
    rpa_tail_fit_kF = -eTF^2 * (c1l_rpa_over_eTF2 + c1nl_rpa_means[ikF_quad]) ./ wns / 2
    rpa_tail_fit_kF_err = -eTF^2 * c1nl_rpa_stdevs[ikF_quad] ./ wns / 2
    rpa_tail_fit_2kF = -eTF^2 * (c1l_rpa_over_eTF2 + c1nl_rpa_means[i2kF_quad]) ./ wns / 2
    rpa_tail_fit_2kF_err = -eTF^2 * c1nl_rpa_stdevs[i2kF_quad] ./ wns / 2

    println()
    println(c1l_N_mean)
    println(c1nl_N_means[ikF])

    # println()
    # println(c1l_N_mean + c1nl_N_means[1])
    # println(c1l_N_mean + c1nl_N_means[ikF])

    # k = 0
    println()
    println(imag(sigma_g0w0_wn_dynamic_k0)[end])
    println(tail_fit_k0[end])

    println()
    println(imag(sigma_g0w0_wn_dynamic_k0) ./ tail_fit_k0)
    println(imag(sigma_g0w0_wn_dynamic_k0) ./ rpa_tail_fit_k0)

    # k = kF
    println()
    println(imag(sigma_g0w0_wn_dynamic_kF)[end])
    println(tail_fit_kF[end])

    println()
    println(imag(sigma_g0w0_wn_dynamic_kF) ./ tail_fit_kF)
    println(imag(sigma_g0w0_wn_dynamic_kF) ./ rpa_tail_fit_kF)

    # k = 2kF
    println()
    println(imag(sigma_g0w0_wn_dynamic_2kF)[end])
    println(tail_fit_2kF[end])

    println()
    println(imag(sigma_g0w0_wn_dynamic_2kF) ./ tail_fit_2kF)
    println(imag(sigma_g0w0_wn_dynamic_2kF) ./ rpa_tail_fit_2kF)

    # Compare ImΣ_{G0W0}(iωₙ) and -C⁽¹⁾_N / ωₙ at k = 2kF
    fig, ax = plt.subplots()
    # k = 2kF
    # ax.plot(wns / param.EF, param.EF * imag(sigma_g0w0_wn_dynamic_kF), "k"; label="\$RPA\$")
    ax.plot(wns / param.EF, imag(sigma_g0w0_wn_dynamic_2kF), "k"; label="\$RPA\$")
    ax.plot(
        wns / param.EF,
        rpa_tail_fit_2kF,
        "C0";
        # linestyle="--",
        label="\$-C^{(1)}_{RPA} / \\omega_n\$",
    )
    ax.fill_between(
        wns / param.EF,
        rpa_tail_fit_2kF - rpa_tail_fit_2kF_err,
        rpa_tail_fit_2kF + rpa_tail_fit_2kF_err;
        color="C0",
        alpha=0.4,
    )
    # ax.plot(wns / param.EF, sigma_g0w0_dynamic_kF, "C1"; label="\$k = k_F (RPA)\$")
    ax.plot(
        wns / param.EF,
        tail_fit_2kF,
        "C1";
        # linestyle="--",
        label="\$-C^{(1)}_N / \\omega_n\$",
    )
    ax.fill_between(
        wns / param.EF,
        tail_fit_2kF - tail_fit_2kF_err,
        tail_fit_2kF + tail_fit_2kF_err;
        color="C1",
        alpha=0.4,
    )
    # ax.plot(wns / param.EF, sigma_g0w0_dynamic_kF, "C1"; label="\$k = k_F (RPA)\$")
    # ax.set_xlim(0, 200)
    # ax.set_xlim(0, maximum(wns) / param.EF)
    ax.set_xlim(0, 100)
    # ax.set_xlim(10, 500)
    # ax.set_ylim(; bottom=-0.003, top=0)
    ax.set_ylim(; bottom=-0.125, top=0)
    ax.legend(; loc="lower right")
    ax.set_xlabel("\$\\omega_n / \\epsilon_F\$")
    ax.set_ylabel("\$\\mathrm{Im}\\Sigma(k = 2k_F, i\\omega_n)\$")
    plt.tight_layout()
    fig.savefig(
        "results/high_frequency_tail/sigma_g0w0_tail_comparison_" *
        "N=$(N)_rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(mass2)_neval=$(neval)_$(solver)_k=2kF.pdf",
    )
    plt.close("all")

    # Compare ImΣ_{G0W0}(iωₙ) and -C⁽¹⁾_N / ωₙ at k = kF
    fig, ax = plt.subplots()
    # k = kF
    # ax.plot(wns / param.EF, param.EF * imag(sigma_g0w0_wn_dynamic_kF), "k"; label="\$RPA\$")
    ax.plot(wns / param.EF, imag(sigma_g0w0_wn_dynamic_kF), "k"; label="\$RPA\$")
    ax.plot(
        wns / param.EF,
        rpa_tail_fit_kF,
        "C0";
        # linestyle="--",
        label="\$-C^{(1)}_{RPA} / \\omega_n\$",
    )
    ax.fill_between(
        wns / param.EF,
        rpa_tail_fit_kF - rpa_tail_fit_kF_err,
        rpa_tail_fit_kF + rpa_tail_fit_kF_err;
        color="C0",
        alpha=0.4,
    )
    # ax.plot(wns / param.EF, sigma_g0w0_dynamic_kF, "C1"; label="\$k = k_F (RPA)\$")
    ax.plot(
        wns / param.EF,
        tail_fit_kF,
        "C1";
        # linestyle="--",
        label="\$-C^{(1)}_N / \\omega_n\$",
    )
    ax.fill_between(
        wns / param.EF,
        tail_fit_kF - tail_fit_kF_err,
        tail_fit_kF + tail_fit_kF_err;
        color="C1",
        alpha=0.4,
    )
    # ax.plot(wns / param.EF, sigma_g0w0_dynamic_kF, "C1"; label="\$k = k_F (RPA)\$")
    # ax.set_xlim(0, 200)
    # ax.set_xlim(0, maximum(wns) / param.EF)
    ax.set_xlim(0, 100)
    # ax.set_xlim(10, 500)
    # ax.set_ylim(; bottom=-0.003, top=0)
    ax.set_ylim(; bottom=-0.125, top=0)
    ax.legend(; loc="lower right")
    ax.set_xlabel("\$\\omega_n / \\epsilon_F\$")
    ax.set_ylabel("\$\\mathrm{Im}\\Sigma(k = k_F, i\\omega_n)\$")
    plt.tight_layout()
    fig.savefig(
        "results/high_frequency_tail/sigma_g0w0_tail_comparison_" *
        "N=$(N)_rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(mass2)_neval=$(neval)_$(solver)_k=kF.pdf",
    )
    plt.close("all")

    # Compare ImΣ_{G0W0}(iωₙ) and -C⁽¹⁾_N / ωₙ at k = 0
    fig, ax = plt.subplots()
    # k = 0
    ax.plot(wns / param.EF, imag(sigma_g0w0_wn_dynamic_k0), "k"; label="\$RPA\$")
    ax.plot(
        wns / param.EF,
        rpa_tail_fit_k0,
        "C0";
        # linestyle="--",
        label="\$-C^{(1)}_{RPA} / \\omega_n\$",
    )
    ax.fill_between(
        wns / param.EF,
        rpa_tail_fit_k0 - rpa_tail_fit_k0_err,
        rpa_tail_fit_k0 + rpa_tail_fit_k0_err;
        color="C0",
        alpha=0.4,
    )
    ax.plot(
        wns / param.EF,
        tail_fit_k0,
        "C1";
        # linestyle="--",
        label="\$-C^{(1)}_N / \\omega_n\$",
    )
    ax.fill_between(
        wns / param.EF,
        tail_fit_k0 - tail_fit_k0_err,
        tail_fit_k0 + tail_fit_k0_err;
        color="C1",
        alpha=0.4,
    )
    # ax.plot(wns / param.EF, sigma_g0w0_dynamic_kF, "C1"; label="\$k = k_F (RPA)\$")
    # ax.set_xlim(0, 200)
    # ax.set_xlim(0, maximum(wns) / param.EF)
    ax.set_xlim(0, 100)
    # ax.set_xlim(10, 500)
    # ax.set_ylim(; bottom=-0.003, top=0)
    ax.set_ylim(; bottom=-0.125, top=0)
    ax.legend(; loc="lower right")
    ax.set_xlabel("\$\\omega_n / \\epsilon_F\$")
    ax.set_ylabel("\$\\mathrm{Im}\\Sigma(k = 0, i\\omega_n)\$")
    plt.tight_layout()
    fig.savefig(
        "results/high_frequency_tail/sigma_g0w0_tail_comparison_" *
        "N=$(N)_rs=$(rs)_beta_ef=$(beta)_" *
        "lambda=$(mass2)_neval=$(neval)_$(solver)_k=0.pdf",
    )
    plt.close("all")

    # TODO: Compare ReΣ(∞) and C⁽⁰⁾_{N - 1}
    return
end

main()
