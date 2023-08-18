using CodecZlib
using CompositeGrids
using DelimitedFiles
using ElectronLiquid
using ElectronGas
using GreenFunc
using Interpolations
using Lehmann
using JLD2
using Measurements
using Parameters
using PyCall
using SOSEM
using LsqFit

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory
const vzn_dir = "results/vzn_paper"

# Small parameter of the UEG theory
const α = (4 / 9π)^(1 / 3)

# Benchmarks for Z(k_F)
const zfactor_benchmarks = Dict(
    1.0 => 0.8601,
    2.0 => 0.7642,
    3.0 => 0.6927,
    4.0 => 0.6367,
    5.0 => 0.5913,
    6.0 => 0.5535,
)

"""@ beta = 1000, small cutoffs"""
# Nk, order = 11, 8
const rpa_sosem = Dict(
    0.01 => 133629.31338476276,
    0.1 => 996.1421229225255,
    0.5 => 30.654593391597565,
    1.0 => 6.723994594646203,
    2.0 => 1.4584858366718452,
    3.0 => 0.5936165034488587,
    4.0 => 0.3130605327419725,
    5.0 => 0.1903819378977648,
    7.5 => 0.07694747219230284,
    10.0 => 0.04040127700428145,
)
const rpa_fl_sosem_ko_takada = Dict(
    0.01 => 113120.39532239494,
    0.1 => 794.082225081879,
    0.5 => 22.706255132471114,
    1.0 => 4.739085706326042,
    2.0 => 0.958442122563793,
    3.0 => 0.3703011474216666,
    4.0 => 0.18744178767319505,
    5.0 => 0.11026002348040444,
    7.5 => 0.041904772294599595,
    10.0 => 0.02107009260794267,
)
# # Nk, order = 20, 12
# const rpa_sosem = Dict(
#     0.01 => 133629.5589712659,
#     0.1 => 996.1453748797056,
#     0.5 => 30.65472082096144,
#     1.0 => 6.724025817780453,
#     2.0 => 1.4584933290770838,
#     3.0 => 0.5936165034488587,
#     4.0 => 0.3130622525590106,
#     5.0 => 0.19038299054878535,
#     7.5 => 0.0769478880550561,
#     10.0 => 0.0404014827795728,
# )
# const rpa_fl_sosem_ko_takada = Dict(
#     0.01 => 113120.53305565231,
#     0.1 => 794.0844302101557,
#     0.5 => 22.706345661868387,
#     1.0 => 4.739108972017714,
#     2.0 => 0.9584480419440262,
#     3.0 => 0.3703011474216666,
#     4.0 => 0.18744320831614875,
#     5.0 => 0.1102609051385906,
#     7.5 => 0.04190513631989273,
#     10.0 => 0.02107028506216691,
# )

"""Given sample y and x values, compute dy/dx using the forward difference method."""
function forward_diff(ys, xs)
    dy_dx = (ys[2:end] - ys[1:(end - 1)]) ./ (xs[2:end] - xs[1:(end - 1)])
    _xs = xs[1:(end - 1)]
    return _xs, dy_dx
end

"""Given sample y and x values, compute dy/dx using the central difference method."""
function central_diff(ys, xs)
    dy_dx = (ys[3:end] - ys[1:(end - 2)]) ./ (xs[3:end] - xs[1:(end - 2)])
    _xs = xs[2:(end - 1)]
    return _xs, dy_dx
end

function rsquared(xs, ys, yhats)
    ybar = sum(yhats) / length(yhats)
    ss_res = sum((yhats .- ys) .^ 2)
    ss_tot = sum((ys .- ybar) .^ 2)
    return 1 - ss_res / ss_tot
end

# function spline(x, y, e=one.(y); npts=1000)
function spline(x, y; npts=1000, xmax=30.0)
    interp = pyimport("scipy.interpolate")
    _x = x[y .≥ 0]
    _y = y[y .≥ 0]
    _x = x[x .≤ xmax]
    _y = y[x .≤ xmax]
    # generate knots with spline without constraints
    spl = interp.CubicSpline(_x, _y)
    # spl = interp.CubicSpline(_x, _y)
    __x = collect(LinRange(_x[1], _x[end], npts))
    yfit = spl(__x)
    return __x, yfit
end

# function get_Fs(rs)
# if rs < 1.0 || rs > 5.0
#     return get_Fs_DMC(rs)
# else
#     return get_Fs_PW(rs)
# end
# end

"""
Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ from DMC data of 
Moroni, Ceperley & Senatore (1995) [Phys. Rev. Lett. 75, 689].
"""
@inline function get_Fs_DMC(rs)
    SOSEM.@todo
end

"""
Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ via interpolation of the 
compressibility ratio data of Perdew & Wang (1992) [Phys. Rev. B 45, 13244].
"""
@inline function get_Fs_PW(rs)
    # if rs < 1.0 || rs > 5.0
    #     @warn "The Perdew-Wang interpolation for Fs may " *
    #           "be inaccurate outside the metallic regime!"
    # end
    kappa0_over_kappa = 1.0025 - 0.1721rs - 0.0036rs^2
    # F⁰ₛ = κ₀/κ - 1
    return kappa0_over_kappa - 1.0
end

"""
Get the value of fₛ(q → ∞) for a given interaction type
(either RPA, or RPA+FL using a constant or Takada ansatz for fₛ(q)).
"""
@inline function get_fs_infty(para::Parameter.Para, int_type=:rpa)
    if int_type == :ko_const
        # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
        rs = round(para.rs; sigdigits=13)
        Fs = get_Fs_PW(rs)
        # Local-field factor at q=∞
        # For :ko_const, fs_infty = Fs / NF
        fs_infty = Fs / para.NF
    elseif int_type == :ko
        # For the Takada ansatz for fs(q), fs(∞) = 0,
        # so the RPA+FL tail is the same as RPA
        fs_infty = 0.0
    elseif int_type == :rpa
        fs_infty = 0.0
    else
        SOSEM.@todo
    end
    return fs_infty
end

# RPA(+FL) analytic tail behavior C ωₙ^{3/2} with coefficient C = -c (1 + fs(∞))/ √2, where
# c = 16 √2 (α rs)² / 3π is taken from the VZN paper
@inline function C_coefficient(rs, fs_infty=0.0)
    # return -16 * (1 + fs_infty) * (α * rs)^2 / 3π
    return -16 * (1 - fs_infty) * (α * rs)^2 / 3π
end

@inline function get_rpa_analytic_tail(wns, para::Parameter.Para, fs_infty=0.0)
    rs = round(para.rs; sigdigits=13)
    return @. C_coefficient(rs) / wns^(3 / 2)
end

@inline function get_rpa_fl_analytic_tail(wns, para::Parameter.Para, fs_infty=0.0)
    # NOTE: Our fs(q) has an extra minus sign relative to VZN,
    # so we have (1 + fs(∞)) C / ω^(3/2) ↦ (1 + fs(∞)) C / ω^(3/2)
    rs = round(para.rs; sigdigits=13)
    return @. C_coefficient(rs, fs_infty) / wns^(3 / 2)
end

function get_sigma_rpa_wn(para::Parameter.Para; ktarget=0.0, atol=1e-3, int_type=:rpa)
    # Make sure we are using parameters for the bare UEG theory
    @assert para.Λs == para.Λa == 0.0

    # Small params
    Euv, rtol = 1000 * para.EF, 1e-11
    maxK, minK = 30 * para.kF, 1e-8 * para.kF
    Nk, order = 11, 8
    # Nk, order = 20, 12

    # Get RPA+FL self-energy
    # sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(para; int_type=:rpa)
    sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(
        para;
        Euv=Euv,
        rtol=rtol,
        Nk=Nk,
        minK=minK,
        maxK=maxK,
        order=order,
        int_type=:rpa,
    )

    # Get sigma in DLR basis
    sigma_dlr_dynamic = to_dlr(sigma_tau_dynamic)

    # Get the static and dynamic components of the Matsubara self-energy
    sigma_wn_static = sigma_tau_instant[1, :]
    sigma_wn_dynamic = to_imfreq(sigma_dlr_dynamic)

    # Get grids
    dlr = sigma_dlr_dynamic.mesh[1].dlr
    kgrid = sigma_dlr_dynamic.mesh[2]
    wns = dlr.ωn[dlr.ωn .≥ 0]  # retain positive Matsubara frequencies only

    # Evaluate at a single kpoint near k = ktarget
    ikval = searchsortedfirst(kgrid, ktarget)
    kval = kgrid[ikval]
    if abs(kval - ktarget) > atol
        @warn "kval = $kval is not within atol = $atol of ktarget = $ktarget!"
    end
    println(
        "Obtaining self-energy at grid point k = $kval near target k-point ktarget = $ktarget",
    )

    # Self-energies at k = 0 and k = kF for positive frequencies
    sigma_wn_static_kval = sigma_wn_static[ikval]
    sigma_wn_dynamic_kval = sigma_wn_dynamic[:, ikval][dlr.ωn .≥ 0]

    # Z-factor for low-frequency parameterization
    zfactor_m10 = SelfEnergy.zfactor(para, sigma_wn_dynamic; kamp=ktarget, ngrid=[-1, 0])[1]
    zfactor_0p1 = SelfEnergy.zfactor(para, sigma_wn_dynamic; kamp=ktarget, ngrid=[0, 1])[1]
    println("Z_RPA(k=$ktarget) (ngrid = [-1, 0]): ", zfactor_m10)
    println("Z_RPA(k=$ktarget) (ngrid = [0, 1]): ", zfactor_0p1)
    # zfactor = zfactor_m10
    zfactor = zfactor_0p1

    # Check zfactor against benchmarks at β = 1000. 
    # It should agree within a few percent (up to finite-temperature effects).
    # (see: https://numericaleft.github.io/ElectronGas.jl/dev/manual/quasiparticle/)
    rs = round(para.rs; sigdigits=13)
    if rs in keys(zfactor_benchmarks) && ktarget == para.kF
        println("Z_RPA (benchmark): ", zfactor_benchmarks[round(para.rs; sigdigits=13)])
        println("Checking zfactor against zero-temperature benchmark...")
        zfactor_zero_temp = zfactor_benchmarks[rs]
        percent_error = 100 * abs(zfactor - zfactor_zero_temp) / zfactor_zero_temp
        println("Percent error vs zero-temperature benchmark of Z_kF: $percent_error")
        @assert percent_error ≤ 5
    else
        println("No zero-temperature benchmark available for rs = $(rs)!")
    end

    return dlr, kval, wns, sigma_wn_static_kval, sigma_wn_dynamic_kval, zfactor
end

function get_sigma_rpa_fl_wn(
    para::Parameter.Para;
    ktarget=0.0,
    atol=1e-3,
    int_type=int_type,
)
    # Make sure we are using parameters for the bare UEG theory
    @assert para.Λs == para.Λa == 0.0

    # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
    rs = round(para.rs; sigdigits=13)
    Fs = get_Fs_PW(rs)
    if int_type == :ko_const
        println("Fermi liquid parameter at rs = $(rs): Fs = $Fs")
    end

    # Small params
    Euv, rtol = 1000 * para.EF, 1e-11
    maxK, minK = 30 * para.kF, 1e-8 * para.kF
    Nk, order = 11, 8
    # Nk, order = 20, 12

    # Get RPA+FL self-energy
    sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(
        para;
        Euv=Euv,
        rtol=rtol,
        Nk=Nk,
        minK=minK,
        maxK=maxK,
        order=order,
        int_type=int_type,
        # Fs=-Fs,  # NOTE: NEFT uses opposite sign convention (Fs > 0)!
    )

    # Get sigma in DLR basis
    sigma_dlr_dynamic = to_dlr(sigma_tau_dynamic)

    # Get the static and dynamic components of the Matsubara self-energy
    sigma_wn_static = sigma_tau_instant[1, :]
    sigma_wn_dynamic = to_imfreq(sigma_dlr_dynamic)

    # Get grids
    dlr = sigma_dlr_dynamic.mesh[1].dlr
    kgrid = sigma_dlr_dynamic.mesh[2]
    wns = dlr.ωn[dlr.ωn .≥ 0]  # retain positive Matsubara frequencies only

    # Evaluate at a single kpoint near k = ktarget
    ikval = searchsortedfirst(kgrid, ktarget)
    kval = kgrid[ikval]
    if abs(kval - ktarget) > atol
        @warn "kval = $kval is not within atol = $atol of ktarget = $ktarget!"
    end
    println(
        "Obtaining self-energy at grid point k = $kval near target k-point ktarget = $ktarget",
    )

    # Self-energies at k = 0 and k = kF for positive frequencies
    sigma_wn_static_kval = sigma_wn_static[ikval]
    sigma_wn_dynamic_kval = sigma_wn_dynamic[:, ikval][dlr.ωn .≥ 0]

    # Z-factor for low-frequency parameterization
    zfactor_m10 = SelfEnergy.zfactor(para, sigma_wn_dynamic; kamp=ktarget, ngrid=[-1, 0])[1]
    zfactor_0p1 = SelfEnergy.zfactor(para, sigma_wn_dynamic; kamp=ktarget, ngrid=[0, 1])[1]
    println("Z_RPA(k=$ktarget) (ngrid = [-1, 0]): ", zfactor_m10)
    println("Z_RPA(k=$ktarget) (ngrid = [0, 1]): ", zfactor_0p1)
    # zfactor = zfactor_m10
    zfactor = zfactor_0p1

    return dlr, kval, wns, sigma_wn_static_kval, sigma_wn_dynamic_kval, zfactor
end

function get_c1_rpa(para::Parameter.Para)
    # Small params
    kF = para.kF
    Euv, rtol = 1000 * para.EF, 1e-11
    maxK, minK = 30 * kF, 1e-8 * kF
    # Nk, order = 20, 12
    Nk, order = 11, 8

    # qgrid for integration
    qgrid =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, maxK], [0.0, 2.0 * kF], Nk, minK, order)

    # RPA Wtilde0(q, iωₙ) / V(q)
    wtilde_0_over_v_wn_q, _ = Interaction.RPAwrapped(Euv, rtol, qgrid.grid, para)
    # Interaction.RPAwrapped(Euv, rtol, qgrid.grid, param; bugfix=true)
    @assert maximum(imag(wtilde_0_over_v_wn_q[1, 1, :])) ≤ 1e-10

    # Get Wtilde0(q, τ = 0) / V(q) from WtildeFs(q, iωₙ) / V(q)
    wtilde_0_over_v_dlr_q = to_dlr(wtilde_0_over_v_wn_q)
    wtilde_0_over_v_tau_q = to_imtime(wtilde_0_over_v_dlr_q)
    # NOTE: Wtilde0 is spin-symmetric => idx1 = 1
    wtilde_0_over_v_q_inst = real(wtilde_0_over_v_tau_q[1, 1, :])

    # Integrate RPA(+FL) 4πe²(Wtilde(q, τ = 0) / V(q)) over q ∈ ℝ⁺
    rpa_integrand = wtilde_0_over_v_q_inst
    c1_rpa = -(2 * para.e0^2 / π) * CompositeGrids.Interp.integrate1D(rpa_integrand, qgrid)

    # println("\nrs = $rs:" * "\nC⁽¹⁾_{RPA} = $c1_rpa")
    return c1_rpa
end

function get_c1_rpa_fl(para::Parameter.Para; int_type=:ko)
    # Small params
    kF = para.kF
    Euv, rtol = 1000 * para.EF, 1e-11
    maxK, minK = 30 * kF, 1e-8 * kF
    # Nk, order = 20, 12
    Nk, order = 11, 8

    if int_type != :ko
        SOSEM.@todo
    end
    # # Get Landau parameter F⁰ₛ from Perdew & Wang compressibility fit
    # Fs = get_Fs(rs)
    #     println("rs = $rs, Fs = $Fs, fs = $(Fs / para.NF)")
    # end

    # Either use constant Fs from P&W, q-dependent Takada ansatz, or Corradini fit to Moroni DMC data
    if int_type == :ko
        landaufunc = Interaction.landauParameterTakada
        # elseif int_type == :ko_const
        #     landaufunc = Interaction.landauParameterConst
        # elseif int_type == :ko_moroni
        #     landaufunc = Interaction.landauParameterMoroni
    end

    # qgrid for integration
    qgrid =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, maxK], [0.0, 2.0 * kF], Nk, minK, order)

    # Get Wtilde_KO(q, iωₙ) / (V(q) - fs)
    wtilde_KO_over_v_wn_q, _ = Interaction.KOwrapped(
        Euv,
        rtol,
        qgrid.grid,
        para;
        regular=true,
        int_type=:ko,
        # int_type=int_type == :ko_const ? :ko_const : :ko,
        landaufunc=landaufunc,
        # Fs=-Fs,  # NOTE: NEFT uses opposite sign convention!
        # bugfix=true,
    )
    @assert maximum(imag(wtilde_KO_over_v_wn_q[1, 1, :])) ≤ 1e-10

    # Get Wtilde_KO(q, τ) / (V(q) - fs) from Wtilde_KO(q, iωₙ) / (V(q) - fs)
    wtilde_KO_over_v_dlr_q = to_dlr(wtilde_KO_over_v_wn_q)
    wtilde_KO_over_v_tau_q = to_imtime(wtilde_KO_over_v_dlr_q)

    # Get Wtilde_KO_s(q, τ = 0) / (V(q) - fs), keeping only the
    # spin-symmetric part of wtilde_KO (we define fa := 0)
    wtilde_KO_s_over_v_q_f_inst = real(wtilde_KO_over_v_tau_q[1, 1, :])

    local fs_int_type
    if int_type == :ko
        # NOTE: The Takada ansatz for fs is q-dependent!
        fs_int_type = [Interaction.landauParameterTakada(q, 0, para)[1] for q in qgrid.grid]
    elseif int_type == :ko_const
        # fs = Fs / NF
        fs_int_type = Fs / para.NF
    else
        error("Not yet implemented!")
    end

    # 1 / Vs(q)
    Vinvs = [Interaction.coulombinv(q, para)[1] for q in qgrid.grid]

    # Wtilde_KO_s / Vs = (Wtilde_KO_s / (Vs - fs)) * (1 - fs Vinvs) 
    #                  = Wtilde_KO_s(...; regular=true) * (1 - fs Vinvs)
    rpa_fl_integrand = @. wtilde_KO_s_over_v_q_f_inst * (1.0 - fs_int_type * Vinvs)

    # Integrate RPA+FL 4πe²(Wtilde(q, τ = 0) / V(q)) over q ∈ ℝ⁺ (regular = true)
    c1_rpa_fl =
        -(2 * para.e0^2 / π) * CompositeGrids.Interp.integrate1D(rpa_fl_integrand, qgrid)

    # println("rs = $rs:" * "\nC⁽¹⁾_{RPA+FL} = $c1_rpa_fl")
    return c1_rpa_fl
end

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    # Physical parameters
    beta = 1000.0
    # rslist = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
    rslist = LinRange(0.01, 10.0, 30)

    # We fit Σ_RPA(k = 0, iw)
    ktarget = 0.0
    # ktarget = para.kF

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # Which fs paramaterization to use
    int_type = :ko

    rs_qmc = 1.0
    beta_qmc = 40.0
    mass2 = 1.0
    solver = :vegasmc
    expand_bare_interactions = 0

    neval_c1l = 1e10
    neval_c1b0 = 5e10
    neval_c1b = 5e10
    neval_c1c = 5e10
    neval_c1d = 5e10
    neval = max(neval_c1b0, neval_c1b, neval_c1c, neval_c1d)

    # Use SOSEM data to max order N calculated in RPT
    N = 5

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions == 2
        intn_str = "no_bare_"
    elseif expand_bare_interactions == 1
        intn_str = "one_bare_"
    end

    # # Filename for new JLD2 format
    # filename =
    # # "results/data/rs=$(rs_qmc)_beta_ef=$(beta_qmc)_" *
    #     "archives/rs=$(rs_qmc)_beta_ef=$(beta_qmc)_lambda=$(mass2)_" *
    #     "$(intn_str)$(solver)_with_ct_mu_lambda_archive1"

    # # Load SOSEM data
    # local param1, kgrid, k_kf_grid, c1l_N_mean, c1l_N_total, c1nl_N_total
    # # Load the data for each observable
    # f = jldopen("$filename.jld2", "r")
    # param1 = f["c1d/N=$N/neval=$neval_c1d/param"]
    # kgrid = f["c1d/N=$N/neval=$neval_c1d/kgrid"]
    # c1l_N_total = f["c1l/N=$N/neval=$neval_c1l/meas"][1]
    # if N == 2
    #     c1nl_N_total =
    #         f["c1b0/N=$N/neval=$neval_c1b0/meas"] +
    #         f["c1c/N=$N/neval=$neval_c1c/meas"] +
    #         f["c1d/N=$N/neval=$neval_c1d/meas"]
    # else
    #     c1nl_N_total =
    #         f["c1b0/N=$N/neval=$neval_c1b0/meas"] +
    #         f["c1b/N=$N/neval=$neval_c1b/meas"] +
    #         f["c1c/N=$N/neval=$neval_c1c/meas"] +
    #         f["c1d/N=$N/neval=$neval_c1d/meas"]
    # end
    # close(f)  # close file

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    darkcolors = ["midnightblue", "saddlebrown", "darkgreen", "darkred"]
    max_sigma = 0.0
    w0_over_EF_rpas = []
    w0_over_EF_rpa_fls = []
    c1_rpas_over_EF2 = []
    c1_rpa_fls_over_EF2 = []
    zfactors_rpa = []
    zfactors_rpa_fl = []
    sigma_rpa_peaks_over_EF = []
    sigma_rpa_fl_peaks_over_EF = []
    sigma_rpa_peak_wns_over_EF = []
    sigma_rpa_fl_peak_wns_over_EF = []
    fig, ax = plt.subplots()
    fig5, ax5 = plt.subplots()
    for (i, rs) in enumerate(rslist)
        println("\nPlotting data for rs = $rs...")

        # Get the G0W0 self-energy and corresponding DLR grid from the ElectronGas package
        # NOTE: Here we need to be careful to generate Σ_G0W0 for the *bare* theory, i.e.,
        #       to use an ElectronGas.Parameter object where Λs (mass2) is zero!
        para = Parameter.rydbergUnit(1 / beta, rs, 3)
        @assert para.Λs == para.Λa == 0.0

        # Get RPA and RPA+FL self-energies
        dlr, kval, wns, sigma_rpa_wn_stat, sigma_rpa_wn_dyn, zfactor_rpa =
            get_sigma_rpa_wn(para; ktarget=ktarget)
        dlr, kval_fl, wns_fl, sigma_rpa_fl_wn_stat, sigma_rpa_fl_wn_dyn, zfactor_rpa_fl =
            get_sigma_rpa_fl_wn(para; ktarget=ktarget, int_type=int_type)

        # The static parts do not contribute to ImΣ
        println(imag(sigma_rpa_wn_stat))
        println(imag(sigma_rpa_fl_wn_stat))
        @assert isapprox(imag(sigma_rpa_wn_stat), 0.0, atol=1e-5)
        @assert isapprox(imag(sigma_rpa_fl_wn_stat), 0.0, atol=1e-5)

        # The grids should be the same for RPA and RPA+FL
        @assert kval ≈ kval_fl
        @assert wns ≈ wns_fl

        # Energy units in Rydbergs for nondimensionalization of self-energy data
        EF = para.EF

        # Nondimensionalize frequencies and self-energies by EF
        wns_over_EF = wns / EF

        # # Static part
        # sigma_rpa_wn_stat_over_EF = sigma_rpa_wn_stat / EF
        # sigma_rpa_fl_wn_stat_over_EF = sigma_rpa_fl_wn_stat / EF

        # Dynamic part
        sigma_rpa_wn_dyn_over_EF = sigma_rpa_wn_dyn / EF
        sigma_rpa_fl_wn_dyn_over_EF = sigma_rpa_fl_wn_dyn / EF

        # # Get first-order RPA and RPA+FL moments at k = ktarget
        # rpa_c1 = rpa_sosem[rs]
        # # rpa_c1 = rpa_moments_converged[rs]
        # if int_type == :ko
        #     rpa_fl_c1 = rpa_fl_sosem_ko_takada[rs]
        #     # rpa_fl_c1 = rpa_fl_moments_ko_takada_converged[rs]
        # elseif int_type == :ko_const
        #     SOSEM.@todo
        #     # rpa_fl_c1 = rpa_fl_sosem_ko_const[rs]
        # end

        # Get first-order RPA and RPA+FL moments at k = ktarget
        rpa_c1    = get_c1_rpa(para)
        rpa_fl_c1 = get_c1_rpa_fl(para; int_type=int_type)

        push!(c1_rpas_over_EF2, rpa_c1 / EF^2)
        push!(c1_rpa_fls_over_EF2, rpa_fl_c1 / EF^2)
        println("First-order RPA moment (Rydberg): ", rpa_c1)
        println("First-order RPA+FL moment (Rydberg): ", rpa_fl_c1)

        println("(RPA) -ImΣ(0, 0) = ", -imag(sigma_rpa_wn_dyn_over_EF[1]))
        println("(RPA+FL) -ImΣ(0, 0) = ", -imag(sigma_rpa_fl_wn_dyn_over_EF[1]))

        # # Plot of RPA(+FL) -ImΣs in chosen units
        # ax.plot(
        #     wns_over_EF,
        #     -imag(sigma_rpa_wn_dyn_over_EF),
        #     "C$(i-1)";
        #     label="\$RPA\$ (\$r_s=$(rs)\$)",
        #     # label="\$RPA\$ (\$r_s=$(rsstrings[i])\$)",
        # )
        # ax.plot(
        #     wns_over_EF,
        #     -imag(sigma_rpa_fl_wn_dyn_over_EF),
        #     "$(darkcolors[i])";
        #     label="\$RPA+FL\$ (\$r_s=$(rs)\$)",
        #     # label="\$RPA+FL\$ (\$r_s=$(rsstrings[i])\$)",
        # )

        max_sigma = max(
            max_sigma,
            maximum(-imag(sigma_rpa_wn_dyn_over_EF)),
            maximum(-imag(sigma_rpa_fl_wn_dyn_over_EF)),
        )

        # High- and low-frequency tails
        coeff_high_freq_rpa = rpa_c1
        coeff_high_freq_rpa_fl = rpa_fl_c1
        coeff_low_freq_rpa = (1 / zfactor_rpa - 1)
        coeff_low_freq_rpa_fl = (1 / zfactor_rpa_fl - 1)
        # w0 is the turning point between high- and low-frequency tails
        w0_rpa = sqrt(coeff_high_freq_rpa / coeff_low_freq_rpa)
        w0_rpa_fl = sqrt(coeff_high_freq_rpa_fl / coeff_low_freq_rpa_fl)
        push!(w0_over_EF_rpas, w0_rpa / EF)
        push!(w0_over_EF_rpa_fls, w0_rpa_fl / EF)
        # Peak positions
        push!(sigma_rpa_peak_wns_over_EF, wns[argmax(-imag(sigma_rpa_wn_dyn))] / EF)
        push!(sigma_rpa_fl_peak_wns_over_EF, wns[argmax(-imag(sigma_rpa_fl_wn_dyn))] / EF)
        # Peak values
        push!(sigma_rpa_peaks_over_EF, maximum(-imag(sigma_rpa_wn_dyn) / EF))
        push!(sigma_rpa_fl_peaks_over_EF, maximum(-imag(sigma_rpa_fl_wn_dyn) / EF))
        # Z-factors
        push!(zfactors_rpa, zfactor_rpa)
        push!(zfactors_rpa_fl, zfactor_rpa_fl)
    end
    # ax.set_xlim(0, 50)
    # ax.set_ylim(; bottom=0, top=1.1 * max_sigma)
    # ax.legend(; loc="best")
    # ax.set_xlabel("\$\\omega_n\$")
    # ax.set_ylabel("\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\$")
    # plt.tight_layout()
    # fig.savefig(
    #     "results/self_energy_fits/$(int_type)/im_sigma_" *
    #     "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_$(int_type).pdf",
    # )
    # plt.close("all")

    # Fine-grained rs list for models
    rslist_big = LinRange(1e-3, 10.0, 600)

    # Plot low/high-frequency turning points vs rs
    fig7, ax7 = plt.subplots()
    # Least-squares fit to low-frequency turning points
    # @. model7(rs, p) = p[1] + p[2] * sqrt(abs(log(rs))) + p[3] * sqrt(rs) + p[4] * sqrt(rs * abs(log(rs)))
    # @. model7(rs, p) = p[1] + p[2] * log(rs) + p[3] * rs + p[4] * rs * log(rs)
    # fit_rpa = curve_fit(model7, rslist, w0_over_EF_rpas, [1.5, 1.5, 1.5, 1.5])
    # fit_rpa_fl = curve_fit(model7, rslist, w0_over_EF_rpa_fls, [1.5, 1.5, 1.5, 1.5])
    @. model7(rs, p) = sqrt(rs) * (p[1] + p[2] * log(rs))
    fit_rpa = curve_fit(model7, rslist, w0_over_EF_rpas, [1.5, 1.5])
    fit_rpa_fl = curve_fit(model7, rslist, w0_over_EF_rpa_fls, [1.5, 1.5])
    # fit_rpa = curve_fit(model7, rslist, w0_over_EF_rpas, [1.5, 0.5, 0.5, 0.25])
    # fit_rpa_fl = curve_fit(model7, rslist, w0_over_EF_rpa_fls, [1.5, 0.5, 0.5, 0.25])
    model_rpa7(rs) = model7(rs, fit_rpa.param)
    model_rpa_fl7(rs) = model7(rs, fit_rpa_fl.param)
    # Coefficients of determination (r²)
    r2_rpa = rsquared(rslist, w0_over_EF_rpas, model_rpa7.(rslist))
    r2_rpa_fl = rsquared(rslist, w0_over_EF_rpa_fls, model_rpa_fl7.(rslist))
    println("RPA fit: ", fit_rpa.param, ", r² = $r2_rpa")
    println("RPA+FL fit: ", fit_rpa_fl.param, ", r² = $r2_rpa_fl")
    # a_rpa, b_rpa, c_rpa, d_rpa = fit_rpa.param
    # a_rpa_fl, b_rpa_fl, c_rpa_fl, d_rpa_fl = fit_rpa_fl.param
    a_rpa, b_rpa = fit_rpa.param
    a_rpa_fl, b_rpa_fl = fit_rpa_fl.param
    sgn_b_rpa = b_rpa ≥ 0 ? "+" : "-"
    sgn_b_rpa_fl = b_rpa_fl ≥ 0 ? "+" : "-"
    # sgn_c_rpa = c_rpa ≥ 0 ? "+" : "-"
    # sgn_c_rpa_fl = c_rpa_fl ≥ 0 ? "+" : "-"
    # sgn_d_rpa = d_rpa ≥ 0 ? "+" : "-"
    # sgn_d_rpa_fl = d_rpa_fl ≥ 0 ? "+" : "-"
    ax7.plot(rslist, w0_over_EF_rpas, "o-"; color="C0", label="\$RPA\$")
    # ax7.plot(rslist, w0_over_EF_rpas ./ sqrt.(rslist), "o-"; color="C0", label="\$RPA\$")
    ax7.plot(
        rslist_big,
        model_rpa7.(rslist_big),
        # model_rpa7.(rslist_big) ./ sqrt.(rslist_big),
        "--";
        color="C0",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\sqrt{r_s} $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s $(sgn_d_rpa) $(round(abs(d_rpa); sigdigits=3)) r_s \\sqrt{r_s}\$",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\sqrt{\\log r_s} $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) \\sqrt{r_s} $(sgn_d_rpa) $(round(abs(d_rpa); sigdigits=3)) \\sqrt{r_s}\$",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s $(sgn_d_rpa) $(round(abs(d_rpa); sigdigits=3)) r_s \\log r_s\$",
        # label="\$\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s\\right)\$",
        label="\$\\sqrt{r_s}\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s\\right)\$",
    )
    ax7.plot(
        rslist,
        w0_over_EF_rpa_fls,
        # w0_over_EF_rpa_fls ./ sqrt.(rslist),
        "o-";
        color="C1",
        label="\$RPA+FL\$",
    )
    ax7.plot(
        rslist_big,
        model_rpa_fl7.(rslist_big),
        # model_rpa_fl7.(rslist_big) ./ sqrt.(rslist_big),
        "--";
        color="C1",
        # label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\sqrt{r_s} $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s $(sgn_d_rpa_fl) $(round(abs(d_rpa_fl); sigdigits=3)) r_s \\sqrt{r_s}\$",
        # label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s $(sgn_d_rpa_fl) $(round(abs(d_rpa_fl); sigdigits=3)) r_s \\log r_s\$",
        # label="\$\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s\\right)\$",
        label="\$\\sqrt{r_s}\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s\\right)\$",
    )
    ax7.set_xlabel("\$r_s\$")
    ax7.set_ylabel("\$\\Omega_t(r_s)\$")
    # ax7.set_ylabel("\$\\Omega_t(r_s) / \\sqrt{r_s}\$")
    # ax7.set_ylabel("\$\\Omega_t(r_s) / \\sqrt{r_s} = \\sqrt{B(r_s) / A(r_s)}\$")
    ax7.legend(; loc="best")
    plt.tight_layout()
    fig7.savefig(
        "results/self_energy_fits/$(int_type)/low_high_turning_points_rs_" *
        # "results/self_energy_fits/$(int_type)/low_high_turning_points_over_sqrt_rs_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_EF_$(int_type).pdf",
    )

    # Plot peak positions vs rs
    println("sigma_rpa_peak_wns_over_EF = ", sigma_rpa_peak_wns_over_EF)
    println("sigma_rpa_fl_peak_wns_over_EF = ", sigma_rpa_fl_peak_wns_over_EF)
    fig2, ax2 = plt.subplots()
    ax2.plot(rslist, sigma_rpa_peak_wns_over_EF, "o-"; color="C0", label="\$RPA\$")
    ax2.plot(rslist, sigma_rpa_fl_peak_wns_over_EF, "o-"; color="C1", label="\$RPA+FL\$")
    ax2.set_xlabel("\$r_s\$")
    ax2.set_ylabel(
        "\${\\mathrm{argmax}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\\right\\rbrace\$",
    )
    ax2.legend(; loc="best")
    plt.tight_layout()
    fig2.savefig(
        "results/self_energy_fits/$(int_type)/peak_positions_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_EF_$(int_type).pdf",
    )

    # Plot peak values vs rs
    println("sigma_rpa_peaks_over_EF = ", sigma_rpa_peaks_over_EF)
    println("sigma_rpa_fl_peaks_over_EF = ", sigma_rpa_fl_peaks_over_EF)
    # Least-squares fit to peak values
    # @. model3(rs, p) = rs^2 * (p[1] + p[2] * log(rs))

    # # Old fit: ~ (r_s)^2
    # @. model3(rs, p) = rs^2 * (p[1] + p[2] * log(rs) + p[3] * rs)

    # New fit: ~ (r_s)^(3/2)
    # @. model3(rs, p) = rs^(p[1]) * (p[2] + p[3] * log(rs))
    # @. model3(rs, p) = rs^(3 / 2) * (p[1] + p[2] * log(rs))
    # @. model3(rs, p) = rs * (p[1] + p[2] * sqrt(rs) + p[3] * rs)

    # @. model3(rs, p) = rs^(3 / 2) * (p[1] + p[2] * log(rs) + p[3] * rs)
    # @. model3(rs, p) = rs^(3 / 2) * (p[1] + p[2] * sqrt(rs) + p[3] * rs)
    @. model3(rs, p) = rs^(3 / 2) * (p[1] + p[2] * log(rs) + p[3] * sqrt(rs))

    # fit_rpa = curve_fit(model3, rslist, sigma_rpa_peaks_over_EF, [1.5])
    # fit_rpa_fl = curve_fit(model3, rslist, sigma_rpa_fl_peaks_over_EF, [1.5])
    # fit_rpa = curve_fit(model3, rslist, sigma_rpa_peaks_over_EF, [1.0, 1.0])
    # fit_rpa_fl = curve_fit(model3, rslist, sigma_rpa_fl_peaks_over_EF, [1.0, 1.0])
    fit_rpa = curve_fit(model3, rslist, sigma_rpa_peaks_over_EF, [0.1, 0.02, -0.02])
    fit_rpa_fl = curve_fit(model3, rslist, sigma_rpa_fl_peaks_over_EF, [0.1, 0.02, -0.02])
    model_rpa3(rs) = model3(rs, fit_rpa.param)
    model_rpa_fl3(rs) = model3(rs, fit_rpa_fl.param)
    # Coefficients of determination (r²)
    r2_rpa = rsquared(rslist, sigma_rpa_peaks_over_EF, model_rpa3.(rslist))
    r2_rpa_fl = rsquared(rslist, sigma_rpa_fl_peaks_over_EF, model_rpa_fl3.(rslist))
    println("Peak value RPA fit: ", fit_rpa.param, ", r² = $r2_rpa")
    println("Peak value RPA+FL fit: ", fit_rpa_fl.param, ", r² = $r2_rpa_fl")
    # a_rpa = fit_rpa.param[1]
    # a_rpa_fl = fit_rpa_fl.param[1]
    # a_rpa, b_rpa = fit_rpa.param
    # a_rpa_fl, b_rpa_fl = fit_rpa_fl.param
    a_rpa, b_rpa, c_rpa = fit_rpa.param
    a_rpa_fl, b_rpa_fl, c_rpa_fl = fit_rpa_fl.param
    sgn_b_rpa = b_rpa ≥ 0 ? "+" : "-"
    sgn_b_rpa_fl = b_rpa_fl ≥ 0 ? "+" : "-"
    sgn_c_rpa = c_rpa ≥ 0 ? "+" : "-"
    sgn_c_rpa_fl = c_rpa_fl ≥ 0 ? "+" : "-"
    fig3, ax3 = plt.subplots()
    # ax3.plot(rslist, sigma_rpa_peaks_over_EF ./ rslist .^ 2, "o-"; color="C0", label="\$RPA\$")
    ax3.plot(
        rslist,
        sigma_rpa_peaks_over_EF,
        # sigma_rpa_peaks_over_EF ./ rslist,
        # sigma_rpa_peaks_over_EF ./ (rslist .^ (3 / 2)),
        # sigma_rpa_peaks_over_EF ./ (rslist .^ a_rpa),
        "o-";
        color="C0",
        label="\$RPA\$",
    )
    # ax3.plot(rslist, sigma_rpa_peaks_over_EF, "o-"; color="C0", label="\$RPA\$")
    ax3.plot(
        rslist_big,
        model_rpa3.(rslist_big),
        # model_rpa3.(rslist_big) ./ (rslist_big .^ (3 / 2)),
        # model_rpa3.(rslist_big) ./ (rslist_big .^ a_rpa),
        "--";
        color="C0",
        # label="\$r^{$(round(a_rpa; sigdigits=3))}_s\$",
        # label="\$r^{3/2}_s\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s\\right)\$",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) \\sqrt{r_s}\$",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s\$",
        label="\$r^{3/2}_s\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) \\sqrt{r_s}\\right)\$",
        # label="\$r^2_s\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s\\right)\$",
        # label="\$\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s\\right)\$",
        #
        # label="\$r^2_s\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s\\right)\$",
        # label="\$$(round(a_rpa; sigdigits=3)) r_s $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) r^2_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r^3_s\$",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s^2\$",
    )
    ax3.plot(
        rslist,
        sigma_rpa_fl_peaks_over_EF,
        # sigma_rpa_fl_peaks_over_EF ./ rslist,
        # sigma_rpa_fl_peaks_over_EF ./ (rslist .^ (3 / 2)),
        # sigma_rpa_fl_peaks_over_EF ./ (rslist .^ a_rpa_fl),
        "o-";
        color="C1",
        label="\$RPA+FL\$",
    )
    ax3.plot(
        rslist_big,
        model_rpa_fl3.(rslist_big),
        # model_rpa_fl3.(rslist_big) ./ rslist_big,
        # model_rpa_fl3.(rslist_big) ./ (rslist_big .^ (3 / 2)),
        # model_rpa_fl3.(rslist_big) ./ (rslist_big .^ a_rpa_fl),
        "--";
        color="C1",
        # label="\$r^{$(round(a_rpa_fl; sigdigits=3))}_s\$",
        # label="\$r^{3/2}_s\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s\\right)\$",
        # label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) \\sqrt{r_s}\$",
        # label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s\$",
        label="\$r^{3/2}_s\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) \\sqrt{r_s}\\right)\$",
        # label="\$r^2_s\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s\\right)\$",
        # label="\$\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s\\right)\$",
        # label="\$$(round(a_rpa_fl; sigdigits=3)) r_s $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) r^2_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r^3_s\$",
        # label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s^2\$",
    )
    ax3.set_xlabel("\$r_s\$")
    ax3.set_ylabel(
        # "\${\\mathrm{max}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) \\right\\rbrace / r^2_s\$",
        # "\${\\mathrm{max}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) \\right\\rbrace / r^{3/2}_s\$",
        "\${\\mathrm{max}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) \\right\\rbrace\$",
    )
    ax3.legend(; loc="best")
    plt.tight_layout()
    fig3.savefig(
        # "results/self_energy_fits/$(int_type)/peak_values_over_rs$(round(a_rpa_fl; sigdigits=3))_" *
        # "results/self_energy_fits/$(int_type)/peak_values_over_rs2_" *
        # "results/self_energy_fits/$(int_type)/peak_values_over_rs32_" *
        "results/self_energy_fits/$(int_type)/peak_values_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_$(int_type).pdf",
        # "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_$(int_type)_v3.pdf",
    )

    # Same as above, but divided by the leading dependence rs^(3/2)
    sigma_rpa_peaks_over_EF_over_rs32 = sigma_rpa_peaks_over_EF ./ (rslist .^ 1.5)
    sigma_rpa_fl_peaks_over_EF_over_rs32 = sigma_rpa_fl_peaks_over_EF ./ (rslist .^ 1.5)
    # Least-squares fit to peak values
    @. model3_over_rs32(rs, p) = p[1] + p[2] * log(rs) + p[3] * sqrt(rs)
    # @. model4_over_rs32(rs, p) = sqrt(rs) * (p[1] + p[2] * log(rs) + p[3] * rs)
    fit_rpa = curve_fit(
        model3_over_rs32,
        rslist,
        sigma_rpa_peaks_over_EF_over_rs32,
        [1.0, 0.1, 0.1],
    )
    fit_rpa_fl = curve_fit(
        model3_over_rs32,
        rslist,
        sigma_rpa_fl_peaks_over_EF_over_rs32,
        [1.0, 0.1, 0.1],
    )
    # fit_rpa_v2 = curve_fit(
    #     model4_over_rs32,
    #     rslist,
    #     sigma_rpa_peaks_over_EF_over_rs32,
    #     [1.0, 0.1, 0.1],
    # )
    # fit_rpa_fl_v2 = curve_fit(
    #     model4_over_rs32,
    #     rslist,
    #     sigma_rpa_fl_peaks_over_EF_over_rs32,
    #     [1.0, 0.1, 0.1],
    # )
    model_rpa3_over_rs32(rs) = model3_over_rs32(rs, fit_rpa.param)
    model_rpa_fl3_over_rs32(rs) = model3_over_rs32(rs, fit_rpa_fl.param)
    # model_rpa4_over_rs32(rs) = model4_over_rs32(rs, fit_rpa_v2.param)
    # model_rpa_fl4_over_rs32(rs) = model4_over_rs32(rs, fit_rpa_fl_v2.param)
    # Coefficients of determination (r²)
    r2_rpa =
        rsquared(rslist, sigma_rpa_peaks_over_EF_over_rs32, model_rpa3_over_rs32.(rslist))
    r2_rpa_fl = rsquared(
        rslist,
        sigma_rpa_fl_peaks_over_EF_over_rs32,
        model_rpa_fl3_over_rs32.(rslist),
    )
    println("(v1) Peak value RPA fit: ", fit_rpa.param, ", r² = $r2_rpa")
    println("(v1) Peak value RPA+FL fit: ", fit_rpa_fl.param, ", r² = $r2_rpa_fl")
    a_rpa, b_rpa, c_rpa = fit_rpa.param
    a_rpa_fl, b_rpa_fl, c_rpa_fl = fit_rpa_fl.param
    sgn_b_rpa = b_rpa ≥ 0 ? "+" : "-"
    sgn_b_rpa_fl = b_rpa_fl ≥ 0 ? "+" : "-"
    sgn_c_rpa = c_rpa ≥ 0 ? "+" : "-"
    sgn_c_rpa_fl = c_rpa_fl ≥ 0 ? "+" : "-"
    # r2_rpa_v2 =
    #     rsquared(rslist, sigma_rpa_peaks_over_EF_over_rs32, model_rpa4_over_rs32.(rslist))
    # r2_rpa_fl_v2 = rsquared(
    #     rslist,
    #     sigma_rpa_fl_peaks_over_EF_over_rs32,
    #     model_rpa_fl4_over_rs32.(rslist),
    # )
    # println("(v2) Peak value RPA fit: ", fit_rpa.param, ", r² = $r2_rpa_v2")
    # println("(v2) Peak value RPA+FL fit: ", fit_rpa_fl.param, ", r² = $r2_rpa_fl_v2")
    # a_rpa_v2, b_rpa_v2, c_rpa_v2 = fit_rpa_v2.param
    # a_rpa_fl_v2, b_rpa_fl_v2, c_rpa_fl_v2 = fit_rpa_fl_v2.param
    # sgn_b_rpa_v2 = b_rpa_v2 ≥ 0 ? "+" : "-"
    # sgn_b_rpa_fl_v2 = b_rpa_fl_v2 ≥ 0 ? "+" : "-"
    # sgn_c_rpa_v2 = c_rpa_v2 ≥ 0 ? "+" : "-"
    # sgn_c_rpa_fl_v2 = c_rpa_fl_v2 ≥ 0 ? "+" : "-"
    fig3, ax3 = plt.subplots()
    ax3.plot(rslist, sigma_rpa_peaks_over_EF_over_rs32, "o-"; color="C0", label="\$RPA\$")
    # v1
    ax3.plot(
        rslist_big,
        model_rpa3_over_rs32.(rslist_big),
        "--";
        color="C0",
        label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) \\sqrt{r_s}\$",
    )
    # # v2
    # ax3.plot(
    #     rslist_big,
    #     model_rpa4_over_rs32.(rslist_big),
    #     "-.";
    #     color="C0",
    #     label="\$\\sqrt{r_s} \\left($(round(a_rpa_v2; sigdigits=3)) $(sgn_b_rpa_v2) $(round(abs(b_rpa_v2); sigdigits=3)) \\log r_s $(sgn_c_rpa_v2) $(round(abs(c_rpa_v2); sigdigits=3)) r_s\\right)\$",
    # )
    ax3.plot(
        rslist,
        sigma_rpa_fl_peaks_over_EF_over_rs32,
        "o-";
        color="C1",
        label="\$RPA+FL\$",
    )
    # v1
    ax3.plot(
        rslist_big,
        model_rpa_fl3_over_rs32.(rslist_big),
        "--";
        color="C1",
        label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) \\sqrt{r_s}\$",
    )
    # # v2
    # ax3.plot(
    #     rslist_big,
    #     model_rpa_fl4_over_rs32.(rslist_big),
    #     "-.";
    #     color="C1",
    #     label="\$\\sqrt{r_s} \\left($(round(a_rpa_fl_v2; sigdigits=3)) $(sgn_b_rpa_fl_v2) $(round(abs(b_rpa_fl_v2); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl_v2) $(round(abs(c_rpa_fl_v2); sigdigits=3)) r_s\\right)\$",
    # )
    ax3.set_xlabel("\$r_s\$")
    ax3.set_ylabel(
        "\${\\mathrm{max}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) \\right\\rbrace / r^{3/2}_s\$",
    )
    ax3.legend(; loc="best")
    plt.tight_layout()
    fig3.savefig(
        "results/self_energy_fits/$(int_type)/peak_values_over_rs32_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_$(int_type).pdf",
    )
    fig3, ax3 = plt.subplots()
    ax3.plot(rslist, sigma_rpa_peaks_over_EF_over_rs32, "o-"; color="C0", label="\$RPA\$")
    # v1
    ax3.plot(
        rslist_big,
        model_rpa3_over_rs32.(rslist_big),
        "--";
        color="C0",
        label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) \\sqrt{r_s}\$",
    )
    # # v2
    # ax3.plot(
    #     rslist_big,
    #     model_rpa4_over_rs32.(rslist_big),
    #     "-.";
    #     color="C0",
    #     label="\$\\sqrt{r_s} \\left($(round(a_rpa_v2; sigdigits=3)) $(sgn_b_rpa_v2) $(round(abs(b_rpa_v2); sigdigits=3)) \\log r_s $(sgn_c_rpa_v2) $(round(abs(c_rpa_v2); sigdigits=3)) r_s\\right)\$",
    # )
    ax3.plot(
        rslist,
        sigma_rpa_fl_peaks_over_EF_over_rs32,
        "o-";
        color="C1",
        label="\$RPA+FL\$",
    )
    # v1
    ax3.plot(
        rslist_big,
        model_rpa_fl3_over_rs32.(rslist_big),
        "--";
        color="C1",
        label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) \\sqrt{r_s}\$",
    )
    # # v2
    # ax3.plot(
    #     rslist_big,
    #     model_rpa_fl4_over_rs32.(rslist_big),
    #     "-.";
    #     color="C1",
    #     label="\$\\sqrt{r_s} \\left($(round(a_rpa_fl_v2; sigdigits=3)) $(sgn_b_rpa_fl_v2) $(round(abs(b_rpa_fl_v2); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl_v2) $(round(abs(c_rpa_fl_v2); sigdigits=3)) r_s\\right)\$",
    # )
    ax3.set_xlabel("\$r_s\$")
    ax3.set_ylabel(
        "\${\\mathrm{max}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) \\right\\rbrace / r^{3/2}_s\$",
    )
    ax3.legend(; loc="best")
    plt.tight_layout()
    fig3.savefig(
        "results/self_energy_fits/$(int_type)/peak_values_over_rs32_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_$(int_type).pdf",
    )
    # Multiply back the leading rs^(3/2) coefficient to fit the peak values
    fig3p, ax3p = plt.subplots()
    ax3p.plot(rslist, sigma_rpa_peaks_over_EF, "o-"; color="C0", label="\$RPA\$")
    ax3p.plot(
        rslist_big,
        model_rpa3_over_rs32.(rslist_big) .* (rslist_big .^ (3 / 2)),
        "--";
        color="C0",
        label="\$r^{3/2}_s \\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) \\sqrt{r_s}\\right)\$",
    )
    ax3p.plot(rslist, sigma_rpa_fl_peaks_over_EF, "o-"; color="C1", label="\$RPA+FL\$")
    ax3p.plot(
        rslist_big,
        model_rpa_fl3_over_rs32.(rslist_big) .* (rslist_big .^ (3 / 2)),
        "--";
        color="C1",
        label="\$r^{3/2}_s \\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) \\sqrt{r_s}\\right)\$",
    )
    ax3p.set_xlabel("\$r_s\$")
    ax3p.set_ylabel(
        "\${\\mathrm{max}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) \\right\\rbrace\$",
    )
    ax3p.legend(; loc="best")
    plt.tight_layout()
    fig3p.savefig(
        "results/self_energy_fits/$(int_type)/peak_values_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_$(int_type)_new.pdf",
    )

    # Plot Z-factors vs rs for RPA(+FL)
    println("\nZ_RPA:\n", zfactors_rpa)
    println("Z_RPA+FL:\n", zfactors_rpa_fl)
    # Least-squares fit to Z-factors
    # @. model4(rs, p) = 1 + p[1] * log(rs) + p[2] * rs + p[3] * rs * log(rs)
    @. model4(rs, p) = 1 + p[1] * rs + p[2] * rs^2 + p[3] * rs^3
    fit_rpa = curve_fit(model4, rslist, zfactors_rpa, [1.0, 1.0, 1.0])
    fit_rpa_fl = curve_fit(model4, rslist, zfactors_rpa_fl, [1.0, 1.0, 1.0])
    # @. model4(rs, p) = 1 / (1 + rs)^p[1]
    # fit_rpa = curve_fit(model4, rslist, zfactors_rpa, [0.5])
    # fit_rpa_fl = curve_fit(model4, rslist, zfactors_rpa_fl, [0.5])
    model_rpa4(rs) = model4(rs, fit_rpa.param)
    model_rpa_fl4(rs) = model4(rs, fit_rpa_fl.param)
    # Coefficients of determination (r²)
    r2_rpa = rsquared(rslist, zfactors_rpa, model_rpa4.(rslist))
    r2_rpa_fl = rsquared(rslist, zfactors_rpa_fl, model_rpa_fl4.(rslist))
    println("RPA fit: ", fit_rpa.param, ", r² = $r2_rpa")
    println("RPA+FL fit: ", fit_rpa_fl.param, ", r² = $r2_rpa_fl")
    # a_rpa = fit_rpa.param[1]
    # a_rpa_fl = fit_rpa_fl.param[1]
    a_rpa, b_rpa, c_rpa = fit_rpa.param
    a_rpa_fl, b_rpa_fl, c_rpa_fl = fit_rpa_fl.param
    sgn_a_rpa = a_rpa ≥ 0 ? "+" : "-"
    sgn_a_rpa_fl = a_rpa_fl ≥ 0 ? "+" : "-"
    sgn_b_rpa = b_rpa ≥ 0 ? "+" : "-"
    sgn_b_rpa_fl = b_rpa_fl ≥ 0 ? "+" : "-"
    sgn_c_rpa = c_rpa ≥ 0 ? "+" : "-"
    sgn_c_rpa_fl = c_rpa_fl ≥ 0 ? "+" : "-"
    fig4, ax4 = plt.subplots()
    ax4.plot(rslist, zfactors_rpa, "o-"; color="C0", label="\$RPA\$")
    ax4.plot(
        rslist_big,
        model_rpa4.(rslist_big),
        "--";
        color="C0",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s\$",
        label="\$1 $(sgn_a_rpa) $(round(abs(a_rpa); sigdigits=3)) r_s $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) r^2_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r^3_s\$",
        # label="\$(1 + r_s)^{-$(round(a_rpa; sigdigits=3))}\$",
    )
    ax4.plot(rslist, zfactors_rpa_fl, "o-"; color="C1", label="\$RPA+FL\$")
    ax4.plot(
        rslist_big,
        model_rpa_fl4.(rslist_big),
        "--";
        color="C1",
        # label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s\$",
        label="\$1 $(sgn_a_rpa_fl) $(round(abs(a_rpa_fl); sigdigits=3)) r_s $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) r^2_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r^3_s\$",
        # label="\$(1 + r_s)^{-$(round(a_rpa_fl; sigdigits=3))}\$",
    )
    ax4.set_xlabel("\$r_s\$")
    ax4.set_ylabel("\$Z_{k_F}\$")
    ax4.legend(; loc="best")
    plt.tight_layout()
    fig4.savefig(
        "results/self_energy_fits/$(int_type)/zfactor_k=kF_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_$(int_type).pdf",
    )

    # Plot A(rs) for RPA(+FL)
    A_rpa = 1 ./ zfactors_rpa .- 1
    A_rpa_fl = 1 ./ zfactors_rpa_fl .- 1
    # Least-squares fit to Z-factors
    # @. model10(rs, p) = rs * (p[1] + p[2] * log(rs) + p[3] * rs)
    # @. model10(rs, p) = rs * (p[1] + p[2] * sqrt(rs) + p[3] * rs)
    @. model10(rs, p) = rs * (p[1] + p[2] * rs + p[3] * rs^2)
    fit_rpa = curve_fit(model10, rslist, A_rpa, [1.0, 1.0, 1.0])
    fit_rpa_fl = curve_fit(model10, rslist, A_rpa_fl, [1.0, 1.0, 1.0])
    # fit_rpa = curve_fit(model10, rslist, A_rpa, [1.0, 1.0, 1.0])
    # fit_rpa_fl = curve_fit(model10, rslist, A_rpa_fl, [1.0, 1.0, 1.0])
    # @. model10(rs, p) = p[1] + p[2] * log(rs) + p[3] * rs + p[4] * rs * log(rs)
    # fit_rpa = curve_fit(model10, rslist, A_rpa, [1.0, 1.0, 1.0, 1.0])
    # fit_rpa_fl = curve_fit(model10, rslist, A_rpa_fl, [1.0, 1.0, 1.0, 1.0])
    model_rpa10(rs) = model10(rs, fit_rpa.param)
    model_rpa_fl10(rs) = model10(rs, fit_rpa_fl.param)
    # Coefficients of determination (r²)
    r2_rpa = rsquared(rslist, A_rpa, model_rpa10.(rslist))
    r2_rpa_fl = rsquared(rslist, A_rpa_fl, model_rpa_fl10.(rslist))
    println("RPA fit: ", fit_rpa.param, ", r² = $r2_rpa")
    println("RPA+FL fit: ", fit_rpa_fl.param, ", r² = $r2_rpa_fl")
    a_rpa, b_rpa, c_rpa = fit_rpa.param
    a_rpa_fl, b_rpa_fl, c_rpa_fl = fit_rpa_fl.param
    sgn_b_rpa = b_rpa ≥ 0 ? "+" : "-"
    sgn_b_rpa_fl = b_rpa_fl ≥ 0 ? "+" : "-"
    sgn_c_rpa = c_rpa ≥ 0 ? "+" : "-"
    sgn_c_rpa_fl = c_rpa_fl ≥ 0 ? "+" : "-"
    # sgn_d_rpa = d_rpa ≥ 0 ? "+" : "-"
    # sgn_d_rpa_fl = d_rpa_fl ≥ 0 ? "+" : "-"
    fig10, ax10 = plt.subplots()
    ax10.plot(rslist, A_rpa, "o-"; color="C0", label="\$RPA\$")
    # ax10.plot(rslist, A_rpa ./ rslist, "o-"; color="C0", label="\$RPA\$")
    ax10.plot(
        rslist_big,
        model_rpa10.(rslist_big),
        # model_rpa10.(rslist_big) ./ rslist_big,
        "--";
        color="C0",
        label="\$r_s\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r^2_s\\right)\$",
        # label="\$\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r^2_s\\right)\$",
        # label="\$r_s\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s\\right)\$",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s $(sgn_d_rpa) $(round(abs(d_rpa); sigdigits=3)) r_s \\log r_s\$",
    )
    ax10.plot(rslist, A_rpa_fl, "o-"; color="C1", label="\$RPA+FL\$")
    # ax10.plot(rslist, A_rpa_fl ./ rslist, "o-"; color="C1", label="\$RPA+FL\$")
    ax10.plot(
        rslist_big,
        model_rpa_fl10.(rslist_big),
        # model_rpa_fl10.(rslist_big) ./ rslist_big,
        "--";
        color="C1",
        label="\$r_s\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r^2_s\\right)\$",
        # label="\$\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r^2_s\\right)\$",
        # label="\$r_s\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s\\right)\$",
        # label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s $(sgn_d_rpa_fl) $(round(abs(d_rpa_fl); sigdigits=3)) r_s \\log r_s\$",
    )
    ax10.set_xlabel("\$r_s\$")
    # ax10.set_ylabel("\$A(r_s) / r_s\$")
    ax10.set_ylabel("\$A(r_s) = \\frac{1}{z(r_s)} - 1\$")
    ax10.legend(; loc="best")
    plt.tight_layout()
    fig10.savefig(
        "results/self_energy_fits/$(int_type)/A_" *
        # "results/self_energy_fits/$(int_type)/A_over_rs_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_$(int_type).pdf",
    )

    # Plot second-order moments vs rs for RPA(+FL)
    fig9, ax9 = plt.subplots()

    # Least-squares fit to B(rs) = C^{(1)} / EF^2
    # alpha = (4 / 9π)^(1/3)
    # c_alpha = 8 * alpha^2 / π
    # @. model9(rs, p) = c_alpha * rs^2 * (p[1] - log(rs) / 2π)
    @. model9(rs, p) = rs^2 * (p[1] + p[2] * log(rs))
    # @. model9(rs, p) = p[1] + p[2] * rs + p[3] * rs^2
    # @. model9(rs, p) = p[1] + p[2] * log(rs) + p[3] * rs + p[4] * rs * log(rs) + p[5] * rs^2
    # fit_rpa = curve_fit(model9, rslist, c1_rpas_over_EF2, [0.7])
    # fit_rpa_fl = curve_fit(model9, rslist, c1_rpa_fls_over_EF2, [0.7])
    fit_rpa = curve_fit(model9, rslist, c1_rpas_over_EF2, [1.0, 1.0])
    fit_rpa_fl = curve_fit(model9, rslist, c1_rpa_fls_over_EF2, [1.0, 1.0])
    # fit_rpa = curve_fit(model9, rslist, c1_rpas_over_EF2, [1.0, 1.0, 1.0])
    # fit_rpa_fl = curve_fit(model9, rslist, c1_rpa_fls_over_EF2, [1.0, 1.0, 1.0])
    # fit_rpa = curve_fit(model9, rslist, c1_rpas_over_EF2, [1.0, 1.0, 1.0, 1.0, 1.0])
    # fit_rpa_fl = curve_fit(model9, rslist, c1_rpa_fls_over_EF2, [1.0, 1.0, 1.0, 1.0, 1.0])
    model_rpa9(rs) = model9(rs, fit_rpa.param)
    model_rpa_fl9(rs) = model9(rs, fit_rpa_fl.param)
    # Coefficients of determination (r²)
    r2_rpa = rsquared(rslist, c1_rpas_over_EF2, model_rpa9.(rslist))
    r2_rpa_fl = rsquared(rslist, c1_rpa_fls_over_EF2, model_rpa_fl9.(rslist))
    println("RPA fit: ", fit_rpa.param, ", r² = $r2_rpa")
    println("RPA+FL fit: ", fit_rpa_fl.param, ", r² = $r2_rpa_fl")
    # a_rpa = fit_rpa.param[1]
    # a_rpa_fl = fit_rpa_fl.param[1]
    a_rpa, b_rpa = fit_rpa.param
    a_rpa_fl, b_rpa_fl = fit_rpa_fl.param
    # a_rpa, b_rpa, c_rpa = fit_rpa.param
    # a_rpa_fl, b_rpa_fl, c_rpa_fl = fit_rpa_fl.param
    # sgn_b_rpa = b_rpa ≥ 0 ? "+" : "-"
    # sgn_b_rpa_fl = b_rpa_fl ≥ 0 ? "+" : "-"
    # sgn_c_rpa = c_rpa ≥ 0 ? "+" : "-"
    # sgn_c_rpa_fl = c_rpa_fl ≥ 0 ? "+" : "-"

    # a_rpa, b_rpa, c_rpa, d_rpa, e_rpa = fit_rpa.param
    # a_rpa_fl, b_rpa_fl, c_rpa_fl, d_rpa_fl, e_rpa_fl = fit_rpa_fl.param
    # sgn_b_rpa = b_rpa ≥ 0 ? "+" : "-"
    # sgn_b_rpa_fl = b_rpa_fl ≥ 0 ? "+" : "-"
    # sgn_c_rpa = c_rpa ≥ 0 ? "+" : "-"
    # sgn_c_rpa_fl = c_rpa_fl ≥ 0 ? "+" : "-"
    # sgn_d_rpa = d_rpa ≥ 0 ? "+" : "-"
    # sgn_d_rpa_fl = d_rpa_fl ≥ 0 ? "+" : "-"$(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) 
    ax9.plot(rslist, c1_rpas_over_EF2, "o-"; color="C0", label="\$RPA\$")
    # ax9.plot(rslist, c1_rpas_over_EF2 ./ rslist .^ 2, "o-"; color="C0", label="\$RPA\$")
    ax9.plot(
        rslist_big,
        model_rpa9.(rslist_big),
        # model_rpa9.(rslist_big) ./ rslist_big .^ 2,
        "--";
        color="C0",
        # label="\$\\frac{8}{\\pi}(\\alpha r_s)^2\\left($(round(a_rpa; sigdigits=3)) - \\frac{\\log r_s}{2\\pi}\\right)\$",
        label="\$r^2_s \\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s\\right)\$",
        # label="\$\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s\\right)\$",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s^2\$",
        # label = "\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s $(sgn_d_rpa) $(round(abs(d_rpa); sigdigits=3)) r_s \\log r_s $(sgn_e_rpa) $(round(abs(e_rpa); sigdigits=3)) r_s^2\$",
    )
    ax9.plot(
        rslist,
        c1_rpa_fls_over_EF2,
        # c1_rpa_fls_over_EF2 ./ rslist .^ 2,
        "o-";
        color="C1",
        label="\$RPA+FL\$",
    )
    ax9.plot(
        rslist_big,
        model_rpa_fl9.(rslist_big),
        # model_rpa_fl9.(rslist_big) ./ rslist_big .^ 2,
        "--";
        color="C1",
        # label="\$\\frac{8}{\\pi}(\\alpha r_s)^2\\left($(round(a_rpa_fl; sigdigits=3)) - \\frac{\\log r_s}{2\\pi}\\right)\$",
        label="\$r^2_s \\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s\\right)\$",
        # label="\$\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s\\right)\$",
        # label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s^2\$",
        # label = "\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s $(sgn_d_rpa_fl) $(round(abs(d_rpa_fl); sigdigits=3)) r_s \\log r_s $(sgn_e_rpa_fl) $(round(abs(e_rpa_fl); sigdigits=3)) r_s^2\$",
    )
    ax9.set_xlabel("\$r_s\$")
    ax9.legend(; loc="best")
    # plt.tight_layout()
    # ax9.set_ylabel("\$C^{(1)} / \\epsilon^2_{F}\$")
    # fig9.savefig(
    #     "results/self_energy_fits/$(int_type)/second_order_moments_" *
    #     "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_EF_$(int_type).pdf",
    #     )
    ax9.set_ylabel("\$B(r_s)\$")
    # ax9.set_ylabel("\$B(r_s) / r^2_s\$")
    # ax9.set_ylabel("\$B(r_s) = C^{(1)}(r_s) / \\epsilon^2_{F}\$")
    plt.tight_layout()
    fig9.savefig(
        "results/self_energy_fits/$(int_type)/B_" *
        # "results/self_energy_fits/$(int_type)/B_over_rs2_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_$(int_type).pdf",
    )

    # Low-frequency behavior of -ImΣ
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 1.5 * max_sigma)
    ax5.set_xlabel("\$\\omega_n\$")
    ax5.set_ylabel("\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\$")
    ax5.legend(; loc="best")
    plt.tight_layout()
    # fig5.savefig(
    #     "results/self_energy_fits/$(int_type)/im_sigma_low_freq_" *
    #     "rs=$(round.(rslist_small; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_$(int_type).pdf",
    # )

    return
end

main()
