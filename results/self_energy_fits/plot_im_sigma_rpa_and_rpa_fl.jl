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
using LsqFit

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory
const vzn_dir = "results/vzn_paper"

# Small parameter of the UEG theory
const α = (4 / 9π)^(1 / 3)

const zfactor_rpa_benchmarks = Dict(
    1.0 => 0.8601,
    2.0 => 0.7642,
    3.0 => 0.6927,
    4.0 => 0.6367,
    5.0 => 0.5913,
    6.0 => 0.5535,
)

"""@ beta = 1000, small cutoffs"""
const rpa_moments = Dict(
    0.01 => 133629.31338476276,
    0.1 => 996.1421229225255,
    0.5 => 30.654593391597565,
    1.0 => 6.723994594646203,
    2.0 => 1.4584858366718452,
    3.0 => 0.5936133107894049,
    4.0 => 0.3130605327419725,
    5.0 => 0.1903819378977648,
    7.5 => 0.07694747219230284,
    10.0 => 0.04040127700428145,
)
const rpa_fl_moments_ko_takada = Dict(
    0.01 => 113120.39532239494,
    0.1 => 794.082225081879,
    0.5 => 22.706255132471114,
    1.0 => 4.739085706326042,
    2.0 => 0.958442122563793,
    3.0 => 0.37029855296213815,
    4.0 => 0.18744178767319505,
    5.0 => 0.11026002348040444,
    7.5 => 0.041904772294599595,
    10.0 => 0.02107009260794267,
)

"""@ beta = 1000, converged cutoffs"""
const rpa_moments_converged = Dict(
    0.1 => 1008.2063101513248,
    0.5 => 31.137158213702797,
    1.0 => 6.844635158665476,
    2.0 => 1.4886456606863716,
    3.0 => 0.6070175380088686,
    4.0 => 0.3206003336440671,
    5.0 => 0.19520736183070095,
    7.5 => 0.07909205243614699,
    10.0 => 0.04160757487540137,
)
const rpa_fl_moments_ko_takada_converged = Dict(
    0.1 => 799.1121831657149,
    0.5 => 22.86600392215787,
    1.0 => 4.770690856098736,
    2.0 => 0.9641346197840088,
    3.0 => 0.37237151488395626,
    4.0 => 0.18847524555192435,
    5.0 => 0.11087443456532428,
    7.5 => 0.042155695620371754,
    10.0 => 0.021207322984162568,
)

# function limit_at_infinity(f; x0=x0, dx=dx, atol=1e-13, maxstep=1e9)
#     x1 = x0 + dx
#     f0 = f(x0)
#     f1 = f(x1)
#     step = 1
#     while abs(f1 - f0) > atol
#         if step > maxstep
#             @warn "Requested atol was not obtainable within $maxstep steps!"
#             break
#         end
#         step += 1
#         x0 = x1
#         f0 = f1
#         x1 += dx
#         f1 = f(x1)
#     end
#     return f1, x1, step
# end

function rsquared(xs, ys, yhats)
    ybar = sum(yhats) / length(yhats)
    ss_res = sum((yhats .- ys) .^ 2)
    ss_tot = sum((ys .- ybar) .^ 2)
    return 1 - ss_res / ss_tot
end

function get_Fs(rs)
    return get_Fs_PW(rs)
    # if rs < 1.0 || rs > 5.0
    #     return get_Fs_DMC(rs)
    # else
    #     return get_Fs_PW(rs)
    # end
end

"""
Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ from DMC data of 
Moroni, Ceperley & Senatore (1995) [Phys. Rev. Lett. 75, 689].
"""
function get_Fs_DMC(rs)
    SOSEM.@todo
end

"""
Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ via interpolation of the 
compressibility ratio data of Perdew & Wang (1992) [Phys. Rev. B 45, 13244].
"""
function get_Fs_PW(rs)
    # if rs < 1.0 || rs > 5.0
    #     @warn "The Perdew-Wang interpolation for Fs may " *
    #           "be inaccurate outside the metallic regime!"
    # end
    kappa0_over_kappa = 1.0025 - 0.1721rs - 0.0036rs^2
    # F⁰ₛ = κ₀/κ - 1
    return kappa0_over_kappa - 1.0
end

function get_rpa_analytic_tail(wns, param::Parameter.Para)
    rs = round(param.rs; sigdigits=13)
    # VZN prefactor C
    # prefactor = -16 * sqrt(2) * (α * rs)^2 / 3π
    # -Im(1 / i^(3/2)) = 1/sqrt(2)
    prefactor = -16 * sqrt(2) * (α * rs)^2 * (1 / sqrt(2)) / 3π
    return @. -prefactor / wns^(3 / 2)
end

function get_rpa_fl_analytic_tail(wns, param::Parameter.Para)
    # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
    rs = round(param.rs; sigdigits=13)
    Fs = get_Fs(rs)

    # Local-field factor fs = Fs / NF
    fs = Fs / param.NF

    # Note that our fs has an extra minus sign relative to VZN,
    # so we have (1 - f) C / ω^(3/2) ↦ (1 - f) C / ω^(3/2)
    rpa_tail = get_rpa_analytic_tail(wns, param)
    return (1 + fs) * rpa_tail
end

function get_sigma_rpa_wn(param::Parameter.Para; ktarget=0.0, atol=1e-3)
    # Make sure we are using parameters for the bare UEG theory
    @assert param.Λs == param.Λa == 0.0

    # # Converged params
    # Euv, rtol = 10000 * param.EF, 1e-11
    # maxK, minK = 1000param.kF, 1e-8param.kF
    # Nk, order = 25, 20

    # Small params
    Euv, rtol = 1000 * param.EF, 1e-11
    maxK, minK = 30param.kF, 1e-8param.kF
    Nk, order = 11, 8

    # Get RPA+FL self-energy
    sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(
        param;
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
    zfactor = SelfEnergy.zfactor(param, sigma_wn_dynamic)[1]

    # Check zfactor against benchmarks at β = 1000. 
    # It should agree within a few percent (*up to finite-temperature effects).
    # (see: https://numericaleft.github.io/ElectronGas.jl/dev/manual/quasiparticle/)
    rs = round(param.rs; sigdigits=13)
    if rs in keys(zfactor_rpa_benchmarks)
        println("Checking zfactor against zero-temperature benchmark...")
        zfactor_zero_temp = zfactor_rpa_benchmarks[rs]
        percent_error = 100 * abs(zfactor - zfactor_zero_temp) / zfactor_zero_temp
        println("Percent error vs zero-temperature benchmark of Z_kF: $percent_error")
        @assert percent_error ≤ 5
    else
        println("No zero-temperature benchmark available for rs = $(rs)!")
    end

    return kval, wns, sigma_wn_static_kval, sigma_wn_dynamic_kval, zfactor
end

function get_sigma_rpa_fl_wn(
    param::Parameter.Para;
    ktarget=0.0,
    int_type=int_type,
    atol=1e-3,
)
    # Make sure we are using parameters for the bare UEG theory
    @assert param.Λs == param.Λa == 0.0

    # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
    rs = round(param.rs; sigdigits=13)
    Fs = get_Fs(rs)
    println("Fermi liquid parameter at rs = $(rs): Fs = $Fs")

    # # Converged params
    # Euv, rtol = 10000 * param.EF, 1e-11
    # maxK, minK = 1000param.kF, 1e-8param.kF
    # Nk, order = 25, 20

    # Small params
    Euv, rtol = 1000 * param.EF, 1e-11
    maxK, minK = 30param.kF, 1e-8param.kF
    Nk, order = 11, 8

    # Get RPA+FL self-energy
    sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(
        param;
        Euv=Euv,
        rtol=rtol,
        Nk=Nk,
        minK=minK,
        maxK=maxK,
        order=order,
        int_type=int_type,
        Fs=-Fs,  # NOTE: NEFT uses opposite sign convention (Fs > 0)!
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
    zfactor = SelfEnergy.zfactor(param, sigma_wn_dynamic)[1]

    return kval, wns, sigma_wn_static_kval, sigma_wn_dynamic_kval, zfactor
end

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    ktarget = 0.0  # k = kF
    beta = 1000.0
    # beta = 40.0
    # rslist = [1.0]
    # rslist = [1 / α]  # Gives kF = EF = 1
    rslist_small = [1.0, 5.0, 10.0]
    # rslist_small = [1.0]
    rslist = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
    # rslist = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
    # rslist = rslist_small

    # rslist = [1.0, 5.0, 10.0]
    # rslist = [1.0, 1ℯ, 1ℯ^2]
    # rsstrings = ["1", "e", "e^2"]

    # The unit system to use for plotting
    # units = :Rydberg
    units = :EF
    # units = :eTF

    int_type = :ko
    # int_type = :ko_const

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
    sigma_rpa_peaks = []
    sigma_rpa_fl_peaks = []
    sigma_rpa_peak_wns_over_EF = []
    sigma_rpa_fl_peak_wns_over_EF = []
    fig, ax = plt.subplots()
    fig5, ax5 = plt.subplots()
    for (i, rs) in enumerate(rslist)
        println("\nPlotting data for rs = $rs...")

        # Get the G0W0 self-energy and corresponding DLR grid from the ElectronGas package
        # NOTE: Here we need to be careful to generate Σ_G0W0 for the *bare* theory, i.e.,
        #       to use an ElectronGas.Parameter object where Λs (mass2) is zero!
        param = Parameter.rydbergUnit(1 / beta, rs, 3)
        @assert param.Λs == param.Λa == 0.0

        # Get RPA and RPA+FL self-energies
        kval, wns, sigma_rpa_wn_stat, sigma_rpa_wn_dyn, zfactor_rpa =
            get_sigma_rpa_wn(param; ktarget=ktarget)
        kval_fl, wns_fl, sigma_rpa_fl_wn_stat, sigma_rpa_fl_wn_dyn, zfactor_rpa_fl =
            get_sigma_rpa_fl_wn(param; ktarget=ktarget, int_type=int_type)

        # The static parts do not contribute to ImΣ
        println(imag(sigma_rpa_wn_stat))
        println(imag(sigma_rpa_fl_wn_stat))
        @assert isapprox(imag(sigma_rpa_wn_stat), 0.0, atol=1e-5)
        @assert isapprox(imag(sigma_rpa_fl_wn_stat), 0.0, atol=1e-5)

        # The grids should be the same for RPA and RPA+FL
        @assert kval ≈ kval_fl
        @assert wns ≈ wns_fl

        # Energy units in Rydbergs for nondimensionalization of self-energy data
        EF = param.EF
        eTF = param.qTF^2 / (2 * param.me)

        # Nondimensionalize frequencies and self-energies by EF
        wns_over_EF = wns / EF
        # Static part
        sigma_rpa_wn_stat_over_EF = sigma_rpa_wn_stat / EF
        sigma_rpa_fl_wn_stat_over_EF = sigma_rpa_fl_wn_stat / EF
        # Dynamic part
        sigma_rpa_wn_dyn_over_EF = sigma_rpa_wn_dyn / EF
        sigma_rpa_fl_wn_dyn_over_EF = sigma_rpa_fl_wn_dyn / EF

        # Nondimensionalize frequencies and self-energies by eTF
        wns_over_eTF = wns / eTF
        # Static part
        sigma_rpa_wn_stat_over_eTF = sigma_rpa_wn_stat / eTF
        sigma_rpa_fl_wn_stat_over_eTF = sigma_rpa_fl_wn_stat / eTF
        # Dynamic part
        sigma_rpa_wn_dyn_over_eTF = sigma_rpa_wn_dyn / eTF
        sigma_rpa_fl_wn_dyn_over_eTF = sigma_rpa_fl_wn_dyn / eTF

        # Get first-order RPA and RPA+FL moments at k = ktarget
        rpa_c1 = rpa_moments[rs]
        # rpa_c1 = rpa_moments_converged[rs]
        if int_type == :ko
            rpa_fl_c1 = rpa_fl_moments_ko_takada[rs]
            # rpa_fl_c1 = rpa_fl_moments_ko_takada_converged[rs]
        elseif int_type == :ko_const
            rpa_fl_c1 = rpa_fl_moments_const[rs]
        end
        push!(c1_rpas_over_EF2, rpa_c1 / EF^2)
        push!(c1_rpa_fls_over_EF2, rpa_fl_c1 / EF^2)
        println("First-order RPA moment (Rydberg): ", rpa_c1)
        println("First-order RPA+FL moment (Rydberg): ", rpa_fl_c1)

        local wns_plot
        local sigma_rpa_dyn_plot, sigma_rpa_fl_dyn_plot
        local sigma_rpa_stat_plot, sigma_rpa_fl_stat_plot
        # wns_plot = wns
        if units == :Rydberg
            wns_plot = wns
            # Static part
            sigma_rpa_stat_plot = sigma_rpa_wn_stat
            sigma_rpa_fl_stat_plot = sigma_rpa_fl_wn_stat
            # Dynamic part
            sigma_rpa_dyn_plot = sigma_rpa_wn_dyn
            sigma_rpa_fl_dyn_plot = sigma_rpa_fl_wn_dyn
        elseif units == :EF
            wns_plot = wns_over_EF
            # Static part
            sigma_rpa_stat_plot = sigma_rpa_wn_stat_over_EF
            sigma_rpa_fl_stat_plot = sigma_rpa_fl_wn_stat_over_EF
            # Dynamic part
            sigma_rpa_dyn_plot = sigma_rpa_wn_dyn_over_EF
            sigma_rpa_fl_dyn_plot = sigma_rpa_fl_wn_dyn_over_EF
        else  # units == :eTF
            wns_plot = wns_over_eTF
            # Static part
            sigma_rpa_stat_plot = sigma_rpa_wn_stat_over_eTF
            sigma_rpa_fl_stat_plot = sigma_rpa_fl_wn_stat_over_eTF
            # Dynamic part
            sigma_rpa_dyn_plot = sigma_rpa_wn_dyn_over_eTF
            sigma_rpa_fl_dyn_plot = sigma_rpa_fl_wn_dyn_over_eTF
        end
        println("(RPA) -ImΣ(0, 0) = ", -imag(sigma_rpa_dyn_plot[1]))
        println("(RPA+FL) -ImΣ(0, 0) = ", -imag(sigma_rpa_fl_dyn_plot[1]))

        # # Plot of RPA(+FL) -ImΣs in chosen units
        # ax.plot(
        #     wns_over_EF,
        #     -imag(sigma_rpa_dyn_plot),
        #     "C$(i-1)";
        #     label="\$RPA\$ (\$r_s=$(rs)\$)",
        #     # label="\$RPA\$ (\$r_s=$(rsstrings[i])\$)",
        # )
        # ax.plot(
        #     wns_over_EF,
        #     -imag(sigma_rpa_fl_dyn_plot),
        #     "$(darkcolors[i])";
        #     label="\$RPA+FL\$ (\$r_s=$(rs)\$)",
        #     # label="\$RPA+FL\$ (\$r_s=$(rsstrings[i])\$)",
        # )

        if rs in rslist_small
            idx = findfirst(rslist_small .== rs)
            @assert idx in eachindex(rslist_small)
            # Plot low-frequency behavior for RPA(+FL)
            rpa_low_freq = (1 / zfactor_rpa - 1) .* wns
            rpa_fl_low_freq = (1 / zfactor_rpa_fl - 1) .* wns
            ax5.plot(
                wns_over_EF,
                rpa_low_freq / EF,
                "--";
                color="C$(idx-1)",
                # label="\$RPA\$ (\$r_s=$(rs)\$)",
            )
            ax5.plot(
                wns_over_EF,
                rpa_fl_low_freq / EF,
                "--";
                color="$(darkcolors[idx])",
                # label="\$RPA+FL\$ (\$r_s=$(rs)\$)",
            )
            ax5.plot(
                wns_over_EF,
                -imag(sigma_rpa_wn_dyn_over_EF);
                # -imag(sigma_rpa_wn_dyn) / EF;
                color="C$(idx-1)",
                label="\$RPA\$ (\$r_s=$(rs)\$)",
            )
            ax5.plot(
                wns_over_EF,
                -imag(sigma_rpa_fl_wn_dyn_over_EF);
                # -imag(sigma_rpa_fl_wn_dyn) / EF;
                color="$(darkcolors[idx])",
                label="\$RPA+FL\$ (\$r_s=$(rs)\$)",
            )
        end

        max_sigma = max(
            max_sigma,
            maximum(-imag(sigma_rpa_dyn_plot)),
            maximum(-imag(sigma_rpa_fl_dyn_plot)),
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
        push!(sigma_rpa_peaks, maximum(-imag(sigma_rpa_dyn_plot)))
        push!(sigma_rpa_fl_peaks, maximum(-imag(sigma_rpa_fl_dyn_plot)))
        # Z-factors
        push!(zfactors_rpa, zfactor_rpa)
        push!(zfactors_rpa_fl, zfactor_rpa_fl)

        # Low-high interpolation: f(ω) = B / (iω + Ωₜ)
        function lo_hi_fit_rpa(omega_over_EF)
            return (rpa_c1 / EF^2) / (im * omega_over_EF + w0_rpa / EF)
        end
        function lo_hi_fit_rpa_fl(omega_over_EF)
            return (rpa_fl_c1 / EF^2) / (im * omega_over_EF + w0_rpa_fl / EF)
        end

        # Plot -ImΣ, low and high frequency tails together at each rs
        if rs in rslist_small
            fig6, ax6 = plt.subplots()
            ax6.axvline(
                w0_rpa / EF;
                color="gray",
                linestyle="-",
                label="\$\\Omega_t\$ (\$RPA\$)",
            )
            ax6.axvline(
                w0_rpa_fl / EF;
                color="k",
                linestyle="-",
                label="\$\\Omega_t\$ (\$RPA+FL\$)",
            )
            # Set the upper ylim as 10% larger than the tail intersection
            max_tail_intersection = max(
                coeff_low_freq_rpa * w0_rpa / EF,
                coeff_low_freq_rpa_fl * w0_rpa_fl / EF,
            )
            # High-frequency behavior of -ImΣ
            ax6.plot(
                wns_over_EF,
                coeff_high_freq_rpa ./ wns ./ EF;
                color="C$(idx-1)",
                linestyle="--",
            )
            ax6.plot(
                wns_over_EF,
                coeff_high_freq_rpa_fl ./ wns ./ EF;
                color="$(darkcolors[idx])",
                linestyle="--",
            )
            # Low-frequency behavior of -ImΣ
            ax6.plot(
                wns_over_EF,
                coeff_low_freq_rpa .* wns_over_EF;
                color="C$(idx-1)",
                linestyle="--",
            )
            ax6.plot(
                wns_over_EF,
                coeff_low_freq_rpa_fl .* wns_over_EF;
                color="$(darkcolors[idx])",
                linestyle="--",
            )
            # Low-high interpolation -ImΣ ≈ -Im{f(ω)}
            ax6.plot(
                wns_over_EF,
                -imag.(lo_hi_fit_rpa.(wns_over_EF));
                color="C$(idx)",
                linestyle="-",
                label="\$\\mathrm{Im}\\left\\lbrace\\frac{B}{i\\omega_n + \\Omega_t}\\right\\rbrace\$ (\$RPA\$)",
            )
            ax6.plot(
                wns_over_EF,
                -imag.(lo_hi_fit_rpa_fl.(wns_over_EF));
                color="$(darkcolors[idx+1])",
                linestyle="-",
                label="\$\\mathrm{Im}\\left\\lbrace\\frac{B}{i\\omega_n + \\Omega_t}\\right\\rbrace\$ (\$RPA+FL\$)",
            )
            # -ImΣ
            ax6.plot(
                wns_over_EF,
                -imag(sigma_rpa_wn_dyn) / EF;
                color="C$(idx-1)",
                label="\$RPA\$ (\$r_s=$(rs)\$)",
            )
            ax6.plot(
                wns_over_EF,
                -imag(sigma_rpa_fl_wn_dyn) / EF;
                color="$(darkcolors[idx])",
                label="\$RPA+FL\$ (\$r_s=$(rs)\$)",
            )
            ax6.set_xlim(0, 20)
            # ax6.set_xlim(0, 50)
            ax6.set_ylim(0, 1.1 * max_tail_intersection)
            ax6.set_xlabel("\$\\omega_n / \\epsilon_F\$")
            ax6.set_ylabel(
                "\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_F\$",
            )
            ax6.legend(; loc="best")
            plt.tight_layout()
            fig6.savefig(
                "results/self_energy_fits/$(int_type)/im_sigma_and_tails_" *
                "rs=$(rs)_beta_ef=$(beta)_k=$(ktarget)_EF_$(int_type).pdf",
            )

            # -ωₙImΣ one-by-one
            peak_this_sigma = max(
                maximum(-imag(sigma_rpa_wn_dyn_over_EF) .* wns_over_EF),
                maximum(-imag(sigma_rpa_fl_wn_dyn_over_EF) .* wns_over_EF),
            )
            fig8, ax8 = plt.subplots()
            ax8.plot(
                wns_over_EF,
                (rpa_c1 / EF^2) * one.(wns_over_EF),
                # rpa_c1 * one.(wns_over_EF) / EF,
                "C$(idx-1)";
                linestyle="dashed",
                label="\$B_{RPA}\$ (\$r_s=$rs\$)",
                # label="\$C^{(1)}_{RPA} / \\epsilon^2_F\$ (\$r_s=$rs\$)",
            )
            ax8.plot(
                wns_over_EF,
                -imag(sigma_rpa_wn_dyn_over_EF) .* wns_over_EF,
                "C$(idx-1)";
                label="\$RPA\$ (\$r_s=$rs\$)",
            )
            ax8.plot(
                wns_over_EF,
                (rpa_fl_c1 / EF^2) * one.(wns_over_EF),
                # rpa_fl_c1 * one.(wns_over_EF) / EF,
                "$(darkcolors[idx])";
                linestyle="dashed",
                label="\$B_{RPA+FL}\$ (\$r_s=$rs\$)",
                # label="\$C^{(1)}_{RPA+FL} / \\epsilon^2_F\$ (\$r_s=$rs\$)",
            )
            ax8.plot(
                wns_over_EF,
                -imag(sigma_rpa_fl_wn_dyn_over_EF) .* wns_over_EF,
                "$(darkcolors[idx])";
                label="\$RPA+FL\$ (\$r_s=$rs\$)",
            )
            ax8.set_xlim(0, wns_over_EF[end])
            ax8.set_ylim(; bottom=0, top=1.1 * peak_this_sigma)
            ax8.legend(; loc="best")
            ax8.set_xlabel("\$\\omega_n / \\epsilon_F\$")
            ax8.set_ylabel(
                "\$-\\omega_n \\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon^2_F\$",
                # "\$-\\widetilde{omega}_n \\mathrm{Im}\\Sigma(k = $ktarget, i\\widetilde{omega}_n)\$",
            )
            plt.tight_layout()
            fig8.savefig(
                "results/self_energy_fits/$(int_type)/wn_times_im_sigma_" *
                "rs=$(rs)_beta_ef=$(beta)_k=$(ktarget)_EF_$(int_type).pdf",
            )

            # -ωₙImΣ one-by-one
            peak_this_sigma = max(
                maximum(-imag(sigma_rpa_wn_dyn_over_EF) .* wns_over_EF),
                maximum(-imag(sigma_rpa_fl_wn_dyn_over_EF) .* wns_over_EF),
            )
            fig8, ax8 = plt.subplots()
            ax8.plot(
                wns_over_EF,
                (rpa_c1 / EF^2) * one.(wns_over_EF),
                # rpa_c1 * one.(wns_over_EF) / EF,
                "C$(idx-1)";
                linestyle="dashed",
                label="\$B_{RPA}\$ (\$r_s=$rs\$)",
                # label="\$C^{(1)}_{RPA} / \\epsilon^2_F\$ (\$r_s=$rs\$)",
            )
            ax8.plot(
                wns_over_EF,
                -imag(sigma_rpa_wn_dyn_over_EF) .* wns_over_EF,
                "C$(idx-1)";
                label="\$RPA\$ (\$r_s=$rs\$)",
            )
            ax8.plot(
                wns_over_EF,
                (rpa_fl_c1 / EF^2) * one.(wns_over_EF),
                # rpa_fl_c1 * one.(wns_over_EF) / EF,
                "$(darkcolors[idx])";
                linestyle="dashed",
                label="\$B_{RPA+FL}\$ (\$r_s=$rs\$)",
                # label="\$C^{(1)}_{RPA+FL} / \\epsilon^2_F\$ (\$r_s=$rs\$)",
            )
            ax8.plot(
                wns_over_EF,
                -imag(sigma_rpa_fl_wn_dyn_over_EF) .* wns_over_EF,
                "$(darkcolors[idx])";
                label="\$RPA+FL\$ (\$r_s=$rs\$)",
            )
            ax8.set_xlim(0, wns_over_EF[end])
            ax8.set_ylim(; bottom=0, top=1.1 * peak_this_sigma)
            ax8.legend(; loc="best")
            ax8.set_xlabel("\$\\omega_n / \\epsilon_F\$")
            ax8.set_ylabel(
                "\$-\\omega_n \\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon^2_F\$",
                # "\$-\\widetilde{omega}_n \\mathrm{Im}\\Sigma(k = $ktarget, i\\widetilde{omega}_n)\$",
            )
            plt.tight_layout()
            fig8.savefig(
                "results/self_energy_fits/$(int_type)/wn_times_im_sigma_" *
                "rs=$(rs)_beta_ef=$(beta)_k=$(ktarget)_EF_$(int_type).pdf",
            )

            # Estimate measured B ≈ -(ω_max / EF) (ImΣ(ω_max) / EF)
            coeff_B_tail_rpa    = rpa_c1 / EF^2
            coeff_B_tail_rpa_fl = rpa_fl_c1 / EF^2
            coeff_B_meas_rpa    = -wns_over_EF[end] * imag(sigma_rpa_wn_dyn_over_EF[end])
            coeff_B_meas_rpa_fl = -wns_over_EF[end] * imag(sigma_rpa_fl_wn_dyn_over_EF[end])
            println(
                "(RPA) Percent error in B: ",
                100 * abs(coeff_B_meas_rpa - coeff_B_tail_rpa) / (coeff_B_tail_rpa),
            )
            println(
                "(RPA+FL) Percent error in B: ",
                100 * abs(coeff_B_meas_rpa_fl - coeff_B_tail_rpa_fl) /
                (coeff_B_tail_rpa_fl),
            )

            # # Get the local-field factor fs = Fs / NF
            # local fs_int_type
            # if int_type == :ko
            #     # NOTE: The Takada ansatz for fs is q-dependent!
            #     # fs_int_type = Interaction.landauParameterTakada(ktarget, 0, param)[1]
            #     fs_int_type = Interaction.landauParameterTakada(1e5, 0, param)[1]
            # elseif int_type == :ko_const
            #     # fs = Fs / NF
            #     Fs = get_Fs(rs)
            #     fs_int_type = Fs / param.NF
            # else
            #     error("Not yet implemented!")
            # end

            # Analytic coefficients D_{RPA(+FL)} from VZN paper
            alpha = (4 / 9π)^(1 / 3)
            coeff_D_rpa_exact = -(16 / 3π) * (alpha * rs)^2

            # Plot isolated 1/ω^{3/2} dependence of -ImΣ
            sigma_rpa_residual =
                -wns_over_EF .* imag(sigma_rpa_wn_dyn_over_EF) .- coeff_B_meas_rpa
            sigma_rpa_fl_residual =
                -wns_over_EF .* imag(sigma_rpa_fl_wn_dyn_over_EF) .- coeff_B_meas_rpa_fl
            @assert sigma_rpa_residual[end] == 0.0
            @assert sigma_rpa_fl_residual[end] == 0.0
            sigma_32_rpa = sigma_rpa_residual .* sqrt.(wns_over_EF)
            sigma_32_rpa_fl = sigma_rpa_fl_residual .* sqrt.(wns_over_EF)
            # The limiting values are D_{RPA(+FL)}
            coeff_D_meas_rpa    = sigma_32_rpa[end]
            coeff_D_meas_rpa_fl = sigma_32_rpa_fl[end]
            # Check percent error of D coefficients
            println(
                "(RPA) Percent error in D: ",
                100 * abs(coeff_D_meas_rpa - coeff_D_rpa_exact) / (coeff_D_rpa_exact),
            )
            # println(
            #     "(RPA+FL) Percent error in D: ",
            #     100 * abs(coeff_D_meas_rpa_fl - coeff_D_rpa_fl_exact) /
            #     (coeff_D_rpa_fl_exact),
            # )
            # Plot the 1/ω^{3/2} dependence of -ImΣ
            fig11, ax11 = plt.subplots()
            ax11.axhline(
                -coeff_D_rpa_exact;
                color="k",
                linestyle="--",
                label="\$\\frac{16}{3\\pi}\\left(\\alpha r_s\\right)^2\$",
            )
            # ax11.axhline(
            #     -coeff_D_meas_rpa;
            #     color="C$(idx-1)",
            #     linestyle="--",
            #     label="\$D_{RPA}\$ (\$r_s=$rs\$)",
            # )
            ax11.plot(
                wns_over_EF,
                -sigma_32_rpa;
                color="C$(idx-1)",
                label="\$RPA\$ (\$r_s=$rs\$)",
            )
            # ax11.axhline(
            #     -coeff_D_meas_rpa_fl;
            #     color="$(darkcolors[idx])",
            #     linestyle="--",
            #     label="\$D_{RPA+FL}\$ (\$r_s=$rs\$)",
            # )
            ax11.plot(
                wns_over_EF,
                -sigma_32_rpa_fl;
                color="$(darkcolors[idx])",
                label="\$RPA+FL\$ (\$r_s=$rs\$)",
            )
            # ax11.set_xlim(0, wns_over_EF[end])
            ax11.set_xlim(0, 50)
            # ax11.set_ylim(; bottom=0, top=1.1 * maximum(sigma_32_rpa))
            ax11.legend(; loc="best")
            ax11.set_xlabel("\$\\omega_n\$")
            ax11.set_ylabel(
                "\$\\sqrt{\\omega_n} \\left(\\omega_n \\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) - \\lim_{\\omega_n \\rightarrow \\infty}\\omega_n \\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\\right)\$",
                # "\$-\\sqrt{\\widetilde{\\omega}_n} \\left(\\widetilde{\\omega}_n \\mathrm{Im}\\widetilde{\\Sigma}(k = $ktarget, i\\widetilde{\\omega}_n) - \\lim_{\\widetilde{\\omega}_n \\rightarrow \\infty}\\widetilde{\\omega}_n \\mathrm{Im}\\widetilde{\\Sigma}(k = $ktarget, i\\widetilde{\\omega}_n)\\right)\$",
            )
            plt.tight_layout()
            fig11.savefig(
                "results/self_energy_fits/$(int_type)/sigma_wn32_dependence_" *
                "rs=$(rs)_beta_ef=$(beta)_k=$(ktarget)_EF_$(int_type).pdf",
            )
        end
    end
    # ax.set_xlim(0, 50)
    # ax.set_ylim(; bottom=0, top=1.1 * max_sigma)
    # ax.legend(; loc="best")
    # ax.set_xlabel("\$\\omega_n / \\epsilon_F\$")
    # if units == :Rydberg
    #     ax.set_ylabel("\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\$")
    # elseif units == :EF
    #     ax.set_ylabel("\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_F\$")
    # else  # units == :eTF
    #     ax.set_ylabel(
    #         "\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_{\\mathrm{TF}}\$",
    #     )
    # end
    # plt.tight_layout()
    # fig.savefig(
    #     "results/self_energy_fits/$(int_type)/im_sigma_" *
    #     "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_$(units)_$(int_type).pdf",
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
    ax7.plot(rslist, w0_over_EF_rpas ./ sqrt.(rslist), "o-"; color="C0", label="\$RPA\$")
    ax7.plot(
        rslist_big,
        model_rpa7.(rslist_big) ./ sqrt.(rslist_big),
        "--";
        color="C0",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\sqrt{r_s} $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s $(sgn_d_rpa) $(round(abs(d_rpa); sigdigits=3)) r_s \\sqrt{r_s}\$",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\sqrt{\\log r_s} $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) \\sqrt{r_s} $(sgn_d_rpa) $(round(abs(d_rpa); sigdigits=3)) \\sqrt{r_s}\$",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s $(sgn_d_rpa) $(round(abs(d_rpa); sigdigits=3)) r_s \\log r_s\$",
        label="\$\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s\\right)\$",
        # label="\$\\sqrt{r_s}\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s\\right)\$",
    )
    ax7.plot(
        rslist,
        w0_over_EF_rpa_fls ./ sqrt.(rslist),
        "o-";
        color="C1",
        label="\$RPA+FL\$",
    )
    ax7.plot(
        rslist_big,
        model_rpa_fl7.(rslist_big) ./ sqrt.(rslist_big),
        "--";
        color="C1",
        # label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\sqrt{r_s} $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s $(sgn_d_rpa_fl) $(round(abs(d_rpa_fl); sigdigits=3)) r_s \\sqrt{r_s}\$",
        # label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s $(sgn_d_rpa_fl) $(round(abs(d_rpa_fl); sigdigits=3)) r_s \\log r_s\$",
        label="\$\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s\\right)\$",
        # label="\$\\sqrt{r_s}\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s\\right)\$",
    )
    ax7.set_xlabel("\$r_s\$")
    ax7.set_ylabel("\$\\Omega_t(r_s) / \\epsilon_F\\sqrt{r_s}\$")
    # ax7.set_ylabel("\$\\Omega_t(r_s) / \\epsilon_F\\sqrt{r_s} = \\sqrt{B(r_s) / A(r_s)}\$")
    ax7.legend(; loc="best")
    plt.tight_layout()
    fig7.savefig(
        "results/self_energy_fits/$(int_type)/low_high_turning_points_over_sqrt_rs_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_EF_$(int_type).pdf",
    )

    # # Plot peak positions vs rs
    # println("sigma_rpa_peak_wns_over_EF = ", sigma_rpa_peak_wns_over_EF)
    # println("sigma_rpa_fl_peak_wns_over_EF = ", sigma_rpa_fl_peak_wns_over_EF)
    # fig2, ax2 = plt.subplots()
    # ax2.plot(rslist, sigma_rpa_peak_wns_over_EF, "o-"; color="C0", label="\$RPA\$")
    # ax2.plot(rslist, sigma_rpa_fl_peak_wns_over_EF, "o-"; color="C1", label="\$RPA+FL\$")
    # ax2.set_xlabel("\$r_s\$")
    # ax2.set_ylabel(
    #     "\${\\mathrm{argmax}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\\right\\rbrace / \\epsilon_F\$",
    # )
    # ax2.legend(; loc="best")
    # plt.tight_layout()
    # fig2.savefig(
    #     "results/self_energy_fits/$(int_type)/peak_positions_" *
    #     "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_EF_$(int_type).pdf",
    # )

    # Plot peak values vs rs
    println("sigma_rpa_peaks = ", sigma_rpa_peaks)
    println("sigma_rpa_fl_peaks = ", sigma_rpa_fl_peaks)
    # Least-squares fit to peak values
    # @. model3(rs, p) = rs^2 * (p[1] + p[2] * log(rs))
    @. model3(rs, p) = rs^2 * (p[1] + p[2] * log(rs) + p[3] * rs)
    # fit_rpa = curve_fit(model3, rslist, sigma_rpa_peaks, [1.0, 1.0])
    # fit_rpa_fl = curve_fit(model3, rslist, sigma_rpa_fl_peaks, [1.0, 1.0])
    fit_rpa = curve_fit(model3, rslist, sigma_rpa_peaks, [1.0, 1.0, 1.0])
    fit_rpa_fl = curve_fit(model3, rslist, sigma_rpa_fl_peaks, [1.0, 1.0, 1.0])
    model_rpa3(rs) = model3(rs, fit_rpa.param)
    model_rpa_fl3(rs) = model3(rs, fit_rpa_fl.param)
    # Coefficients of determination (r²)
    r2_rpa = rsquared(rslist, sigma_rpa_peaks, model_rpa3.(rslist))
    r2_rpa_fl = rsquared(rslist, sigma_rpa_fl_peaks, model_rpa_fl3.(rslist))
    println("RPA fit: ", fit_rpa.param, ", r² = $r2_rpa")
    println("RPA+FL fit: ", fit_rpa_fl.param, ", r² = $r2_rpa_fl")
    # a_rpa, b_rpa = fit_rpa.param
    # a_rpa_fl, b_rpa_fl = fit_rpa_fl.param
    a_rpa, b_rpa, c_rpa = fit_rpa.param
    a_rpa_fl, b_rpa_fl, c_rpa_fl = fit_rpa_fl.param
    sgn_b_rpa = b_rpa ≥ 0 ? "+" : "-"
    sgn_b_rpa_fl = b_rpa_fl ≥ 0 ? "+" : "-"
    sgn_c_rpa = c_rpa ≥ 0 ? "+" : "-"
    sgn_c_rpa_fl = c_rpa_fl ≥ 0 ? "+" : "-"
    fig3, ax3 = plt.subplots()
    ax3.plot(rslist, sigma_rpa_peaks ./ rslist .^ 2, "o-"; color="C0", label="\$RPA\$")
    ax3.plot(
        rslist_big,
        model_rpa3.(rslist_big) ./ rslist_big .^ 2,
        "--";
        color="C0",
        # label="\$r^2_s\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s\\right)\$",
        label="\$\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s\\right)\$",
        # label="\$r^2_s\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s\\right)\$",
        # label="\$$(round(a_rpa; sigdigits=3)) r_s $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) r^2_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r^3_s\$",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s^2\$",
    )
    ax3.plot(
        rslist,
        sigma_rpa_fl_peaks ./ rslist .^ 2,
        "o-";
        color="C1",
        label="\$RPA+FL\$",
    )
    ax3.plot(
        rslist_big,
        model_rpa_fl3.(rslist_big) ./ rslist_big .^ 2,
        "--";
        color="C1",
        # label="\$r^2_s\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s\\right)\$",
        label="\$\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s\\right)\$",
        # label="\$$(round(a_rpa_fl; sigdigits=3)) r_s $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) r^2_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r^3_s\$",
        # label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s^2\$",
    )
    ax3.set_xlabel("\$r_s\$")
    if units == :Rydberg
        ax3.set_ylabel(
            "\${\\mathrm{max}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\\right\\rbrace / r^2_s\$",
        )
    elseif units == :EF
        ax3.set_ylabel(
            "\${\\mathrm{max}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) \\right\\rbrace / \\epsilon_F r^2_s\$",
        )
    else  # units == :eTF
        ax3.set_ylabel(
            "\${\\mathrm{max}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) \\right\\rbrace / \\epsilon_{\\mathrm{TF}} r^2_s\$",
        )
    end
    ax3.legend(; loc="best")
    plt.tight_layout()
    fig3.savefig(
        "results/self_energy_fits/$(int_type)/peak_values_over_rs2_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_$(units)_$(int_type).pdf",
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
    ax10.plot(rslist, A_rpa ./ rslist, "o-"; color="C0", label="\$RPA\$")
    ax10.plot(
        rslist_big,
        model_rpa10.(rslist_big) ./ rslist_big,
        "--";
        color="C0",
        # label="\$r_s\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r^2_s\\right)\$",
        label="\$\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r^2_s\\right)\$",
        # label="\$r_s\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s\\right)\$",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s $(sgn_d_rpa) $(round(abs(d_rpa); sigdigits=3)) r_s \\log r_s\$",
    )
    ax10.plot(rslist, A_rpa_fl ./ rslist, "o-"; color="C1", label="\$RPA+FL\$")
    ax10.plot(
        rslist_big,
        model_rpa_fl10.(rslist_big) ./ rslist_big,
        "--";
        color="C1",
        # label="\$r_s\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r^2_s\\right)\$",
        label="\$\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r^2_s\\right)\$",
        # label="\$r_s\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s\\right)\$",
        # label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s $(sgn_d_rpa_fl) $(round(abs(d_rpa_fl); sigdigits=3)) r_s \\log r_s\$",
    )
    ax10.set_xlabel("\$r_s\$")
    ax10.set_ylabel("\$A(r_s) / r_s\$")
    # ax10.set_ylabel("\$A(r_s) = \\frac{1}{z(r_s)} - 1\$")
    ax10.legend(; loc="best")
    plt.tight_layout()
    fig10.savefig(
        "results/self_energy_fits/$(int_type)/A_over_rs_" *
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
    ax9.plot(rslist, c1_rpas_over_EF2 ./ rslist .^ 2, "o-"; color="C0", label="\$RPA\$")
    ax9.plot(
        rslist_big,
        model_rpa9.(rslist_big) ./ rslist_big .^ 2,
        "--";
        color="C0",
        # label="\$\\frac{8}{\\pi}(\\alpha r_s)^2\\left($(round(a_rpa; sigdigits=3)) - \\frac{\\log r_s}{2\\pi}\\right)\$",
        # label="\$r^2_s \\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s\\right)\$",
        label="\$\\left($(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s\\right)\$",
        # label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s^2\$",
        # label = "\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s $(sgn_d_rpa) $(round(abs(d_rpa); sigdigits=3)) r_s \\log r_s $(sgn_e_rpa) $(round(abs(e_rpa); sigdigits=3)) r_s^2\$",
    )
    ax9.plot(
        rslist,
        c1_rpa_fls_over_EF2 ./ rslist .^ 2,
        "o-";
        color="C1",
        label="\$RPA+FL\$",
    )
    ax9.plot(
        rslist_big,
        model_rpa_fl9.(rslist_big) ./ rslist_big .^ 2,
        "--";
        color="C1",
        # label="\$\\frac{8}{\\pi}(\\alpha r_s)^2\\left($(round(a_rpa_fl; sigdigits=3)) - \\frac{\\log r_s}{2\\pi}\\right)\$",
        # label="\$r^2_s \\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s\\right)\$",
        label="\$\\left($(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s\\right)\$",
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
    ax9.set_ylabel("\$B(r_s) / r^2_s\$")
    # ax9.set_ylabel("\$B(r_s) = C^{(1)}(r_s) / \\epsilon^2_{F}\$")
    plt.tight_layout()
    fig9.savefig(
        "results/self_energy_fits/$(int_type)/B_over_rs2_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_$(int_type).pdf",
    )

    # Low-frequency behavior of -ImΣ
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 1.5 * max_sigma)
    ax5.set_xlabel("\$\\omega_n / \\epsilon_F\$")
    ax5.set_ylabel("\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_F\$")
    ax5.legend(; loc="best")
    plt.tight_layout()
    # fig5.savefig(
    #     "results/self_energy_fits/$(int_type)/im_sigma_low_freq_" *
    #     "rs=$(round.(rslist_small; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_$(units)_$(int_type).pdf",
    # )

    return
end

main()
