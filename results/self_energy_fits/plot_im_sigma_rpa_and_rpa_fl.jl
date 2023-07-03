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
    rslist = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
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
                -imag(sigma_rpa_wn_dyn) / EF;
                color="C$(idx-1)",
                label="\$RPA\$ (\$r_s=$(rs)\$)",
            )
            ax5.plot(
                wns_over_EF,
                -imag(sigma_rpa_fl_wn_dyn) / EF;
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

        # Plot -ImΣ, low and high frequency tails together at each rs
        if rs in rslist_small
            fig6, ax6 = plt.subplots()
            ax6.axvline(
                # w0_rpa / eTF;
                w0_rpa / EF;
                color="gray",
                linestyle="-",
                label="\$\\omega_0\$ (\$RPA\$)",
            )
            ax6.axvline(
                # w0_rpa_fl / eTF;
                w0_rpa_fl / EF;
                color="k",
                linestyle="-",
                label="\$\\omega_0\$ (\$RPA+FL\$)",
            )
            # Set the upper ylim as 10% larger than the tail intersection
            max_tail_intersection = max(
                # coeff_low_freq_rpa * w0_rpa / eTF,
                # coeff_low_freq_rpa_fl * w0_rpa_fl / eTF,
                coeff_low_freq_rpa * w0_rpa / EF,
                coeff_low_freq_rpa_fl * w0_rpa_fl / EF,
            )
            # High-frequency behavior of -ImΣ
            ax6.plot(
                # wns_over_eTF,
                # coeff_high_freq_rpa ./ wns ./ eTF;
                wns_over_EF,
                coeff_high_freq_rpa ./ wns ./ EF;
                color="C$(idx-1)",
                linestyle="--",
            )
            ax6.plot(
                # wns_over_eTF,
                # coeff_high_freq_rpa_fl ./ wns ./ eTF;
                wns_over_EF,
                coeff_high_freq_rpa_fl ./ wns ./ EF;
                color="$(darkcolors[idx])",
                linestyle="--",
            )
            # Low-frequency behavior of -ImΣ
            ax6.plot(
                # wns_over_eTF,
                # coeff_low_freq_rpa .* wns_over_eTF;
                wns_over_EF,
                coeff_low_freq_rpa .* wns_over_EF;
                color="C$(idx-1)",
                linestyle="--",
            )
            ax6.plot(
                # wns_over_eTF,
                # coeff_low_freq_rpa_fl .* wns_over_eTF;
                wns_over_EF,
                coeff_low_freq_rpa_fl .* wns_over_EF;
                color="$(darkcolors[idx])",
                linestyle="--",
            )
            # -ImΣ
            ax6.plot(
                # wns_over_eTF,
                # -imag(sigma_rpa_wn_dyn) / eTF;
                wns_over_EF,
                -imag(sigma_rpa_wn_dyn) / EF;
                color="C$(idx-1)",
                label="\$RPA\$ (\$r_s=$(rs)\$)",
            )
            ax6.plot(
                # wns_over_eTF,
                # -imag(sigma_rpa_fl_wn_dyn) / eTF;
                wns_over_EF,
                -imag(sigma_rpa_fl_wn_dyn) / EF;
                color="$(darkcolors[idx])",
                label="\$RPA+FL\$ (\$r_s=$(rs)\$)",
            )
            # ax6.set_xlim(0, 20)
            ax6.set_xlim(0, 50)
            ax6.set_ylim(0, 1.1 * max_tail_intersection)
            # ax6.set_xlabel("\$\\omega_n / \\epsilon_{\\mathrm{TF}}\$")
            # ax6.set_ylabel(
            #     "\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_{\\mathrm{TF}}\$",
            # )
            ax6.set_xlabel("\$\\omega_n / \\epsilon_F\$")
            ax6.set_ylabel(
                "\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_F\$",
            )
            ax6.legend(; loc="best")
            plt.tight_layout()
            # fig6.savefig(
            #     "results/self_energy_fits/$(int_type)/im_sigma_and_tails_" *
            #     "rs=$(rs)_beta_ef=$(beta)_k=$(ktarget)_eTF_$(int_type).pdf",
            # )
            fig6.savefig(
                "results/self_energy_fits/$(int_type)/im_sigma_and_tails_" *
                "rs=$(rs)_beta_ef=$(beta)_k=$(ktarget)_EF_$(int_type).pdf",
            )

            # -ωₙImΣ one-by-one
            peak_this_sigma = max(
                maximum(-imag(sigma_rpa_wn_dyn) .* wns_over_EF),
                maximum(-imag(sigma_rpa_fl_wn_dyn) .* wns_over_EF),
            )
            fig8, ax8 = plt.subplots()
            ax8.plot(
                wns_over_EF,
                rpa_c1 * one.(wns_over_EF) / EF,
                "C$(idx-1)";
                linestyle="dashed",
                label="\$C^{(1)}_{RPA} / \\omega_n \\epsilon_F\$ (\$r_s=$rs\$)",
            )
            ax8.plot(
                wns_over_EF,
                -imag(sigma_rpa_wn_dyn) .* wns_over_EF,
                "C$(idx-1)";
                label="\$RPA\$ (\$r_s=$rs\$)",
            )
            ax8.plot(
                wns_over_EF,
                rpa_fl_c1 * one.(wns_over_EF) / EF,
                "$(darkcolors[idx])";
                linestyle="dashed",
                label="\$C^{(1)}_{RPA+FL} / \\omega_n \\epsilon_F\$ (\$r_s=$rs\$)",
            )
            ax8.plot(
                wns_over_EF,
                -imag(sigma_rpa_fl_wn_dyn) .* wns_over_EF,
                "$(darkcolors[idx])";
                label="\$RPA+FL\$ (\$r_s=$rs\$)",
            )
            ax8.set_xlim(0, wns_over_EF[end])
            ax8.set_ylim(; bottom=0, top=1.1 * peak_this_sigma)
            ax8.legend(; loc="best")
            ax8.set_xlabel("\$\\omega_n / \\epsilon_F\$")
            ax8.set_ylabel(
                "\$-\\omega_n \\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_F\$",
            )
            plt.tight_layout()
            fig8.savefig(
                "results/self_energy_fits/$(int_type)/wn_times_im_sigma_" *
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
    rslist_big = LinRange(0.1, 10.0, 600)

    # Plot low/high-frequency turning points vs rs
    fig7, ax7 = plt.subplots()
    # Least-squares fit to low-frequency turning points
    @. model7(rs, p) = p[1] + p[2] * log(rs) + p[3] * rs
    fit_rpa = curve_fit(model7, rslist, w0_over_EF_rpas, [1.5, 0.5, 0.25])
    fit_rpa_fl = curve_fit(model7, rslist, w0_over_EF_rpa_fls, [1.5, 0.5, 0.25])
    model_rpa7(rs) = model7(rs, fit_rpa.param)
    model_rpa_fl7(rs) = model7(rs, fit_rpa_fl.param)
    # Coefficients of determination (r²)
    r2_rpa = rsquared(rslist, w0_over_EF_rpas, model_rpa7.(rslist))
    r2_rpa_fl = rsquared(rslist, w0_over_EF_rpa_fls, model_rpa_fl7.(rslist))
    println("RPA fit: ", fit_rpa.param, ", r² = $r2_rpa")
    println("RPA+FL fit: ", fit_rpa_fl.param, ", r² = $r2_rpa_fl")
    a_rpa, b_rpa, c_rpa = fit_rpa.param
    a_rpa_fl, b_rpa_fl, c_rpa_fl = fit_rpa_fl.param
    sgn_b_rpa = b_rpa ≥ 0 ? "+" : "-"
    sgn_b_rpa_fl = b_rpa_fl ≥ 0 ? "+" : "-"
    sgn_c_rpa = c_rpa ≥ 0 ? "+" : "-"
    sgn_c_rpa_fl = c_rpa_fl ≥ 0 ? "+" : "-"
    ax7.plot(rslist, w0_over_EF_rpas, "o-"; color="C0", label="\$RPA\$")
    ax7.plot(
        rslist_big,
        model_rpa7.(rslist_big),
        "--";
        color="C0",
        label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s\$",
    )
    ax7.plot(rslist, w0_over_EF_rpa_fls, "o-"; color="C1", label="\$RPA+FL\$")
    ax7.plot(
        rslist_big,
        model_rpa_fl7.(rslist_big),
        "--";
        color="C1",
        label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s\$",
    )
    ax7.set_xlabel("\$r_s\$")
    ax7.set_ylabel("\$\\omega_0 / \\epsilon_F = \\sqrt{A(r_s) / B(r_s)}\$")
    ax7.legend(; loc="best")
    plt.tight_layout()
    fig7.savefig(
        "results/self_energy_fits/$(int_type)/low_high_turning_points_" *
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
    @. model3(rs, p) = p[1] + p[2] * rs + p[3] * rs^2
    fit_rpa = curve_fit(model3, rslist, sigma_rpa_peaks, [1.0, 1.0, 1.0])
    fit_rpa_fl = curve_fit(model3, rslist, sigma_rpa_fl_peaks, [1.0, 1.0, 1.0])
    model_rpa3(rs) = model3(rs, fit_rpa.param)
    model_rpa_fl3(rs) = model3(rs, fit_rpa_fl.param)
    # Coefficients of determination (r²)
    r2_rpa = rsquared(rslist, sigma_rpa_peaks, model_rpa3.(rslist))
    r2_rpa_fl = rsquared(rslist, sigma_rpa_fl_peaks, model_rpa_fl3.(rslist))
    println("RPA fit: ", fit_rpa.param, ", r² = $r2_rpa")
    println("RPA+FL fit: ", fit_rpa_fl.param, ", r² = $r2_rpa_fl")
    a_rpa, b_rpa, c_rpa = fit_rpa.param
    a_rpa_fl, b_rpa_fl, c_rpa_fl = fit_rpa_fl.param
    sgn_b_rpa = b_rpa ≥ 0 ? "+" : "-"
    sgn_b_rpa_fl = b_rpa_fl ≥ 0 ? "+" : "-"
    sgn_c_rpa = c_rpa ≥ 0 ? "+" : "-"
    sgn_c_rpa_fl = c_rpa_fl ≥ 0 ? "+" : "-"
    fig3, ax3 = plt.subplots()
    ax3.plot(rslist, sigma_rpa_peaks, "o-"; color="C0", label="\$RPA\$")
    ax3.plot(
        rslist_big,
        model_rpa3.(rslist_big),
        "--";
        color="C0",
        label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s^2\$",
    )
    ax3.plot(rslist, sigma_rpa_fl_peaks, "o-"; color="C1", label="\$RPA+FL\$")
    ax3.plot(
        rslist_big,
        model_rpa_fl3.(rslist_big),
        "--";
        color="C1",
        label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s^2\$",
    )
    if units == :Rydberg
        a = 0.3456675143953697
        b = -0.14002149217828802
        peak_fit(rs) = a + b * log(rs)
        ax3.plot(
            rslist,
            peak_fit.(rslist),
            "--";
            color="C0",
            label="\$0.346 - 0.14 \\log r_s\$",
        )
    end
    ax3.set_xlabel("\$r_s\$")
    if units == :Rydberg
        ax3.set_ylabel(
            "\${\\mathrm{max}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\\right\\rbrace\$",
        )
    elseif units == :EF
        ax3.set_ylabel(
            "\${\\mathrm{max}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) \\right\\rbrace / \\epsilon_F\$",
        )
    else  # units == :eTF
        ax3.set_ylabel(
            "\${\\mathrm{max}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) \\right\\rbrace / \\epsilon_{\\mathrm{TF}}\$",
        )
    end
    ax3.legend(; loc="best")
    plt.tight_layout()
    fig3.savefig(
        "results/self_energy_fits/$(int_type)/peak_values_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_$(units)_$(int_type).pdf",
    )

    # Plot Z-factors vs rs for RPA(+FL)
    println("\nZ_RPA:\n", zfactors_rpa)
    println("Z_RPA+FL:\n", zfactors_rpa_fl)
    # Least-squares fit to Z-factors
    @. model4(rs, p) = p[1] + p[2] * log(rs) + p[3] * rs
    fit_rpa = curve_fit(model4, rslist, zfactors_rpa, [1.0, 1.0, 1.0])
    fit_rpa_fl = curve_fit(model4, rslist, zfactors_rpa_fl, [1.0, 1.0, 1.0])
    model_rpa4(rs) = model4(rs, fit_rpa.param)
    model_rpa_fl4(rs) = model4(rs, fit_rpa_fl.param)
    # Coefficients of determination (r²)
    r2_rpa = rsquared(rslist, zfactors_rpa, model_rpa4.(rslist))
    r2_rpa_fl = rsquared(rslist, zfactors_rpa_fl, model_rpa_fl4.(rslist))
    println("RPA fit: ", fit_rpa.param, ", r² = $r2_rpa")
    println("RPA+FL fit: ", fit_rpa_fl.param, ", r² = $r2_rpa_fl")
    a_rpa, b_rpa, c_rpa = fit_rpa.param
    a_rpa_fl, b_rpa_fl, c_rpa_fl = fit_rpa_fl.param
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
        label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s\$",
    )
    ax4.plot(rslist, zfactors_rpa_fl, "o-"; color="C1", label="\$RPA+FL\$")
    ax4.plot(
        rslist_big,
        model_rpa_fl4.(rslist_big),
        "--";
        color="C1",
        label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s\$",
    )
    ax4.set_xlabel("\$r_s\$")
    ax4.set_ylabel("\$Z_{k_F}\$")
    ax4.legend(; loc="best")
    plt.tight_layout()
    fig4.savefig(
        "results/self_energy_fits/$(int_type)/zfactor_k=kF_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_$(int_type).pdf",
    )

    # Plot B(rs) for RPA(+FL)
    B_rpa = 1 ./ zfactors_rpa .- 1
    B_rpa_fl = 1 ./ zfactors_rpa_fl .- 1
    # Least-squares fit to Z-factors
    @. model10(rs, p) = p[1] + p[2] * log(rs) + p[3] * rs
    fit_rpa = curve_fit(model10, rslist, B_rpa, [1.0, 1.0, 1.0])
    fit_rpa_fl = curve_fit(model10, rslist, B_rpa_fl, [1.0, 1.0, 1.0])
    model_rpa10(rs) = model10(rs, fit_rpa.param)
    model_rpa_fl10(rs) = model10(rs, fit_rpa_fl.param)
    # Coefficients of determination (r²)
    r2_rpa = rsquared(rslist, B_rpa, model_rpa10.(rslist))
    r2_rpa_fl = rsquared(rslist, B_rpa_fl, model_rpa_fl10.(rslist))
    println("RPA fit: ", fit_rpa.param, ", r² = $r2_rpa")
    println("RPA+FL fit: ", fit_rpa_fl.param, ", r² = $r2_rpa_fl")
    a_rpa, b_rpa, c_rpa = fit_rpa.param
    a_rpa_fl, b_rpa_fl, c_rpa_fl = fit_rpa_fl.param
    sgn_b_rpa = b_rpa ≥ 0 ? "+" : "-"
    sgn_b_rpa_fl = b_rpa_fl ≥ 0 ? "+" : "-"
    sgn_c_rpa = c_rpa ≥ 0 ? "+" : "-"
    sgn_c_rpa_fl = c_rpa_fl ≥ 0 ? "+" : "-"
    fig10, ax10 = plt.subplots()
    ax10.plot(rslist, B_rpa, "o-"; color="C0", label="\$RPA\$")
    ax10.plot(
        rslist_big,
        model_rpa10.(rslist_big),
        "--";
        color="C0",
        label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s\$",
    )
    ax10.plot(rslist, B_rpa_fl, "o-"; color="C1", label="\$RPA+FL\$")
    ax10.plot(
        rslist_big,
        model_rpa_fl10.(rslist_big),
        "--";
        color="C1",
        label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s\$",
    )
    ax10.set_xlabel("\$r_s\$")
    ax10.set_ylabel("\$B(r_s)\$")
    ax10.legend(; loc="best")
    plt.tight_layout()
    fig10.savefig(
        "results/self_energy_fits/$(int_type)/B_k=kF_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_$(int_type).pdf",
    )

    # Plot second-order moments vs rs for RPA(+FL)
    fig9, ax9 = plt.subplots()

    # Least-squares fit to A(rs) = C^{(1)} / EF^2
    @. model9(rs, p) = p[1] + p[2] * rs + p[3] * rs^2
    # @. model9(rs, p) = p[1] + p[2] * log(rs) + p[3] * rs + p[4] * rs * log(rs) + p[5] * rs^2
    fit_rpa = curve_fit(model9, rslist, c1_rpas_over_EF2, [1.0, 1.0, 1.0])
    fit_rpa_fl = curve_fit(model9, rslist, c1_rpa_fls_over_EF2, [1.0, 1.0, 1.0])
    # fit_rpa = curve_fit(model9, rslist, c1_rpas_over_EF2, [1.0, 1.0, 1.0, 1.0, 1.0])
    # fit_rpa_fl = curve_fit(model9, rslist, c1_rpa_fls_over_EF2, [1.0, 1.0, 1.0, 1.0, 1.0])
    model_rpa9(rs) = model9(rs, fit_rpa.param)
    model_rpa_fl9(rs) = model9(rs, fit_rpa_fl.param)
    # Coefficients of determination (r²)
    r2_rpa = rsquared(rslist, c1_rpas_over_EF2, model_rpa9.(rslist))
    r2_rpa_fl = rsquared(rslist, c1_rpa_fls_over_EF2, model_rpa_fl9.(rslist))
    println("RPA fit: ", fit_rpa.param, ", r² = $r2_rpa")
    println("RPA+FL fit: ", fit_rpa_fl.param, ", r² = $r2_rpa_fl")
    a_rpa, b_rpa, c_rpa = fit_rpa.param
    a_rpa_fl, b_rpa_fl, c_rpa_fl = fit_rpa_fl.param
    sgn_b_rpa = b_rpa ≥ 0 ? "+" : "-"
    sgn_b_rpa_fl = b_rpa_fl ≥ 0 ? "+" : "-"
    sgn_c_rpa = c_rpa ≥ 0 ? "+" : "-"
    sgn_c_rpa_fl = c_rpa_fl ≥ 0 ? "+" : "-"
    # a_rpa, b_rpa, c_rpa, d_rpa, e_rpa = fit_rpa.param
    # a_rpa_fl, b_rpa_fl, c_rpa_fl, d_rpa_fl, e_rpa_fl = fit_rpa_fl.param
    # sgn_b_rpa = b_rpa ≥ 0 ? "+" : "-"
    # sgn_b_rpa_fl = b_rpa_fl ≥ 0 ? "+" : "-"
    # sgn_c_rpa = c_rpa ≥ 0 ? "+" : "-"
    # sgn_c_rpa_fl = c_rpa_fl ≥ 0 ? "+" : "-"
    # sgn_d_rpa = d_rpa ≥ 0 ? "+" : "-"
    # sgn_d_rpa_fl = d_rpa_fl ≥ 0 ? "+" : "-"
    # sgn_e_rpa = e_rpa ≥ 0 ? "+" : "-"
    # sgn_e_rpa_fl = e_rpa_fl ≥ 0 ? "+" : "-"
    ax9.plot(rslist, c1_rpas_over_EF2, "o-"; color="C0", label="\$RPA\$")
    ax9.plot(
        rslist_big,
        model_rpa9.(rslist_big),
        "--";
        color="C0",
        label="\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s^2\$",
        # label = "\$$(round(a_rpa; sigdigits=3)) $(sgn_b_rpa) $(round(abs(b_rpa); sigdigits=3)) \\log r_s $(sgn_c_rpa) $(round(abs(c_rpa); sigdigits=3)) r_s $(sgn_d_rpa) $(round(abs(d_rpa); sigdigits=3)) r_s \\log r_s $(sgn_e_rpa) $(round(abs(e_rpa); sigdigits=3)) r_s^2\$",
    )
    ax9.plot(rslist, c1_rpa_fls_over_EF2, "o-"; color="C1", label="\$RPA+FL\$")
    ax9.plot(
        rslist_big,
        model_rpa_fl9.(rslist_big),
        "--";
        color="C1",
        label="\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s^2\$",
        # label = "\$$(round(a_rpa_fl; sigdigits=3)) $(sgn_b_rpa_fl) $(round(abs(b_rpa_fl); sigdigits=3)) \\log r_s $(sgn_c_rpa_fl) $(round(abs(c_rpa_fl); sigdigits=3)) r_s $(sgn_d_rpa_fl) $(round(abs(d_rpa_fl); sigdigits=3)) r_s \\log r_s $(sgn_e_rpa_fl) $(round(abs(e_rpa_fl); sigdigits=3)) r_s^2\$",
    )
    ax9.set_xlabel("\$r_s\$")
    ax9.set_ylabel("\$C^{(1)} / \\epsilon^2_{F}\$")
    ax9.legend(; loc="best")
    plt.tight_layout()
    fig9.savefig(
        "results/self_energy_fits/$(int_type)/second_order_moments_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_EF_$(int_type).pdf",
        )
    ax9.set_ylabel("\$A(r_s)\$")
    fig9.savefig(
        "results/self_energy_fits/$(int_type)/A_" *
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
