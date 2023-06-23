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
    if rs < 1.0 || rs > 5.0
        @warn "The Perdew-Wang interpolation for Fs may " *
              "be inaccurate outside the metallic regime!"
    end
    kappa0_over_kappa = 1.0025 - 0.1721rs - 0.0036rs^2
    # F⁰ₛ = κ₀/κ - 1
    return kappa0_over_kappa - 1.0
end

function get_sigma_rpa_wn(param::Parameter.Para; ktarget=0.0, atol=1e-3)
    # Make sure we are using parameters for the bare UEG theory
    @assert param.Λs == param.Λa == 0.0

    # Get RPA+FL self-energy
    sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(
        param;
        Euv=1000 * param.EF,
        Nk=16,
        minK=1e-8 * param.kF,
        maxK=30 * param.kF,
        order=10,
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

    # Get RPA+FL self-energy
    sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(
        param;
        Euv=1000 * param.EF,
        Nk=16,
        minK=1e-8 * param.kF,
        maxK=30 * param.kF,
        order=10,
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
    rslist = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]

    # rslist = [1.0, 5.0, 10.0]
    # rslist = [1.0, 1ℯ, 1ℯ^2]
    # rsstrings = ["1", "e", "e^2"]

    # The unit system to use for plotting
    units = :Rydberg
    # units = :EF
    # units = :eTF

    int_type = :ko
    # int_type = :ko_const

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    darkcolors = ["midnightblue", "saddlebrown", "darkgreen", "darkred"]
    max_sigma = 0.0
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
            idx = findfirst(rslist .== rs)
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

        # Peak positions
        push!(sigma_rpa_peak_wns_over_EF, wns[argmax(-imag(sigma_rpa_wn_dyn))] / EF)
        push!(sigma_rpa_fl_peak_wns_over_EF, wns[argmax(-imag(sigma_rpa_fl_wn_dyn))] / EF)
        # Peak values
        push!(sigma_rpa_peaks, maximum(-imag(sigma_rpa_dyn_plot)))
        push!(sigma_rpa_fl_peaks, maximum(-imag(sigma_rpa_fl_dyn_plot)))
        # Z-factors
        push!(zfactors_rpa, zfactor_rpa)
        push!(zfactors_rpa_fl, zfactor_rpa_fl)
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

    # Plot peak positions vs rs
    println("sigma_rpa_peak_wns_over_EF = ", sigma_rpa_peak_wns_over_EF)
    println("sigma_rpa_fl_peak_wns_over_EF = ", sigma_rpa_fl_peak_wns_over_EF)
    fig2, ax2 = plt.subplots()
    ax2.plot(rslist, sigma_rpa_peak_wns_over_EF, "o-"; color="C0", label="\$RPA\$")
    ax2.plot(rslist, sigma_rpa_fl_peak_wns_over_EF, "o-"; color="C1", label="\$RPA+FL\$")
    ax2.set_xlabel("\$r_s\$")
    ax2.set_ylabel(
        "\${\\mathrm{argmax}}_{\\omega_n}\\left\\lbrace-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\\right\\rbrace / \\epsilon_F\$",
    )
    ax2.legend(; loc="best")
    plt.tight_layout()
    fig2.savefig(
        "results/self_energy_fits/$(int_type)/peak_positions_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_EF_$(int_type).pdf",
    )

    a = 0.3456675143953697
    b = -0.14002149217828802
    peak_fit(rs) = a + b * log(rs)

    # Plot peak values vs rs
    println("sigma_rpa_peaks = ", sigma_rpa_peaks)
    println("sigma_rpa_fl_peaks = ", sigma_rpa_fl_peaks)
    fig3, ax3 = plt.subplots()
    ax3.plot(rslist, sigma_rpa_peaks, "o-"; color="C0", label="\$RPA\$")
    ax3.plot(rslist, sigma_rpa_fl_peaks, "o-"; color="C1", label="\$RPA+FL\$")
    ax3.plot(
        rslist,
        peak_fit.(rslist),
        "--";
        color="C0",
        label="\$0.346 - 0.14 \\log r_s\$",
    )
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
    fig4, ax4 = plt.subplots()
    ax4.plot(rslist, zfactors_rpa, "o-"; color="C0", label="\$RPA\$")
    ax4.plot(rslist, zfactors_rpa_fl, "o-"; color="C1", label="\$RPA+FL\$")
    ax4.set_xlabel("\$r_s\$")
    ax4.set_ylabel("\$Z_{k_F}\$")
    ax4.legend(; loc="best")
    plt.tight_layout()
    fig4.savefig(
        "results/self_energy_fits/$(int_type)/zfactor_k=kF_" *
        "rs=$(round.(rslist; sigdigits=3))_beta_ef=$(beta)_$(int_type).pdf",
    )

    # Low-frequency behavior of -ImΣ
    ax5.set_xlim(0, 5)
    ax5.set_xlabel("\$\\omega_n / \\epsilon_F\$")
    ax5.set_ylabel("\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_F\$")
    ax5.legend(; loc="best")
    plt.tight_layout()
    fig5.savefig(
        "results/self_energy_fits/$(int_type)/im_sigma_low_freq_" *
        "rs=$(round.(rslist_small; sigdigits=3))_beta_ef=$(beta)_k=$(ktarget)_$(units)_$(int_type).pdf",
    )

    return
end

main()
