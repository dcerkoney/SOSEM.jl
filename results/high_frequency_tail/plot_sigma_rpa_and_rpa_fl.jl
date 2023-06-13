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

function load_csv(filename)
    # assumes csv format: (x, y)
    d = readdlm(filename, ',')
    @assert ndims(d) == 2
    xdata = d[:, 1]
    ydata = d[:, 2]
    return xdata, ydata
end

"""
Exact expression for the Fock self-energy
in terms of the dimensionless Lindhard function.
"""
function fock_self_energy_exact(k, p::Parameter.Para)
    # The (dimensionful) value at k = 0 is minus the Thomas-Fermi energy
    eTF = p.qTF^2 / (2 * p.me)
    return -eTF * UEG_MC.lindhard(k / p.kF)
end

"""
Exact expression for the Fock quasiparticle energy
in terms of the dimensionless Lindhard function.
"""
function qp_fock_exact(k, p::Parameter.Para)
    return k^2 / (2 * p.me) + fock_self_energy_exact(k, p)
end

function get_Fs(rs)
    if rs < 1.0 || rs > 5.0
        return get_Fs_DMC(rs)
    else
        return get_Fs_PW(rs)
    end
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
    return (1 - fs) * rpa_tail
end

function get_sigma_rpa_wn(param::Parameter.Para; ktarget=0.0, atol=1e-3)
    # Make sure we are using parameters for the bare UEG theory
    @assert param.Λs == param.Λa == 0.0

    # Get RPA+FL self-energy
    sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(param; int_type=:rpa)
    #     param;
    #     Euv=1000 * param.EF,
    #     Nk=12,
    #     minK=1e-8 * param.kF,
    #     maxK=50 * param.kF,
    #     order=6,
    #     int_type=:rpa,
    # )

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

    # Check zfactor against benchmarks at β = 1000. 
    # It should agree within a few percent due to finite-temperature effects.
    # (see: https://numericaleft.github.io/ElectronGas.jl/dev/manual/quasiparticle/)
    rs = round(param.rs; sigdigits=13)
    zfactor = SelfEnergy.zfactor(param, sigma_wn_dynamic)[1]
    zfactor_zero_temp = zfactor_rpa_benchmarks[rs]
    percent_error = 100 * abs(zfactor - zfactor_zero_temp) / zfactor_zero_temp
    println("Percent error vs zero-temperature benchmark of Z_kF: $percent_error")
    @assert percent_error ≤ 5

    return kval, wns, sigma_wn_static_kval, sigma_wn_dynamic_kval
end

function get_sigma_rpa_fl_wn(param::Parameter.Para; ktarget=0.0, atol=1e-3)
    # Make sure we are using parameters for the bare UEG theory
    @assert param.Λs == param.Λa == 0.0

    # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
    rs = round(param.rs; sigdigits=13)
    Fs = get_Fs(rs)
    println("Fermi liquid parameter at rs = $(rs): Fs = $Fs")

    # Get RPA+FL self-energy
    sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(param; int_type=:ko_const, Fs=Fs)
    #     param;
    #     Euv=1000 * param.EF,
    #     Nk=12,
    #     minK=1e-8 * param.kF,
    #     maxK=50 * param.kF,
    #     order=6,
    #     int_type=:ko_const,
    #     Fs=Fs,
    # )

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

    return kval, wns, sigma_wn_static_kval, sigma_wn_dynamic_kval
end

function get_rpa_moments(param::Parameter.Para; k=0.0)
    # Make sure we are using parameters for the bare UEG theory
    @assert param.Λs == param.Λa == 0.0
    rs = round(param.rs; sigdigits=13)

    # Get RPA value of the local moment at this rs from VZN paper
    k_kf_grid_vzn, c1l_rpa_over_rs2_vzn = load_csv("$vzn_dir/c1l_over_rs2_rpa.csv")
    P = sortperm(k_kf_grid_vzn)
    c1l_rpa_over_rs2_vzn_interp = linear_interpolation(
        k_kf_grid_vzn[P],
        c1l_rpa_over_rs2_vzn[P];
        extrapolation_bc=Line(),
    )
    # VZN work in units of EF & kF ⟹ convert back to Rydbergs
    c1l_rpa_vzn = c1l_rpa_over_rs2_vzn_interp(rs) * rs^2 * param.EF^2

    # Non-dimensionalize bare and RPA+FL non-local moments (stored in Hartree a.u.!)
    sosem_hartrees = np.load("results/data/soms_rs=$(rs)_beta_ef=40.0.npz")

    # Momentum grid for HEG SOMS data
    kgrid = np.linspace(0.0, 3.0 * param.kF; num=600)

    # RPA kgrid and results (stored in Hartree a.u.)
    c1nl_rpa = (
        sosem_hartrees.get("rpa_b") +
        sosem_hartrees.get("bare_c") +
        sosem_hartrees.get("bare_d")
    )
    # Error bar comes from δC⁽¹ᵇ⁾_{RPA} only
    c1nl_rpa_err = sosem_hartrees.get("rpa_b_err")

    # Hartree to Rydberg conversions:
    # • Zeroth-order:
    #       [c0] = [E] ⟹ c0 E_H = c0 E_Ry * (E_H / E_Ry) = 2 c0 E_Ry
    # • First-order:
    #       [c1nl] = [c1l] = [E]² ⟹ c1nl (E_H)² = c1nl (E_Ry)² * (E_H / E_Ry)² = 4 c1nl (E_Ry)²
    c1nl_rpa *= 4
    c1nl_rpa_err *= 4

    # Linearly interpolate data to evaluate exactly at momentum k
    c1nl_rpa_interp = linear_interpolation(kgrid, c1nl_rpa; extrapolation_bc=Line())
    c1nl_rpa_err_interp = linear_interpolation(kgrid, c1nl_rpa_err; extrapolation_bc=Line())

    # RPA(+FL) means and error bars
    c1nl_rpa_mean = c1nl_rpa_interp(k)
    c1nl_rpa_stdev = c1nl_rpa_err_interp(k)

    # Get local RPA moment from HEG SOMS data for comparison
    c1l_rpa_hartrees = sosem_hartrees.get("rpa_a_T=0")[1]  # singleton array
    # TODO: Fix overall sign error on local moment in get_heg_som.py
    c1l_rpa_hartrees *= -1.0
    c1l_rpa = 4 * c1l_rpa_hartrees  # Hartree to Rydberg
    @assert c1l_rpa > 0

    # Compare heg_soms and VZN results for RPA local moment
    println("(VZN) C⁽¹⁾ˡ (RPA, rs = $(rs)): $c1l_rpa_vzn")
    println("(SOM) C⁽¹⁾ˡ (RPA, rs = $(rs)): $c1l_rpa")
    println(
        "(SOM) C⁽¹⁾ⁿˡ(k = $(k)) (RPA, rs = $(rs)): $(c1nl_rpa_mean) ± $(c1nl_rpa_stdev)",
    )

    # NOTE: Mixing VZN and SOSEM data here
    # TODO: Benchmark local RPA moment against VZN to check our systematic errors!
    # rpa_c1 = c1l_rpa_over_eTF2_vzn + c1nl_rpa_mean
    # rpa_c1_err = c1nl_rpa_stdevs[1]

    # Tail fit to -ImΣ using RPA results
    rpa_c1 = c1l_rpa + c1nl_rpa_mean
    rpa_c1_err = c1nl_rpa_stdev
    return rpa_c1, rpa_c1_err
end

function get_rpa_fl_moments(param::Parameter.Para; k=0.0)
    # Make sure we are using parameters for the bare UEG theory
    @assert param.Λs == param.Λa == 0.0
    rs = round(param.rs; sigdigits=13)

    # Non-dimensionalize bare and RPA+FL non-local moments (stored in Hartree a.u.!)
    sosem_hartrees = np.load("results/data/soms_rs=$(rs)_beta_ef=40.0.npz")

    # Momentum grid for HEG SOMS data
    kgrid = np.linspace(0.0, 3.0 * param.kF; num=600)

    # Bare and RPA(+FL) results (stored in Hartree a.u.)
    c1nl_rpa_fl =
        sosem_hartrees.get("rpa+fl_b") +
        sosem_hartrees.get("bare_c") +
        sosem_hartrees.get("bare_d")
    # Error bar comes from δC⁽¹ᵇ⁾_{RPA+FL} only
    c1nl_rpa_fl_err = sosem_hartrees.get("rpa+fl_b_err")

    # Hartree to Rydberg conversions:
    # • Zeroth-order:
    #       [c0] = [E] ⟹ c0 E_H = c0 E_Ry * (E_H / E_Ry) = 2 c0 E_Ry
    # • First-order:
    #       [c1nl] = [c1l] = [E]² ⟹ c1nl (E_H)² = c1nl (E_Ry)² * (E_H / E_Ry)² = 4 c1nl (E_Ry)²
    c1nl_rpa_fl *= 4
    c1nl_rpa_fl_err *= 4

    # Linearly interpolate data to evaluate exactly at momentum k
    c1nl_rpa_fl_interp = linear_interpolation(kgrid, c1nl_rpa_fl; extrapolation_bc=Line())
    c1nl_rpa_fl_err_interp =
        linear_interpolation(kgrid, c1nl_rpa_fl_err; extrapolation_bc=Line())

    # RPA(+FL) means and error bars
    c1nl_rpa_fl_mean = c1nl_rpa_fl_interp(k)
    c1nl_rpa_fl_stdev = c1nl_rpa_fl_err_interp(k)

    # Get local RPA+FL moment from HEG SOMS data (no VZN data available!)
    c1l_rpa_fl_hartrees = sosem_hartrees.get("rpa+fl_a_T=0")[1]  # singleton array
    # TODO: Fix overall sign error on local moment in get_heg_som.py
    c1l_rpa_fl_hartrees *= -1.0
    c1l_rpa_fl = 4 * c1l_rpa_fl_hartrees  # Hartree to Rydberg
    @assert c1l_rpa_fl > 0

    # Print heg_soms results for RPA+FL local moment
    println("(SOM) C⁽¹⁾ˡ (RPA+FL, rs = $(rs)): $c1l_rpa_fl")
    println(
        "(SOM) C⁽¹⁾ⁿˡ(k = $(k)) (RPA+FL, rs = $(rs): $(c1nl_rpa_fl_mean) ± $(c1nl_rpa_fl_stdev)",
    )

    # Tail fit to -ImΣ using RPA+FL results
    rpa_fl_c1 = c1l_rpa_fl + c1nl_rpa_fl_mean
    rpa_fl_c1_err = c1nl_rpa_fl_stdev
    return rpa_fl_c1, rpa_fl_c1_err
end

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    ktarget = 0.0  # k = kF
    beta = 40.0
    # rslist = [1.0]
    # rslist = [1 / α]  # Gives kF = EF = 1
    rslist = [1.0, 2.0, 5.0]

    # The unit system to use for plotting
    # units = :Rydberg
    units = :EF
    # units = :eTF

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    fig2, ax2 = plt.subplots()
    darkcolors = ["midnightblue", "saddlebrown", "darkgreen", "darkred"]
    for (i, rs) in enumerate(rslist)
        sigma_peak = 0.0
        fig, ax = plt.subplots()

        println("\nPlotting data for rs = $rs...")
        # Get the G0W0 self-energy and corresponding DLR grid from the ElectronGas package
        # NOTE: Here we need to be careful to generate Σ_G0W0 for the *bare* theory, i.e.,
        #       to use an ElectronGas.Parameter object where Λs (mass2) is zero!
        param = Parameter.rydbergUnit(1 / beta, rs, 3)
        @assert param.Λs == param.Λa == 0.0

        # Get RPA and RPA+FL self-energies
        kval, wns, sigma_rpa_wn_stat, sigma_rpa_wn_dyn =
            get_sigma_rpa_wn(param; ktarget=ktarget)
        kval_fl, wns_fl, sigma_rpa_fl_wn_stat, sigma_rpa_fl_wn_dyn =
            get_sigma_rpa_fl_wn(param; ktarget=ktarget)

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

        # The zeroth-order RPA(+FL) moment is the same as in
        # Hartree-Fock, i.e., C⁽⁰⁾_RPA(k) = ϵ₀(k) + Σ_HF(k).
        hf_c0 = qp_fock_exact(ktarget, param)

        # Get first-order RPA and RPA+FL moments at k = ktarget
        rpa_c1, rpa_c1_err = get_rpa_moments(param; k=ktarget)
        rpa_fl_c1, rpa_fl_c1_err = get_rpa_fl_moments(param; k=ktarget)
        println("First-order RPA moment (Rydberg): ", rpa_c1, " ± ", rpa_c1_err)
        println("First-order RPA+FL moment (Rydberg): ", rpa_fl_c1, " ± ", rpa_fl_c1_err)

        # Nondimensionalize moments in units of EF
        rpa_c1_over_EF2 = rpa_c1 / EF^2
        rpa_c1_err_over_EF2 = rpa_c1_err / EF^2
        rpa_fl_c1_over_EF2 = rpa_fl_c1 / EF^2
        rpa_fl_c1_err_over_EF2 = rpa_fl_c1_err / EF^2

        # Nondimensionalize moments in units of eTF
        rpa_c1_over_eTF2 = rpa_c1 / eTF^2
        rpa_c1_err_over_eTF2 = rpa_c1_err / eTF^2
        rpa_fl_c1_over_eTF2 = rpa_fl_c1 / eTF^2
        rpa_fl_c1_err_over_eTF2 = rpa_fl_c1_err / eTF^2

        # Nondimensionalize zeroth-order moment in units of EF
        hf_c0_over_EF = hf_c0 / EF

        # Nondimensionalize zeroth-order moment in units of eTF
        hf_c0_over_eTF = hf_c0 / eTF

        local wns_plot
        local rpa_c1_plot, rpa_c1_err_plot
        local rpa_fl_c1_plot, rpa_fl_c1_err_plot
        local sigma_rpa_dyn_plot, sigma_rpa_fl_dyn_plot
        local sigma_rpa_stat_plot, sigma_rpa_fl_stat_plot
        # wns_plot = wns
        if units == :Rydberg
            wns_plot = wns
            # Zeroth-order moment
            hf_c0_plot = hf_c0
            # First-order moment tails
            rpa_c1_plot = rpa_c1
            rpa_c1_err_plot = rpa_c1_err
            rpa_fl_c1_plot = rpa_fl_c1
            rpa_fl_c1_err_plot = rpa_fl_c1_err
            # Static part
            sigma_rpa_stat_plot = sigma_rpa_wn_stat
            sigma_rpa_fl_stat_plot = sigma_rpa_fl_wn_stat
            # Dynamic part
            sigma_rpa_dyn_plot = sigma_rpa_wn_dyn
            sigma_rpa_fl_dyn_plot = sigma_rpa_fl_wn_dyn
        elseif units == :EF
            wns_plot = wns_over_EF
            # Zeroth-order moment
            hf_c0_plot = hf_c0_over_EF
            # First-order moment tails
            rpa_c1_plot = rpa_c1_over_EF2
            rpa_c1_err_plot = rpa_c1_err_over_EF2
            rpa_fl_c1_plot = rpa_fl_c1_over_EF2
            rpa_fl_c1_err_plot = rpa_fl_c1_err_over_EF2
            # Static part
            sigma_rpa_stat_plot = sigma_rpa_wn_stat_over_EF
            sigma_rpa_fl_stat_plot = sigma_rpa_fl_wn_stat_over_EF
            # Dynamic part
            sigma_rpa_dyn_plot = sigma_rpa_wn_dyn_over_EF
            sigma_rpa_fl_dyn_plot = sigma_rpa_fl_wn_dyn_over_EF
        else  # units == :eTF
            wns_plot = wns_over_eTF
            # Zeroth-order moment
            hf_c0_plot = hf_c0_over_eTF
            # First-order moment tails
            rpa_c1_plot = rpa_c1_over_eTF2
            rpa_c1_err_plot = rpa_c1_err_over_eTF2
            rpa_fl_c1_plot = rpa_fl_c1_over_eTF2
            rpa_fl_c1_err_plot = rpa_fl_c1_err_over_eTF2
            # Static part
            sigma_rpa_stat_plot = sigma_rpa_wn_stat_over_eTF
            sigma_rpa_fl_stat_plot = sigma_rpa_fl_wn_stat_over_eTF
            # Dynamic part
            sigma_rpa_dyn_plot = sigma_rpa_wn_dyn_over_eTF
            sigma_rpa_fl_dyn_plot = sigma_rpa_fl_wn_dyn_over_eTF
        end
        if units != :Rydberg
            println(
                "First-order RPA moment ($units): ",
                rpa_c1_plot,
                " ± ",
                rpa_c1_err_plot,
            )
            println(
                "First-order RPA+FL moment ($units): ",
                rpa_fl_c1_plot,
                " ± ",
                rpa_fl_c1_err_plot,
            )
        end
        println("(RPA) -ImΣ(0, 0) = ", -imag(sigma_rpa_dyn_plot[1]))
        println("(RPA+FL) -ImΣ(0, 0) = ", -imag(sigma_rpa_fl_dyn_plot[1]))

        # TODO: Include QMC result at rs = 1, try sign flip on c1b3+

        ### Comparison of ReΣ to zeroth-order moment ###
        # ReΣ includes both static and dynamic contributions
        re_sigma_rpa = real(sigma_rpa_stat_plot) .+ real(sigma_rpa_dyn_plot)
        re_sigma_rpa_fl = real(sigma_rpa_fl_stat_plot) .+ real(sigma_rpa_fl_dyn_plot)
        # RPA(+FL) tails in chosen unit system
        ax2.plot(
            wns_plot,
            hf_c0_plot * one.(wns_plot),
            # "C$(i-1)";
            "k";
            linestyle="dashed",
            # label="\$C^{(0)}_{RPA(+FL)}(k) = \\frac{k^2}{2 m} + \\Sigma_{HF}(k)\$ (\$r_s=$rs\$)",
        )
        # RPA(+FL) in chosen unit system
        ax2.plot(wns_plot, re_sigma_rpa, "C$(i-1)"; label="\$RPA\$ (\$r_s=$rs\$)")
        ax2.plot(
            wns_plot,
            re_sigma_rpa_fl,
            "$(darkcolors[i])";
            label="\$RPA+FL\$ (\$r_s=$rs\$)",
        )

        ### Comparison of -ImΣ to first-order moment tails ###
        sigma_peak = max(
            sigma_peak,
            maximum(-imag(sigma_rpa_dyn_plot)),
            maximum(-imag(sigma_rpa_fl_dyn_plot)),
        )
        # RPA -ImΣ and tail fit in chosen unit system
        ax.plot(
            wns_plot,
            rpa_c1_plot ./ wns_plot,
            "C$(i-1)";
            linestyle="dashed",
            label="\$C^{(1)}_{RPA} / \\omega_n\$ (\$r_s=$rs\$)",
        )
        ax.fill_between(
            wns_plot,
            (rpa_c1_plot - rpa_c1_err_plot) ./ wns_plot,
            (rpa_c1_plot + rpa_c1_err_plot) ./ wns_plot;
            color="C$(i-1)",
            alpha=0.4,
        )
        ax.plot(
            wns_plot,
            -imag(sigma_rpa_dyn_plot),
            "C$(i-1)";
            label="\$RPA\$ (\$r_s=$rs\$)",
        )
        # RPA+FL -ImΣ and tail fit in chosen unit system
        ax.plot(
            wns_plot,
            rpa_fl_c1_plot ./ wns_plot,
            "$(darkcolors[i])";
            linestyle="dashed",
            label="\$C^{(1)}_{RPA+FL} / \\omega_n\$ (\$r_s=$rs\$)",
        )
        ax.fill_between(
            wns_plot,
            (rpa_fl_c1_plot - rpa_fl_c1_err_plot) ./ wns_plot,
            (rpa_fl_c1_plot + rpa_fl_c1_err_plot) ./ wns_plot;
            color="$(darkcolors[i])",
            alpha=0.4,
        )
        ax.plot(
            wns_plot,
            -imag(sigma_rpa_fl_dyn_plot),
            "$(darkcolors[i])";
            label="\$RPA+FL\$ (\$r_s=$rs\$)",
        )
        # Analytic RPA(+FL) tail behaviors from VZN
        # rpa_analytic_tail = get_rpa_analytic_tail(wns_plot, param)
        # rpa_fl_analytic_tail = get_rpa_fl_analytic_tail(wns_plot, param)
        # ax.plot(
        #     wns_plot / param.EF,
        #     rpa_analytic_tail,
        #     "C$(i-1)";
        #     linestyle="dotted",
        #     label="\$-\\frac{16\\sqrt{2}}{3\\pi}\\frac{(\\alpha r_s)^2}{\\omega^{3/2}_n}\$",
        # )
        # ax.plot(
        #     wns_plot / param.EF,
        #     rpa_fl_analytic_tail,
        #     "$(darkcolors[i])";
        #     linestyle="dotted",
        #     label="\$-\\left(1 - f_s\\right)\\frac{16\\sqrt{2}}{3\\pi}\\frac{(\\alpha r_s)^2}{\\omega^{3/2}_n}\$",
        # )
        ax.set_xlim(0, 50)
        ax.set_ylim(; bottom=0, top=1.1 * sigma_peak)
        # ax2.set_ylim(; bottom=0, top=0.35)
        ax.legend(; loc="best")
        # ax.set_xlabel("\$\\omega_n / \\epsilon_F\$")
        if units == :Rydberg
            ax.set_xlabel("\$\\omega_n\$")
            ax2.set_xlabel("\$\\omega_n\$")
            ax.set_ylabel("\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\$")
            ax2.set_ylabel("\$\\mathrm{Re}\\Sigma(k = $ktarget, i\\omega_n)\$")
        elseif units == :EF
            ax.set_xlabel("\$\\omega_n / \\epsilon_F\$")
            ax2.set_xlabel("\$\\omega_n / \\epsilon_F\$")
            ax.set_ylabel(
                "\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_F\$",
            )
            ax2.set_ylabel(
                "\$\\mathrm{Re}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_F\$",
            )
        else  # units == :eTF
            ax.set_xlabel("\$\\omega_n / \\epsilon_{\\mathrm{TF}}\$")
            ax2.set_xlabel("\$\\omega_n / \\epsilon_{\\mathrm{TF}}\$")
            ax.set_ylabel(
                "\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_{\\mathrm{TF}}\$",
            )
            ax2.set_ylabel(
                "\$\\mathrm{Re}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_{\\mathrm{TF}}\$",
            )
        end
        plt.tight_layout()
        fig.savefig(
            "results/high_frequency_tail/im_sigma_tail_comparisons_" *
            "rs=$(rs)_beta_ef=$(beta)_k=$(ktarget)_$(units).pdf",
        )
    end
    ax2.set_xlim(0, 50)
    ax2.legend(; loc="best")
    ax2.set_xlabel("\$\\omega_n / \\epsilon_F\$")
    fig2.savefig(
        "results/high_frequency_tail/re_sigma_tail_comparisons_" *
        "rs=$(rslist)_beta_ef=$(beta)_k=$(ktarget)_$(units).pdf",
    )
    plt.close("all")

    # TODO: Compare -ωₙImΣ(k, iωₙ) and C⁽¹⁾'s for better resolution of the limiting behavior
    # ...

    return
end

main()
