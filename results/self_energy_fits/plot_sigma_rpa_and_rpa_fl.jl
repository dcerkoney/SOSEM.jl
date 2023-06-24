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

"""@ beta = 1000, same cutoffs as Σ"""
# const rpa_moments = Dict(
#     1.0 => 6.724025818442236,
#     2.0 => 1.4584933292439444,
#     5.0 => 0.1903829905411742,
#     10.0 => 0.040401482779086714,
# )
"""@ beta = 1000, huge cutoffs"""
const rpa_moments = Dict(
    1.0 => 6.81819756264996,
    2.0 => 1.4820362616264573,
    5.0 => 0.19414985801946902,
    10.0 => 0.04134319891382209,
)

"""@ beta = 1000, same cutoffs as Σ"""
# const rpa_fl_moments_ko_takada = Dict(
#     1.0 => 5.584998269054832,
#     2.0 => 1.1645705530347419,
#     5.0 => 0.1415640362155207,
#     10.0 => 0.028430803103939722,
# )
"""@ beta = 1000, huge cutoffs"""
const rpa_fl_moments_ko_takada = Dict(
    1.0 => 5.633206066182058,
    2.0 => 1.1748004545377624,
    5.0 => 0.1429082385231008,
    10.0 => 0.028748420315316787,
)

"""@ beta = 1000"""
# const rpa_fl_moments_ko_const =
#     Dict(1.0 => 26.20962808733749, 2.0 => 6.405695659453301, 5.0 => 1.0131504691181614)
# Dict(1.0 => 26.209244247361564, 2.0 => 6.405597087315782, 5.0 => 1.0131337017277458)
# const rpa_fl_moments_ko_const =
#     Dict(1.0 => 6.522450229958651, 2.0 => 1.3949688122808277, 5.0 => 0.17693882777214262)

# const rpa_moments = Dict(
#     1.0 => 6.841840386874177,
#     2.0 => 1.4879469677757577,
#     5.0 => 0.1950955710387994,
#     10.0 => 0.04157962718143167,
# )
# # 1000EF
# const rpa_moments = Dict(
#     1.0 => 6.817909128698095,
#     2.0 => 1.4819641531734868,
#     5.0 => 0.19413832062094072,
#     10.0 => 0.04134031456832556,
# )
# 1.0 => 6.6576879877687105,
# 2.0 => 1.441908895471402,
# 5.0 => 0.1877294925890505,
# 10.0 => 0.039738113070479135,
# )

# # # 1000EF
# const rpa_fl_moments_ko_takada = Dict(
#     1.0 => 5.633058721030482,
#     2.0 => 1.1747692177749955,
#     5.0 => 0.14290413621560322,
#     10.0 => 0.028747450611868244,
# )
# Dict(1.0 => 5.550864541581695, 2.0 => 1.157309342837587, 5.0 => 0.14060858522583466)
# Dict(1.0 => 5.550858338402098, 2.0 => 1.1573079578851801, 5.0 => 0.1406083932258038)
# const rpa_fl_moments_ko_takada = Dict(
#     1.0 => 6.733699951212692,
#     2.0 => 1.4669405347601496,
#     5.0 => 0.1933642135060841,
#     10.0 => 0.041487912673176086,
# )
# TODO: evaluate with integrate_c1_rpa_fl.jl
const rpa_moments = Dict(1.0 => 6.817909128698095, 2.0 => 0.0, 5.0 => 0.0, 10.0 => 0.0)
const rpa_fl_moments_takada = Dict(1.0 => 0.0, 2.0 => 0.0, 5.0 => 0.0, 10.0 => 0.0)

# TODO: debug ko_const int_type in ElectronGas and subtract tail of integrand / remap to finite interval
# const rpa_fl_moments_const = Dict(1.0 => 0.0, 2.0 => 0.0, 5.0 => 0.0, 10.0 => 0.0)

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
    return get_Fs_PW(rs)
end
# function get_Fs(rs)
#     if rs < 1.0 || rs > 5.0
#         return get_Fs_DMC(rs)
#     else
#         return get_Fs_PW(rs)
#     end
# end

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
    return (1 + fs) * rpa_tail
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
    #     Euv=10000 * param.EF,
    #     Nk=16,
    #     minK=1e-8 * param.kF,
    #     maxK=20 * param.kF,
    #     order=10,
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
    if haskey(zfactor_rpa_benchmarks, rs)
        zfactor_zero_temp = zfactor_rpa_benchmarks[rs]
        percent_error = 100 * abs(zfactor - zfactor_zero_temp) / zfactor_zero_temp
        println("Percent error vs zero-temperature benchmark of Z_kF: $percent_error")
        @assert percent_error ≤ 5
    end

    return kval, wns, sigma_wn_static_kval, sigma_wn_dynamic_kval
end

function get_sigma_rpa_fl_wn(param::Parameter.Para; ktarget=0.0, int_type=:ko, atol=1e-3)
    # Make sure we are using parameters for the bare UEG theory
    @assert param.Λs == param.Λa == 0.0

    # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
    rs = round(param.rs; sigdigits=13)
    Fs = get_Fs(rs)
    println("Fermi liquid parameter at rs = $(rs): Fs = $Fs")

    # Get RPA+FL self-energy
    # sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(param; int_type=:ko_const, Fs=Fs)
    sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(
        param;
        Euv=1000 * param.EF,
        Nk=16,
        minK=1e-8 * param.kF,
        maxK=30 * param.kF,
        order=10,
        int_type=int_type,
        Fs=-Fs,        # NOTE: sign convention differs in NEFT!
        bugfix=false,  # Test before/after bug
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

    # Get local RPA moment from HEG SOMS data for comparison
    c1l_rpa_hartrees = sosem_hartrees.get("rpa_a_T=0")[1]  # singleton array
    # TODO: Fix overall sign error on local moment in get_heg_som.py
    c1l_rpa_hartrees *= -1.0
    c1l_rpa = 4 * c1l_rpa_hartrees  # Hartree to Rydberg
    @assert c1l_rpa > 0

    # Compare heg_soms and VZN results for RPA local moment
    println("(VZN) C⁽¹⁾ˡ (RPA, rs = $(rs)): $c1l_rpa_vzn")
    println("(SOM) C⁽¹⁾ˡ (RPA, rs = $(rs)): $c1l_rpa")
    println("Relative error (SOM vs VZN): $(abs(c1l_rpa - c1l_rpa_vzn) / c1l_rpa_vzn)")

    # G₀W₀ asymptotics => all local tail!
    return c1l_rpa
    # return c1l_rpa_vzn
end

function get_rpa_fl_moments(param::Parameter.Para; k=0.0)
    # Make sure we are using parameters for the bare UEG theory
    @assert param.Λs == param.Λa == 0.0
    rs = round(param.rs; sigdigits=13)

    # Non-dimensionalize bare and RPA+FL non-local moments (stored in Hartree a.u.!)
    sosem_hartrees = np.load("results/data/soms_rs=$(rs)_beta_ef=40.0.npz")

    # Get local RPA+FL moment from HEG SOMS data (no VZN data available!)
    c1l_rpa_fl_hartrees = sosem_hartrees.get("rpa+fl_a_T=0")[1]  # singleton array
    # TODO: Fix overall sign error on local moment in get_heg_som.py
    c1l_rpa_fl_hartrees *= -1.0
    c1l_rpa_fl = 4 * c1l_rpa_fl_hartrees  # Hartree to Rydberg
    @assert c1l_rpa_fl > 0

    # Print heg_soms results for RPA+FL local moment
    println("(SOM) C⁽¹⁾ˡ (RPA+FL, rs = $(rs)): $c1l_rpa_fl")

    # G₀W₀+FL asymptotics => all local tail!
    return c1l_rpa_fl
end

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    ktarget = 0.0  # k = kF
    beta = 40.0
    rslist = [1.0, 5.0, 10.0]

    # rslist = [1.0]
    # rslist = [1 / α]  # Gives kF = EF = 1
    # rslist = [1.0, 2.0, 5.0]

    # The unit system to use for plotting
    units = :Rydberg
    # units = :EF
    # units = :eTF

    int_type = :ko
    # int_type = :ko_const

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

    # Filename for new JLD2 format
    filename =
    # "results/data/rs=$(rs_qmc)_beta_ef=$(beta_qmc)_" *
        "archives/rs=$(rs_qmc)_beta_ef=$(beta_qmc)_lambda=$(mass2)_" *
        "$(intn_str)$(solver)_with_ct_mu_lambda_archive1"

    # Load SOSEM data
    local param1, kgrid, k_kf_grid, c1l_N_mean, c1l_N_total, c1nl_N_total
    # Load the data for each observable
    f = jldopen("$filename.jld2", "r")
    param1 = f["c1d/N=$N/neval=$neval_c1d/param"]
    kgrid = f["c1d/N=$N/neval=$neval_c1d/kgrid"]
    c1l_N_total = f["c1l/N=$N/neval=$neval_c1l/meas"][1]
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
    k_kf_grid = kgrid / param1.kF

    # Get means and error bars from the result up to this order
    c1l_N_mean, c1l_N_stdev =
        Measurements.value.(c1l_N_total), Measurements.uncertainty.(c1l_N_total)
    c1nl_N_means, c1nl_N_stdevs =
        Measurements.value.(c1nl_N_total), Measurements.uncertainty.(c1nl_N_total)
    @assert length(k_kf_grid) == length(c1nl_N_means) == length(c1nl_N_stdevs)

    # The high-frequency tail of ImΣ(iωₙ) is -C⁽¹⁾ / ωₙ.
    iktarget = searchsortedfirst(kgrid, ktarget)

    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    darkcolors = ["midnightblue", "saddlebrown", "darkgreen", "darkred"]
    sigma_peak = 0.0
    peak_values_rpa = []
    peak_positions_rpa = []
    peak_values_rpa_fl = []
    peak_positions_rpa_fl = []
    for (i, rs) in enumerate(rslist)
        sigma_subpeak = 0.0
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
            get_sigma_rpa_fl_wn(param; ktarget=ktarget, int_type=int_type)

        # The static parts do not contribute to ImΣ
        println(imag(sigma_rpa_wn_stat))
        println(imag(sigma_rpa_fl_wn_stat))
        # @assert isapprox(imag(sigma_rpa_wn_stat), 0.0, atol=1e-4)
        # @assert isapprox(imag(sigma_rpa_fl_wn_stat), 0.0, atol=1e-4)

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
        rpa_c1 = rpa_moments[rs]
        if int_type == :ko
            rpa_fl_c1 = rpa_fl_moments_takada[rs]
        elseif int_type == :ko_const
            rpa_fl_c1 = rpa_fl_moments_const[rs]
        end
        # rpa_c1 = get_rpa_moments(param; k=ktarget)
        # rpa_fl_c1 = get_rpa_fl_moments(param; k=ktarget)
        println("First-order RPA moment (Rydberg): ", rpa_c1)
        println("First-order RPA+FL moment (Rydberg): ", rpa_fl_c1)

        # RPT tail calculation was nondimensionalized via division by eTF^2
        rpt_c1_over_eTF2 = (c1l_N_mean + c1nl_N_means[iktarget])
        rpt_err_c1_over_eTF2 = (c1l_N_stdev + c1nl_N_stdevs[iktarget])
        rpt_c1 = rpt_c1_over_eTF2 * eTF^2
        rpt_err_c1 = rpt_err_c1_over_eTF2 * eTF^2
        rpt_c1_over_EF2 = rpt_c1 / EF^2
        rpt_err_c1_over_EF2 = rpt_err_c1 / EF^2

        # Nondimensionalize moments in units of EF
        rpa_c1_over_EF2 = rpa_c1 / EF^2
        rpa_fl_c1_over_EF2 = rpa_fl_c1 / EF^2

        # Nondimensionalize moments in units of eTF
        rpa_c1_over_eTF2 = rpa_c1 / eTF^2
        rpa_fl_c1_over_eTF2 = rpa_fl_c1 / eTF^2

        # Nondimensionalize zeroth-order moment in units of EF
        hf_c0_over_EF = hf_c0 / EF

        # Nondimensionalize zeroth-order moment in units of eTF
        hf_c0_over_eTF = hf_c0 / eTF

        local wns_plot
        local rpt_c1_plot, rpt_c1_err_plot
        local rpa_c1_plot, rpa_fl_c1_plot
        local sigma_rpa_dyn_plot, sigma_rpa_fl_dyn_plot
        local sigma_rpa_stat_plot, sigma_rpa_fl_stat_plot
        # wns_plot = wns
        if units == :Rydberg
            wns_plot = wns
            # Zeroth-order moment
            hf_c0_plot = hf_c0
            # RPT moment
            rpt_c1_plot = rpt_c1
            rpt_c1_err_plot = rpt_err_c1
            # First-order moment tails
            rpa_c1_plot = rpa_c1
            rpa_fl_c1_plot = rpa_fl_c1
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
            # RPT tail
            rpt_c1_plot = rpt_c1_over_EF2
            rpt_c1_err_plot = rpt_err_c1_over_EF2
            # First-order moment tails
            rpa_c1_plot = rpa_c1_over_EF2
            rpa_fl_c1_plot = rpa_fl_c1_over_EF2
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
            # RPT tail
            rpt_c1_plot = rpt_c1_over_eTF2
            rpt_c1_err_plot = rpt_err_c1_over_eTF2
            # First-order moment tails
            rpa_c1_plot = rpa_c1_over_eTF2
            rpa_fl_c1_plot = rpa_fl_c1_over_eTF2
            # Static part
            sigma_rpa_stat_plot = sigma_rpa_wn_stat_over_eTF
            sigma_rpa_fl_stat_plot = sigma_rpa_fl_wn_stat_over_eTF
            # Dynamic part
            sigma_rpa_dyn_plot = sigma_rpa_wn_dyn_over_eTF
            sigma_rpa_fl_dyn_plot = sigma_rpa_fl_wn_dyn_over_eTF
        end
        if units != :Rydberg
            println("First-order RPA moment ($units): ", rpa_c1_plot)
            println("First-order RPA+FL moment ($units): ", rpa_fl_c1_plot)
        end
        println("(RPA) -ImΣ(0, 0) = ", -imag(sigma_rpa_dyn_plot[1]))
        println("(RPA+FL) -ImΣ(0, 0) = ", -imag(sigma_rpa_fl_dyn_plot[1]))

        ### Comparison of ReΣ to zeroth-order moment ###
        # ReΣ includes both static and dynamic contributions
        re_sigma_rpa = real(sigma_rpa_stat_plot) .+ real(sigma_rpa_dyn_plot)
        re_sigma_rpa_fl = real(sigma_rpa_fl_stat_plot) .+ real(sigma_rpa_fl_dyn_plot)
        # RPA(+FL) tails in chosen unit system
        ax2.plot(
            wns_over_EF,
            hf_c0_plot * one.(wns_plot),
            # "C$(i-1)";
            "k";
            linestyle="dashed",
            # label="\$C^{(0)}_{RPA(+FL)}(k) = \\frac{k^2}{2 m} + \\Sigma_{HF}(k)\$ (\$r_s=$rs\$)",
        )
        # RPA(+FL) in chosen unit system
        ax2.plot(wns_over_EF, re_sigma_rpa, "C$(i-1)"; label="\$RPA\$ (\$r_s=$rs\$)")
        ax2.plot(
            wns_over_EF,
            re_sigma_rpa_fl,
            "$(darkcolors[i])";
            label="\$RPA+FL\$ (\$r_s=$rs\$)",
        )

        labels = [
            [
                "\$C^{(1)}_{RPA} / \\omega_n\$ (\$r_s=$rs\$)",
                "\$C^{(1)}_{RPA+FL} / \\omega_n\$ (\$r_s=$rs\$)",
            ],
            [nothing, nothing],
        ]
        sigma_peak = max(
            sigma_peak,
            maximum(-imag(sigma_rpa_dyn_plot)),
            maximum(-imag(sigma_rpa_fl_dyn_plot)),
        )
        sigma_subpeak = max(
            sigma_subpeak,
            maximum(-imag(sigma_rpa_dyn_plot)),
            maximum(-imag(sigma_rpa_fl_dyn_plot)),
        )
        for (axis, axis_labels) in zip([ax, ax3], labels)
            ### Comparison of -ImΣ to RPA(+FL) asymptotics and RPT first-order moment tail ###
            # RPA -ImΣ and tail fit in chosen unit system
            axis.plot(
                wns_over_EF,
                rpa_c1_plot ./ wns_plot,
                "C$(i-1)";
                linestyle="dashed",
                label=axis_labels[1],
                # label="\$C^{(1)}_{RPA} / \\omega_n\$ (\$r_s=$rs\$)",
            )
            axis.plot(
                wns_over_EF,
                -imag(sigma_rpa_dyn_plot),
                "C$(i-1)";
                label="\$RPA\$ (\$r_s=$rs\$)",
            )
            # # RPA+FL -ImΣ and tail fit in chosen unit system
            axis.plot(
                wns_over_EF,
                rpa_fl_c1_plot ./ wns_plot,
                "$(darkcolors[i])";
                linestyle="dashed",
                label=axis_labels[2],
                # label="\$C^{(1)}_{RPA+FL} / \\omega_n\$ (\$r_s=$rs\$)",
            )
            axis.plot(
                wns_over_EF,
                -imag(sigma_rpa_fl_dyn_plot),
                "$(darkcolors[i])";
                label="\$RPA+FL\$ (\$r_s=$rs\$)",
            )
            # Analytic RPA(+FL) tail behaviors from VZN
            # rpa_analytic_tail = get_rpa_analytic_tail(wns_plot, param)
            # rpa_fl_analytic_tail = get_rpa_fl_analytic_tail(wns_plot, param)
            # axis.plot(
            #     wns_over_EF,
            #     rpa_analytic_tail,
            #     "C$(i-1)";
            #     linestyle="dotted",
            #     label="\$-\\frac{16\\sqrt{2}}{3\\pi}\\frac{(\\alpha r_s)^2}{\\omega^{3/2}_n}\$",
            # )
            # axis.plot(
            #     wns_over_EF,
            #     rpa_fl_analytic_tail,
            #     "$(darkcolors[i])";
            #     linestyle="dotted",
            #     label="\$-\\left(1 - f_s\\right)\\frac{16\\sqrt{2}}{3\\pi}\\frac{(\\alpha r_s)^2}{\\omega^{3/2}_n}\$",
            # )
        end

        # Plot of -ωₙImΣ vs C1 moments for RPA(+FL)
        fig4, ax4 = plt.subplots()
        ax4.plot(
            wns_over_EF,
            rpa_c1 * one.(wns) / EF,
            # rpa_c1 * one.(wns),
            "C$(i-1)";
            linestyle="dashed",
            # label=axis_labels[1],
            label="\$C^{(1)}_{RPA} / \\epsilon_F\$",
            # label="\$C^{(1)}_{RPA}\$ (\$r_s=$rs\$)",
        )
        ax4.plot(
            wns_over_EF,
            -imag(sigma_rpa_wn_dyn) .* wns_over_EF,
            # -imag(sigma_rpa_wn_dyn) .* wns,
            "C$(i-1)";
            label="\$RPA\$",
            # label="\$RPA\$ (\$r_s=$rs\$)",
        )
        ax4.plot(
            wns_over_EF,
            rpa_fl_c1 * one.(wns) / EF,
            # rpa_fl_c1 * one.(wns),
            "$(darkcolors[i])";
            linestyle="dashed",
            # label=axis_labels[2],
            label="\$C^{(1)}_{RPA+FL} / \\epsilon_F\$",
            # label="\$C^{(1)}_{RPA+FL}\$ (\$r_s=$rs\$)",
        )
        ax4.plot(
            wns_over_EF,
            -imag(sigma_rpa_fl_wn_dyn) .* wns_over_EF,
            # -imag(sigma_rpa_fl_wn_dyn) .* wns,
            "$(darkcolors[i])";
            label="\$RPA+FL\$",
            # label="\$RPA+FL\$ (\$r_s=$rs\$)",
        )
        # Compare -ωₙImΣ(k, iωₙ) and C⁽¹⁾'s for better resolution of the limiting behavior
        # ax4.set_xlim(0, 1000)
        ax4.set_xlim(0, wns_over_EF[end])
        # ax4.set_ylim(; bottom=-0.25, top=nothing)
        ax4.legend(; loc="best")
        ax4.set_xlabel("\$\\omega_n / \\epsilon_F\$")
        ax4.set_ylabel(
            "\$-\\omega_n \\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_F\$",
            # "\$-\\omega_n \\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\$",
        )
        plt.tight_layout()
        fig4.savefig(
            "results/self_energy_fits/$(int_type)/im_sigma_times_wn_over_EF_" *
            "rs=$(rs)_beta_ef=$(beta)_k=$(ktarget)_$(int_type).pdf",
        )

        # Include QMC result at rs = 1
        if rs == 1.0
            ax.plot(
                wns_over_EF,
                rpt_c1_plot ./ wns_plot,
                "k";
                linestyle="dashed",
                label="\$C^{(1)}_N / \\omega_n\$ (\$r_s = $rs_qmc, \\beta = $beta_qmc\$)",
            )
            ax.fill_between(
                wns_over_EF,
                (rpt_c1_plot - rpt_c1_err_plot) ./ wns_plot,
                (rpt_c1_plot + rpt_c1_err_plot) ./ wns_plot;
                color="k",
                alpha=0.4,
            )
        end

        # -ImΣ one-by-one
        ax.set_xlim(0, 100)
        ax.set_ylim(; bottom=0, top=1.1 * sigma_subpeak)
        ax.legend(; loc="best")
        ax.set_xlabel("\$\\omega_n / \\epsilon_F\$")
        if units == :Rydberg
            ax.set_ylabel("\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\$")
            ax2.set_ylabel("\$\\mathrm{Re}\\Sigma(k = $ktarget, i\\omega_n)\$")
        elseif units == :EF
            ax.set_ylabel(
                "\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_F\$",
            )
            ax2.set_ylabel(
                "\$\\mathrm{Re}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_F\$",
            )
        else  # units == :eTF
            ax.set_ylabel(
                "\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_{\\mathrm{TF}}\$",
            )
            ax2.set_ylabel(
                "\$\\mathrm{Re}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_{\\mathrm{TF}}\$",
            )
        end
        plt.tight_layout()
        fig.savefig(
            "results/high_frequency_tail/$(int_type)/im_sigma_tail_comparisons_" *
            "rs=$(rs)_beta_ef=$(beta)_k=$(ktarget)_$(units)_$(int_type).pdf",
        )

        # -ωₙImΣ one-by-one
        ax4.plot(
            wns_over_EF,
            rpa_c1 / EF,
            "C$(i-1)";
            linestyle="dashed",
            label="\$C^{(1)}_{RPA} / \\omega_n \\epsilon_F\$ (\$r_s=$rs\$)",
        )
        ax4.plot(
            wns_over_EF,
            -imag(sigma_rpa_wn_dyn) .* wns_over_EF,
            "C$(i-1)";
            label="\$RPA\$ (\$r_s=$rs\$)",
        )
        # # RPA+FL -ImΣ and tail fit in chosen unit system
        ax4.plot(
            wns_over_EF,
            rpa_fl_c1 / EF,
            "$(darkcolors[i])";
            linestyle="dashed",
            label="\$C^{(1)}_{RPA+FL} / \\omega_n \\epsilon_F\$ (\$r_s=$rs\$)",
        )
        ax4.plot(
            wns_over_EF,
            -imag(sigma_rpa_fl_wn_dyn) .* wns_over_EF,
            "$(darkcolors[i])";
            label="\$RPA+FL\$ (\$r_s=$rs\$)",
        )
        ax4.set_xlim(0, 1000)
        # ax4.set_xlim(0, wns_over_EF[end])
        ax4.set_ylim(; bottom=0, top=1.1 * sigma_subpeak)
        ax4.legend(; loc="best")
        ax4.set_xlabel("\$\\omega_n / \\epsilon_F\$")
        ax4.set_ylabel(
            "\$-\\omega_n \\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_F\$",
        )
        plt.tight_layout()
        fig4.savefig(
            "results/high_frequency_tail/$(int_type)/wn_times_im_sigma_" *
            "rs=$(rs)_beta_ef=$(beta)_k=$(ktarget)_EF_$(int_type).pdf",
        )

        # Add peak positions/values to list
        peak_value_rpa = maximum(-imag(sigma_rpa_wn_dyn))
        peak_value_rpa_fl = maximum(-imag(sigma_rpa_fl_wn_dyn))
        peak_position_rpa = wns_over_EF[argmax(-imag(sigma_rpa_wn_dyn))]
        peak_position_rpa_fl = wns_over_EF[argmax(-imag(sigma_rpa_fl_wn_dyn))]
        push!(peak_values_rpa, peak_value_rpa)
        push!(peak_values_rpa_fl, peak_value_rpa_fl)
        push!(peak_positions_rpa, peak_position_rpa)
        push!(peak_positions_rpa_fl, peak_position_rpa_fl)
    end

    # Plot peak values
    fig5, ax5 = plt.subplots()
    ax5.plot(rslist, peak_values_rpa; label="RPA")
    ax5.plot(rslist, peak_values_rpa_fl; label="RPA+FL")
    ax5.set_xlabel(
        "\$\\mathrm{max}_{\\omega_n} \\big|\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\\big|\$",
    )
    ax5.legend(; loc="best")
    plt.tight_layout()
    fig5.savefig(
        "results/high_frequency_tail/$(int_type)/peak_values_" *
        "rs=$(rslist)_beta_ef=$(beta)_k=$(ktarget)_$(int_type).pdf",
    )

    # Plot peak positions
    fig6, ax6 = plt.subplots()
    ax6.plot(rslist, peak_positions_rpa; label="RPA")
    ax6.plot(rslist, peak_positions_rpa_fl; label="RPA+FL")
    ax6.set_xlabel(
        "\$\\mathrm{argmax}_{\\omega_n} \\big|\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\\big| / \\epsilon_F\$",
    )
    ax6.legend(; loc="best")
    plt.tight_layout()
    fig6.savefig(
        "results/high_frequency_tail/$(int_type)/peak_positions_" *
        "rs=$(rslist)_beta_ef=$(beta)_k=$(ktarget)_$(int_type).pdf",
    )

    # ReΣ together
    ax2.set_xlabel("\$\\omega_n / \\epsilon_F\$")
    ax2.set_xlim(0, 50)
    ax2.legend(; loc="best")
    fig2.savefig(
        "results/high_frequency_tail/$(int_type)/re_sigma_tail_comparisons_" *
        "rs=$(rslist)_beta_ef=$(beta)_k=$(ktarget)_$(units)_$(int_type).pdf",
    )

    # ImΣ together
    ax3.set_xlim(0, 50)
    ax3.set_ylim(; bottom=0, top=1.1 * sigma_peak)
    ax3.legend(; loc="best")
    ax3.set_xlabel("\$\\omega_n / \\epsilon_F\$")
    if units == :Rydberg
        ax3.set_ylabel("\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n)\$")
    elseif units == :EF
        ax3.set_ylabel("\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_F\$")
    else  # units == :eTF
        ax3.set_ylabel(
            "\$-\\mathrm{Im}\\Sigma(k = $ktarget, i\\omega_n) / \\epsilon_{\\mathrm{TF}}\$",
        )
    end
    plt.tight_layout()
    fig3.savefig(
        "results/high_frequency_tail/$(int_type)/im_sigma_tail_comparisons_" *
        "rs=$(rslist)_beta_ef=$(beta)_k=$(ktarget)_$(units)_$(int_type).pdf",
    )
    plt.close("all")

    return
end

main()
