using CodecZlib
using DataStructures
using DelimitedFiles
using ElectronLiquid
using ElectronGas
using GreenFunc
using Interpolations
using JLD2
using Measurements
using PyCall
using PyPlot
using SOSEM
using LsqFit

# For style "science"
@pyimport scienceplots

# For saving/loading numpy data
@pyimport numpy as np
# @pyimport scipy.interpolate as interp

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

# Sunset diverging colour scheme from https://personal.sron.nl/~pault/
const cdict2 = OrderedDict([
    1 => "#364B9A",
    2 => "#4A7BB7",
    3 => "#6EA6CD",
    4 => "#98CAE1",
    5 => "#C2E4EF",
    6 => "#EAECCC",
    7 => "#FEDA8B",
    8 => "#FDB366",
    9 => "#F67E4B",
    10 => "#DD3D2D",
    11 => "#A50026",
]);

# Light qualitative colour scheme from https://personal.sron.nl/~pault/
const cdict3 = OrderedDict([
    "light blue" => "#77AADD",
    "light cyan" => "#99DDFF",
    "mint" => "#44BB99",
    "pear" => "#BBCC33",
    "olive" => "#AAAA00",
    "light yellow" => "#EEDD88",
    "orange" => "#EE8866",
    "pink" => "#FFAABB",
    "pale grey" => "#DDDDDD",
]);

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
# Nk, order = 20, 12
const rpa_sosem = Dict(
    0.01 => 133629.5589712659,
    0.1 => 996.1453748797056,
    0.5 => 30.65472082096144,
    1.0 => 6.724025817780453,
    2.0 => 1.4584933290770838,
    3.0 => 0.5936165034488587,
    4.0 => 0.3130622525590106,
    5.0 => 0.19038299054878535,
    7.5 => 0.0769478880550561,
    10.0 => 0.0404014827795728,
)
const rpa_fl_sosem_ko_takada = Dict(
    0.01 => 113120.53305565231,
    0.1 => 794.0844302101557,
    0.5 => 22.706345661868387,
    1.0 => 4.739108972017714,
    2.0 => 0.9584480419440262,
    3.0 => 0.3703011474216666,
    4.0 => 0.18744320831614875,
    5.0 => 0.1102609051385906,
    7.5 => 0.04190513631989273,
    10.0 => 0.02107028506216691,
)

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
    # Nk, order = 11, 8
    Nk, order = 20, 12

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
    # Nk, order = 11, 8
    Nk, order = 20, 12

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
        "k",
        cdict["orange"],
        cdict["blue"],
        cdict["cyan"],
        cdict["magenta"],
        cdict["red"],
        # cdict["teal"],
    ]
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    # Use LaTex fonts for plots
    rcParams["font.size"] = 16
    rcParams["mathtext.fontset"] = "cm"

    # Physical parameters
    rslist = [1.0]
    # rslist = [1.0, 5.0, 10.0]
    beta = 1000.0

    # We fit Σ_RPA(k = 0, iw)
    ktarget = 0.0

    # Which fs paramaterization to use
    int_type = :ko
    # The data for B_{RPA+FL} depends on int_type
    local rpa_fl_sosem
    if int_type == :ko
        rpa_fl_sosem = rpa_fl_sosem_ko_takada
    elseif int_type == :ko_const
        rpa_fl_sosem = rpa_fl_sosem_ko_const
    else
        SOSEM.@todo
    end

    modes = [:rpa, :rpa_fl]
    int_types = [:rpa, int_type]
    modestrings = ["RPA", "RPA+FL"]
    sosem_data = [rpa_sosem, rpa_fl_sosem]
    sigma_getters = [get_sigma_rpa_wn, get_sigma_rpa_fl_wn]
    tail_getters = [get_rpa_analytic_tail, get_rpa_fl_analytic_tail]
    colors = [
        [cdict["orange"], cdict["magenta"], cdict["red"]],
        [cdict["blue"], cdict["cyan"], cdict["teal"]],
    ]
    for rs in rslist
        # Get UEG parameters
        # NOTE: We need to be careful to generate Σ_RPA for the *bare* theory, i.e.,
        #       to use an ElectronGas.Parameter object where Λs (mass2) is zero!
        para = Parameter.rydbergUnit(1 / beta, rs, 3)
        @assert para.Λs == para.Λa == 0.0
        EF = para.EF
        # Plot fits for RPA(+FL)
        for rpa_and_rpa_fl_data in zip(
            modes,
            int_types,
            modestrings,
            sosem_data,
            sigma_getters,
            tail_getters,
            colors,
        )
            mode, int_type, modestring, sosem, sigma_getter, tail_getter, modecolors =
                rpa_and_rpa_fl_data

            # TODO: Fix f(∞) ↦ G⁺(∞) for RPA+FL!
            mode == :rpa_fl && continue

            println("\nPlotting $modestring fits for rs = $(rs)...\n")
            @assert haskey(sosem, rs) "No $modestring moments available for rs = $(rs)!"

            # Get the RPA(+FL) self-energy and corresponding DLR grid from the ElectronGas package
            dlr, kval, omega_ns, sigma_stat, sigma_dyn, zfactor =
                sigma_getter(para; ktarget=ktarget, int_type=int_type)

            ds_dw = (imag(sigma_dyn)[2] - imag(sigma_dyn)[1]) / (2π / para.β)
            zmanual = 1 / (1 - ds_dw)
            println("1 - 1/z(k=$ktarget) = $ds_dw")
            println("zmanual: $zmanual")
            println("zfactor: $zfactor")

            # Dimensionless UV cutoff used for the DLR grid
            Euv = dlr.Euv / EF

            # wₙ = ωₙ / EF
            wns = omega_ns / EF
            println(maximum(wns))

            # The static part should not contribute to ImΣ
            println(imag(sigma_stat))
            @assert isapprox(imag(sigma_stat), 0.0, atol=1e-5)
            im_sigma_over_EF = imag(sigma_dyn) / EF

            # Get fₛ(∞) for use in the RPA(+FL) C coefficient
            fs_infty = get_fs_infty(para, int_type)

            # Parameters A(rₛ), B(rₛ), and Ωₜ(rₛ) for the RPA self-energy
            A = (1 / zfactor - 1)            # A = (z⁻¹ - 1)
            B = sosem[rs] / EF^2             # B = C⁽¹⁾ / EF²
            C = C_coefficient(rs, fs_infty)  # C = -(16√2 / 3π) (αrₛ)²
            Omega_t = sqrt(B / A)            # Ωₜ = √(B / A)
            # Measured B coefficient
            iwmed = searchsortedfirst(wns, 15)
            B_meas = @. -wns * im_sigma_over_EF
            B_meas_med = -wns[iwmed] * (im_sigma_over_EF[iwmed])
            B_rel_err = 100 * abs((B - B_meas[end]) / B)
            B_rel_err_med = 100 * abs((B - B_meas_med) / B)
            println("B = $B")
            println("B_meas = $(B_meas[end])")
            println(
                "Relative error between the exact/measured $modestring SOSEM (wmax = $(round(wns[end])); sigdigits=5): $B_rel_err",
            )
            println(
                "Relative error between the exact/measured $modestring SOSEM (wmax = $(round(wns[iwmed]; sigdigits=5))): $B_rel_err_med",
            )
            # Measured C coefficient
            _wns = wns[2:(end - 1)]
            C_meas = @. (
                (sqrt(_wns * wns[end]) / (sqrt(wns[end]) - sqrt(_wns))) *
                (-_wns * im_sigma_over_EF[2:(end - 1)] - B_meas[end])
            )
            C_rel_err = @. 100 * abs((C - C_meas[end]) / C)
            println("C = $C")
            println("C_meas = $(C_meas[end])")
            println(
                "Relative error between the exact/measured $modestring C coefficient (wmax = $(round(_wns[end])); sigdigits=5): $C_rel_err",
            )

            # # Approximate f₂(w) using leading-order estimates for A(rₛ) and B(rₛ)
            # function f2_leading_order(w)
            #     if int_type == :rpa
            #         return 0.48w * rs^2 / (w^2 + 3.03rs)
            #     elseif int_type == :ko
            #         SOSEM.@todo
            #         # return e1 * w * rs^2 / (w^2 + e2 * rs)
            #     else
            #         SOSEM.@todo
            #     end
            # end

            # Simple interpolation: f₂(w) = -Im{B / (iw + Ωₜ)} = B w / (w² + Ωₜ²), 
            # where w = ω / EF and Ωₜ = sqrt(B / A)
            function f2(w)
                return B * w / (w^2 + Omega_t^2)
            end

            # For f₃(w), fit parameters a1 and a3 are derived from the
            # low/high-frequency tails, while a2 is a free parameter
            function f3_params(a2, sgn_a3)
                @assert sgn_a3 ∈ [-1, +1] "sgn(a3) must be either +1 or -1!"
                a1 = B
                a3 = sgn_a3 * sqrt(Complex((a2 / Omega_t)^2 - a2))
                return a1, a2, a3
            end
            # Next-order continued fraction: f₃(w) = -Im{a₁ / (iw + a₂ / (iw + a₃))}, w = ω / EF
            function f3(w, p; sgn_a3=1)
                @assert sgn_a3 ∈ [-1, +1] "sgn(a3) must be either +1 or -1!"
                iw = im * w
                # Get the fit parameters
                a1, a2, a3 = f3_params(p[1], sgn_a3)
                # Return the imaginary part of the continued fraction fit
                continued_fraction_fit = @. a1 / (iw - a2 / (iw - a3))
                return -imag(continued_fraction_fit)
            end

            # wn_fit_cutoff = 15
            wn_fit_cutoff = 1000
            fit_window = wns .≤ wn_fit_cutoff

            # Helper functions for f^{±}_3
            f3p = (w, p) -> f3(w, p; sgn_a3=+1)
            f3m = (w, p) -> f3(w, p; sgn_a3=-1)
            # Perform a least-square fit of f₃(w) to the imaginary part of the RPA self-energy
            model_f3p = curve_fit(
                f3p,
                wns[fit_window],
                -im_sigma_over_EF[fit_window],
                [2.0 * Omega_t^2],
            )
            model_f3m = curve_fit(
                f3m,
                wns[fit_window],
                -im_sigma_over_EF[fit_window],
                [2.0 * Omega_t^2],
            )
            fit_f3p(w) = f3p(w, model_f3p.param)
            fit_f3m(w) = f3m(w, model_f3m.param)
            r2_f3p = rsquared(
                wns[fit_window],
                -im_sigma_over_EF[fit_window],
                fit_f3p.(wns[fit_window]),
            )
            r2_f3m = rsquared(
                wns[fit_window],
                -im_sigma_over_EF[fit_window],
                fit_f3m.(wns[fit_window]),
            )

            # For g₃(w), fit parameters a3 and a4 are derived from the
            # low/high-frequency tails, while a1 and a2 are free parameters
            function g3_params(a1p, a2p, sgn_a4p)
                @assert sgn_a4p ∈ [-1, +1] "sgn(a4p) must be either +1 or -1!"
                a3p = B - a1p
                a4p = sgn_a4p * sqrt(Complex(a3p / (A - a1p / a2p^2)))
                # a4p = sgn_a4p * real(sqrt(Complex(a3p / (A - a1p / a2p^2))))
                return a1p, a2p, a3p, a4p
            end
            # Next-order multipole: g₃(w) = -Im{a₁ / (iw + a₂) + a₃ / (iw + a₄)}, w = ω / EF
            function g3(w, p; sgn_a4p=1)
                @assert sgn_a4p ∈ [-1, +1] "sgn(a4) must be either +1 or -1!"
                iw = im * w
                # Get the fit parameters
                a1p, a2p, a3p, a4p = g3_params(p[1], p[2], sgn_a4p)
                # Return the imaginary part of the multipole fit
                multipole_fit = @. a1p / (iw - a2p) + a3p / (iw - a4p)
                return -imag(multipole_fit)
            end
            # Helper functions for g^{±}_3
            g3p = (w, p) -> g3(w, p; sgn_a4p=+1)
            g3m = (w, p) -> g3(w, p; sgn_a4p=-1)
            # Perform a least-square fit of g₃(w) to the imaginary part of the RPA self-energy
            model_g3p =
                curve_fit(g3p, wns[fit_window], -im_sigma_over_EF[fit_window], [-2.0, -1.0])
            model_g3m =
                curve_fit(g3m, wns[fit_window], -im_sigma_over_EF[fit_window], [-2.0, -1.0])
            fit_g3p(w) = g3p(w, model_g3p.param)
            fit_g3m(w) = g3m(w, model_g3m.param)
            r2_g3p = rsquared(
                wns[fit_window],
                -im_sigma_over_EF[fit_window],
                fit_g3p.(wns[fit_window]),
            )
            r2_g3m = rsquared(
                wns[fit_window],
                -im_sigma_over_EF[fit_window],
                fit_g3m.(wns[fit_window]),
            )

            println(
                "\nf3p parameters: ",
                round.(f3_params(model_f3p.param..., 1); sigdigits=5),
            )
            println(
                "f3m parameters: ",
                round.(f3_params(model_f3m.param..., -1); sigdigits=5),
            )
            println(
                "\ng3p parameters: ",
                round.(g3_params(model_g3p.param..., 1); sigdigits=5),
            )
            println(
                "g3m parameters: ",
                round.(g3_params(model_g3m.param..., -1); sigdigits=5),
            )

            println("\nfit_f3p: ", round.(fit_f3p.(collect(1:5)); sigdigits=5))
            println("fit_f3m: ", round.(fit_f3m.(collect(1:5)); sigdigits=5))
            println("\nfit_g3p: ", round.(fit_g3p.(collect(1:5)); sigdigits=5))
            println("fit_g3m: ", round.(fit_g3m.(collect(1:5)); sigdigits=5))

            println("\nr² for f3p: ", round(r2_f3p; sigdigits=5))
            println("r² for f3m: ", round(r2_f3m; sigdigits=5))
            println("\nr² for g3p: ", round(r2_g3p; sigdigits=5))
            println("r² for g3m: ", round(r2_g3m; sigdigits=5))

            println(
                "\nf2 - f3: ",
                round.(f2.(collect(1:5)) - fit_f3p(collect(1:5)), sigdigits=5),
            )

            # wmax_plot = 30
            wmax_plot = 15
            wns_fine = collect(LinRange(0, wmax_plot, 1000))
            # println("\n-ImΣ / EF:\n$(-im_sigma_over_EF)\n")

            # Plot a comparison of minus the imaginary part of the RPA self-energy
            # to the simple, continued-fraction, and multipole fits
            fig = figure(; figsize=(6, 4))
            plot(wns_fine, A .* wns_fine; linestyle="--", color="gray", zorder=500)
            plot(wns_fine, B ./ wns_fine; linestyle="--", color="gray", zorder=500)
            wns_spl, minus_im_sigma_over_EF_spl =
                spline(wns, -im_sigma_over_EF; xmax=wns[searchsortedfirst(wns, wmax_plot)])
            plot(
                wns_spl,
                minus_im_sigma_over_EF_spl;
                color=color[1],
                label="\$-\\mathrm{Im}\\Sigma_{\\mathrm{$modestring}}(k = 0, i\\omega_n)\$",
                zorder=1000,
            )
            plot(
                wns_fine,
                fit_f3p.(wns_fine);
                color=color[1],
                linestyle="--",
                label="\$b^\\star_2 = $(round(model_f3p.param[1] / Omega_t^2; sigdigits=3))\\Omega^2_t\$",
                zorder=1001,
            )
            println("Omega_t = $Omega_t")
            a2s = [2, 5, 10, 50, Inf, -50, -10, -5, -2]
            for (i, a2_plot) in enumerate(a2s)
                if isinf(a2_plot)
                    plot(
                        wns_fine,
                        f2.(wns_fine);
                        color=reverse(collect(values(cdict2)))[i + 1],
                        label="\$b_2 = \\mp \\infty\$ \$(f_2(i\\omega_n))\$",
                        zorder=length(a2s) - i,
                    )
                else
                    plot(
                        wns_fine,
                        [f3p.(w, [a2_plot * Omega_t^2]) for w in wns_fine];
                        color=reverse(collect(values(cdict2)))[i + 1],
                        label="\$b_2 = $(Int(a2_plot))\\Omega^2_t\$",
                        zorder=abs(a2_plot) == 50 ? -1 : length(a2s) - i,
                    )
                end
            end
            xlim(0, wmax_plot)
            ylim(-0.02, 0.42)
            if rs == 0.1
                ylim(0.0, 0.008)
            elseif rs == 1.0
                ylim(-0.02, 0.42)
            elseif rs == 5.0
                ylim(-0.1, 6.5)
            elseif rs == 10.0
                ylim(-1.0, 25.0)
            end
            xlabel("\$\\omega_n\$")
            ylabel("\$f_3(b_2, i\\omega_n)\$")
            legend(; loc="upper right", ncol=1, fontsize=10)
            plt.tight_layout()
            savefig(
                "results/self_energy_fits/fits/$(lowercase(modestring))/$(mode)_fit_comparisons_rs=$(rs).pdf",
            )

            # Plot a comparison of minus the imaginary part of the RPA self-energy
            # to the simple, continued-fraction, and multipole fits
            fig = figure(; figsize=(6, 4))
            plot(wns_fine, A .* wns_fine; linestyle="--", color="0.6")
            plot(wns_fine, B ./ wns_fine; linestyle="--", color="0.6")
            axvline(Omega_t; color="0.6", linestyle="-", label="\$\\Omega_t\$", zorder=-1)
            plot(
                wns_fine,
                B ./ wns_fine + tail_getter(wns_fine, para, fs_infty);
                color="0.4",
                linestyle="--",
                label="\$\\frac{B}{\\omega_n} + \\frac{C}{\\omega^{3/2}_n}\$",
                zorder=1000,
            )
            plot(
                wns_spl,
                minus_im_sigma_over_EF_spl;
                color=color[1],
                label=modestring,
                zorder=1000,
            )
            wns_spl, minus_im_sigma_over_EF_spl =
                spline(wns, -im_sigma_over_EF; xmax=wns[searchsortedfirst(wns, wmax_plot)])

            plot(
                wns_fine,
                f2.(wns_fine);
                color=color[5],
                linestyle="-",
                label="\$f_2(i\\omega_n)\$",
            )
            plot(
                wns_fine,
                fit_f3p.(wns_fine);
                color=color[2],
                label="\$f^\\star_3(i\\omega_n)\$",
            )
            plot(
                wns_fine,
                fit_g3p.(wns_fine);
                color=color[3],
                label="\$g^\\star_3(i\\omega_n)\$",
            )
            # plot(wns_fine, fit_f3m.(wns_fine); color=color[4], label="\$f^-_3(i\\omega_n)\$")
            # plot(wns_fine, fit_g3m.(wns_fine); color=color[6], label="\$g^-_3(i\\omega_n)\$")
            xlim(0, wmax_plot)
            if rs == 0.1
                ylim(0.0, 0.006)
            elseif rs == 1.0
                ylim(-0.01, 0.26)
                text(
                    3.875,
                    0.2,
                    "\$r_s = $(rs)\$";
                    # "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
                )
            elseif rs == 5.0
                ylim(-0.1, 3.5)
                text(
                    6.25,
                    2.5,
                    "\$r_s = $(rs)\$";
                    # "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
                )
            elseif rs == 10.0
                ylim(-0.5, 12.0)
                text(
                    3.75,
                    10.0,
                    "\$r_s = $(rs)\$";
                    # "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
                )
            end
            # ylim(0, 2.2 * max(maximum(-im_sigma_over_EF), maximum(f2.(wns_fine))))
            xlabel("\$\\omega_n\$")
            ylabel("\$-\\mathrm{Im}\\,\\Sigma(k = 0, i\\omega_n)\$")
            legend(; loc="upper right")
            plt.tight_layout()
            savefig(
                "results/self_energy_fits/fits/$(lowercase(modestring))/im_sigma_$(mode)_fits_rs=$(rs).pdf",
            )

            # Plot the derivative of -ImΣ
            fig = figure(; figsize=(6, 4))
            # Get ds/dw from central difference method
            __wns, ds_dw = forward_diff(-im_sigma_over_EF, wns)
            __wns_fine, df2_dw = forward_diff(f2.(wns_fine), wns_fine)
            __wns_fine, df3_dw = forward_diff(fit_f3p.(wns_fine), wns_fine)
            __wns_fine, dg3_dw = forward_diff(fit_g3p.(wns_fine), wns_fine)
            iwlow = 1
            A_rel_err = 100 * abs((ds_dw[iwlow] - A) / A)
            println(
                "\nRelative error between the exact/measured low-frequency moment (w = $(wns[iwlow])): $A_rel_err",
            )
            plot(
                __wns,
                A * one.(__wns);
                color=color[1],
                linestyle="--",
                label="\$A(r_s)\$",
                zorder=1000,
            )
            plot(__wns, ds_dw; color=color[1], label=modestring, zorder=1000)
            plot(
                __wns_fine,
                df2_dw;
                color=color[5],
                linestyle="-",
                label="\${f^\\star}^\\prime_2(i\\omega_n)\$",
            )
            plot(
                __wns_fine,
                df3_dw;
                color=color[2],
                label="\${f^\\star}^\\prime_3(i\\omega_n)\$",
            )
            plot(
                __wns_fine,
                dg3_dw;
                color=color[3],
                label="\${g^\\star}^\\prime_3(i\\omega_n)\$",
            )
            xlim(0, 1)
            # ylim(0.025, 0.115)
            if rs == 1.0
                ylim(0.025, 0.115)
                text(
                    0.7,
                    0.095,
                    "\$r_s = $(rs)\$";
                    # "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
                )
            elseif rs == 5.0
                ylim(0.18, nothing)
                text(
                    0.7,
                    0.25,
                    "\$r_s = $(rs)\$";
                    # "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
                )
            elseif rs == 10.0
                ylim(0.5, nothing)
                text(
                    0.7,
                    0.65,
                    "\$r_s = $(rs)\$";
                    # "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
                )
            end
            # ylim(
            #     0.9 * minimum([ds_dw; df2_dw; df3_dw; dg3_dw]),
            #     1.1 * maximum([ds_dw; df2_dw; df3_dw; dg3_dw]),
            # )
            xlabel("\$\\omega_n\$")
            ylabel("\$A(r_s, i\\omega_n)\$")
            legend(; loc="best")
            plt.tight_layout()
            savefig(
                "results/self_energy_fits/fits/$(lowercase(modestring))/A_coefficient_$(mode)_rs=$(rs).pdf",
            )

            # Plot ωₙ|ImΣ|
            fig = figure(; figsize=(6, 4))
            # wmax_big = 15
            wmax_big = maximum(wns)
            wns_big = collect(LinRange(0, wmax_big, 1000))
            wns_big_spl, B_meas_big_spl =
                spline(wns, B_meas; xmax=wns[searchsortedfirst(wns, wmax_big)])
            # wns_big_spl, minus_im_sigma_over_EF_big_spl =
            #     spline(wns, -im_sigma_over_EF; xmax=wns[searchsortedfirst(wns, wmax_big)])
            if wmax_big > Euv
                axvline(
                    Euv;
                    color="gray",
                    linestyle="-",
                    label="\$\\Lambda_{\\text{DLR}}\$",
                    # label="\$\\Lambda_{\\text{DLR}} = $(Int(Euv))\$",
                    zorder=-1,
                )
            end
            plot(
                wns_big,
                B * one.(wns_big);
                color=color[1],
                linestyle="--",
                label="\$B(r_s)\$",
                zorder=100,
            )
            plot(
                wns,
                B_meas;
                # wns_big_spl,
                # B_meas_big_spl;
                color=color[1],
                label=modestring,
                zorder=1000,
            )
            plot(
                wns_big,
                wns_big .* f2.(wns_big);
                color=color[5],
                label="\$\\omega_n f_2(i\\omega_n)\$",
                # zorder=6,
            )
            plot(
                wns_big,
                wns_big .* fit_f3p.(wns_big);
                color=color[2],
                label="\$\\omega_n f^\\star_3(i\\omega_n)\$",
                # zorder=5,
            )
            plot(
                wns_big,
                wns_big .* fit_g3p.(wns_big);
                color=color[3],
                label="\$\\omega_n g^\\star_3(i\\omega_n)\$",
                # zorder=4,
            )
            # xlim(0, wmax_big)
            xlim(0, maximum(wns_big))
            # ylim(-0.04, nothing)
            if rs == 1.0
                ylim(0.39, 0.51)
                # ylim(-0.04, nothing)
                text(
                    250,
                    0.425,
                    "\$r_s = $(rs)\$";
                    # "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
                )
            end
            # if rs == 1.0
            #     # ylim(0.38, nothing)
            #     ylim(-0.04, nothing)
            #     text(
            #         2.5,
            #         0.05,
            #         "\$r_s = $(rs)\$";
            #         # "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
            #     )
            # elseif rs == 5.0
            #     # ylim(0.38, nothing)
            #     # ylim(-0.04, nothing)
            #     text(
            #         3.75,
            #         1.0,
            #         "\$r_s = $(rs)\$";
            #         # "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
            #     )
            # elseif rs == 10.0
            #     ylim(-6, nothing)
            #     text(
            #         3.75,
            #         0.0,
            #         "\$r_s = $(rs)\$";
            #         # "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
            #     )
            # end
            # ylim(0, 1.5 * maximum([ds_dw; df2_dw; df3_dw; dg3_dw]))
            xlabel("\$\\omega_n\$")
            ylabel("\$B(r_s, i\\omega_n)\$")
            legend(; loc="best")
            plt.tight_layout()
            savefig(
                "results/self_energy_fits/fits/$(lowercase(modestring))/B_coefficient_$(mode)_large_rs=$(rs).pdf",
                # "results/self_energy_fits/fits/$(lowercase(modestring))/B_coefficient_$(mode)_rs=$(rs).pdf",
            )

            # Compare C_meas ≈ (sqrt(ωₙ ωₘₐₓ) / (sqrt(ωₘₐₓ) - sqrt(ωₙ))) * (ωₙ|ImΣ(iωₙ)| - B_meas)
            # to analytic coefficient C via VZN
            fig = figure(; figsize=(6, 4))
            wmax = 15
            # wmax = maximum(_wns)
            wns_big = collect(LinRange(0, wmax, 1000))
            wns_spl, C_meas_spl =
                spline(_wns, C_meas; xmax=_wns[searchsortedfirst(_wns, wmax)])
            plot(
                wns_big,
                C * one.(wns_big);
                color=color[1],
                linestyle="--",
                label="\$-\\frac{16}{3\\pi}(\\alpha r_s)^2\$",
                zorder=100,
            )
            # plot(_wns, C_meas; color=color[1], label=modestring, zorder=1000)
            plot(wns_spl, C_meas_spl; color=color[1], label=modestring, zorder=1000)
            if wmax > Euv
                axvline(
                    Euv;
                    color=cdict["red"],
                    linestyle="-",
                    label="\$\\Lambda_{\\text{DLR}} = $(Int(Euv))\$",
                    zorder=-1,
                )
            end
            xlim(0, wmax)
            ylim(nothing, 0.0)
            if rs == 1.0
                text(
                    2.5,
                    -0.1,
                    "\$r_s = $(rs)\$";
                    # "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
                )
            elseif rs == 5.0
                text(
                    2.5,
                    -2.5,
                    "\$r_s = $(rs)\$";
                    # "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
                )
            elseif rs == 10.0
                text(
                    2.5,
                    -10.0,
                    "\$r_s = $(rs)\$";
                    # "\$r_s = $(rs),\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\$";
                )
            end
            # ylim(-0.5, nothing)
            xlabel("\$\\omega_n\$")
            ylabel("\$C(r_s, i\\omega_n)\$")
            legend(; loc="best")
            plt.tight_layout()
            # savefig("results/self_energy_fits/fits/$(lowercase(modestring))/C_coefficient_$(mode)_large_rs=$(rs).pdf")
            savefig(
                "results/self_energy_fits/fits/$(lowercase(modestring))/C_coefficient_$(mode)_rs=$(rs).pdf",
            )

            println("Done!")
            plt.close("all")
        end
    end
    return
end

main()
