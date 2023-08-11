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

const zfactor_benchmarks = Dict(
    1.0 => 0.8601,
    2.0 => 0.7642,
    3.0 => 0.6927,
    4.0 => 0.6367,
    5.0 => 0.5913,
    6.0 => 0.5535,
)

"""@ beta = 1000, small cutoffs"""
const second_order_moments = Dict(
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

"""@ beta = 1000, converged cutoffs"""
const second_order_moments_converged = Dict(
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

function get_rpa_analytic_tail(wns, para::Parameter.Para)
    rs = round(para.rs; sigdigits=13)
    # VZN prefactor C
    # prefactor = -16 * sqrt(2) * (α * rs)^2 / 3π
    # -Im(1 / i^(3/2)) = 1/sqrt(2)
    prefactor = 16 * sqrt(2) * (α * rs)^2 / 3π
    # prefactor = 16 * (α * rs)^2 / 3π
    return @. prefactor / wns^(3 / 2)
end

function get_sigma_rpa_wn(para::Parameter.Para; ktarget=0.0, atol=1e-3)
    # Make sure we are using parameters for the bare UEG theory
    @assert para.Λs == para.Λa == 0.0

    # # Converged params
    # Euv, rtol = 10000 * para.EF, 1e-11
    # maxK, minK = 1000 * para.kF, 1e-8 * para.kF
    # Nk, order = 25, 20

    # Small params
    Euv, rtol = 1000 * para.EF, 1e-11
    maxK, minK = 30 * para.kF, 1e-8 * para.kF
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
    zfactor = SelfEnergy.zfactor(para, sigma_wn_dynamic; ngrid=[-1, 0])[1]
    # zfactor_0p1 = SelfEnergy.zfactor(para, sigma_wn_dynamic; ngrid=[0, 1])[1]
    # zfactor = SelfEnergy.zfactor(para, sigma_wn_dynamic)[1]

    # println("Z_RPA (benchmark): ", zfactor_benchmarks[round(para.rs; sigdigits=13)])
    # println("Z_RPA (ngrid = [-1, 0]): ", zfactor_m10)
    # println("Z_RPA (ngrid = [0, 1]): ", zfactor_0p1)
    # zfactor = zfactor_m10
    # zfactor = zfactor_0p1

    # Check zfactor against benchmarks at β = 1000. 
    # It should agree within a few percent (up to finite-temperature effects).
    # (see: https://numericaleft.github.io/ElectronGas.jl/dev/manual/quasiparticle/)
    rs = round(para.rs; sigdigits=13)
    if rs in keys(zfactor_benchmarks)
        println("Checking zfactor against zero-temperature benchmark...")
        zfactor_zero_temp = zfactor_benchmarks[rs]
        percent_error = 100 * abs(zfactor - zfactor_zero_temp) / zfactor_zero_temp
        println("Percent error vs zero-temperature benchmark of Z_kF: $percent_error")
        @assert percent_error ≤ 5
    else
        println("No zero-temperature benchmark available for rs = $(rs)!")
    end

    return kval, wns, sigma_wn_static_kval, sigma_wn_dynamic_kval, zfactor
end

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    # We fit Σ_RPA(k = 0, iw)
    ktarget = 0.0

    # Physical parameters
    rs = 1.0
    beta = 1000.0
    @assert haskey(second_order_moments, rs) "No RPA moments available for rs = $(rs)!"
    # @assert haskey(second_order_moments_converged, rs) "No RPA moments available for rs = $(rs)!"

    # Get UEG parameters
    # NOTE: We need to be careful to generate Σ_RPA for the *bare* theory, i.e.,
    #       to use an ElectronGas.Parameter object where Λs (mass2) is zero!
    para = Parameter.rydbergUnit(1 / beta, rs, 3)
    @assert para.Λs == para.Λa == 0.0
    EF = para.EF

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
    # rcParams["font.family"] = "Times New Roman"

    # Get the RPA self-energy and corresponding DLR grid from the ElectronGas package
    kval, omega_ns, sigma_stat, sigma_dyn, zfactor = get_sigma_rpa_wn(para; ktarget=ktarget)

    # wₙ = ωₙ / EF
    wns = omega_ns / EF

    # The static part should not contribute to ImΣ
    println(imag(sigma_stat))
    @assert isapprox(imag(sigma_stat), 0.0, atol=1e-5)
    im_sigma_over_EF = imag(sigma_dyn) / EF

    # Parameters A(rₛ), B(rₛ), and Ωₜ(rₛ) for the RPA self-energy
    A = (1 / zfactor - 1)                # A = (z⁻¹ - 1)
    B = second_order_moments[rs] / EF^2  # B = C⁽¹⁾ / EF²
    # B = second_order_moments_converged[rs] / EF^2  # B = C⁽¹⁾ / EF²
    Omega_t = sqrt(B / A)                # Ωₜ = √(B / A)

    iwmed = searchsortedfirst(wns, 15)
    B_meas = -wns[end] * (im_sigma_over_EF[end])
    B_meas_med = -wns[iwmed] * (im_sigma_over_EF[iwmed])
    B_rel_err = 100 * abs((B - B_meas) / B)
    B_rel_err_med = 100 * abs((B - B_meas_med) / B)
    println("B = $B")
    println("B_meas = $B_meas")
    println("Relative error between the exact/measured SOSEM (w = $(wns[end])): $B_rel_err")
    println(
        "Relative error between the exact/measured SOSEM (w = $(wns[iwmed])): $B_rel_err_med",
    )

    # Simple interpolation: f₂(w) = -Im{B / (iw + Ωₜ)} = B w / (w² + Ωₜ²), 
    # where w = ω / EF and Ωₜ = sqrt(B / A)
    function f2(w)
        return B * w / (w^2 + Omega_t^2)
    end

    # Approximate f₂(w) using leading-order estimates for A(rₛ) and B(rₛ)
    function f2_leading_order(w)
        return 0.48w * rs^2 / (w^2 + 3.03rs)
    end

    # # For h₃(w), fit parameters a3 and a4 are derived from the
    # # low/high-frequency tails, while a1, a2, a5, and a6 are free parameters
    # function get_h3_params(a1, a2, a5, a6, sgn_a4)
    #     @assert sgn_a4 ∈ [-1, +1] "sgn(a4) must be either +1 or -1!"
    #     a3 = B - a1 - a5
    #     a4 = sgn_a4 * sqrt(Complex(a3 / (A - a1 / a2^2 - a5 / a6^2)))
    #     return a1, a2, a3, a4, a5, a6
    # end

    # For f₃(w), fit parameters a1 and a3 are derived from the
    # low/high-frequency tails, while a2 is a free parameter
    function get_f3_params(a2, sgn_a3)
        @assert sgn_a3 ∈ [-1, +1] "sgn(a3) must be either +1 or -1!"
        a1 = B
        a3 = sgn_a3 * sqrt(Complex(a2 + (a2 / Omega_t)^2))
        # a3 = sgn_a3 * real(sqrt(Complex(a2 + (a2 / Omega_t)^2)))
        return a1, a2, a3
    end
    # Next-order continued fraction: f₃(w) = -Im{a₁ / (iw + a₂ / (iw + a₃))}, w = ω / EF
    function f3(w, p; sgn_a3=1)
        @assert sgn_a3 ∈ [-1, +1] "sgn(a3) must be either +1 or -1!"
        iw = im * w
        # Get the fit parameters
        a1, a2, a3 = get_f3_params(p[1], sgn_a3)
        # Return the imaginary part of the continued fraction fit
        continued_fraction_fit = @. a1 / (iw + a2 / (iw + a3))
        return -imag(continued_fraction_fit)
    end
    # Helper functions for f^{±}_3
    f3p = (w, p) -> f3(w, p; sgn_a3=+1)
    f3m = (w, p) -> f3(w, p; sgn_a3=-1)
    # Perform a least-square fit of f₃(w) to the imaginary part of the RPA self-energy
    model_f3p = curve_fit(f3p, wns, -im_sigma_over_EF, [-2.0])
    model_f3m = curve_fit(f3m, wns, -im_sigma_over_EF, [-2.0])
    fit_f3p(w) = f3p(w, model_f3p.param)
    fit_f3m(w) = f3m(w, model_f3m.param)
    r2_f3p = rsquared(wns, -im_sigma_over_EF, fit_f3p.(wns))
    r2_f3m = rsquared(wns, -im_sigma_over_EF, fit_f3m.(wns))

    # For g₃(w), fit parameters a3 and a4 are derived from the
    # low/high-frequency tails, while a1 and a2 are free parameters
    function get_g3_params(a1p, a2p, sgn_a4p)
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
        a1p, a2p, a3p, a4p = get_g3_params(p[1], p[2], sgn_a4p)
        # Return the imaginary part of the multipole fit
        multipole_fit = @. a1p / (iw + a2p) + a3p / (iw + a4p)
        return -imag(multipole_fit)
    end
    # Helper functions for g^{±}_3
    g3p = (w, p) -> g3(w, p; sgn_a4p=+1)
    g3m = (w, p) -> g3(w, p; sgn_a4p=-1)
    # Perform a least-square fit of g₃(w) to the imaginary part of the RPA self-energy
    model_g3p = curve_fit(g3p, wns, -im_sigma_over_EF, [-2.0, 1.0])
    model_g3m = curve_fit(g3m, wns, -im_sigma_over_EF, [-2.0, 1.0])
    fit_g3p(w) = g3p(w, model_g3p.param)
    fit_g3m(w) = g3m(w, model_g3m.param)
    r2_g3p = rsquared(wns, -im_sigma_over_EF, fit_g3p.(wns))
    r2_g3m = rsquared(wns, -im_sigma_over_EF, fit_g3m.(wns))

    # # Next-next-order multipole: h₃(w) = -Im{a₁ / (iw + a₂) + a₃ / (iw + a₄) + a₅ / (iw + a₆)}, w = ω / EF
    # function h3(w, p; sgn_a4=1)
    #     # Get the fit parameters
    #     iw = im * w
    #     a1, a2, a3, a4, a5, a6 = get_h3_params(p[1], p[2], p[3], p[4], sgn_a4)
    #     # Return the imaginary part of the multipole fit
    #     multipole_fit = @. a1 / (iw + a2) + a3 / (iw + a4) + a5 / (iw + a6)
    #     return -imag(multipole_fit)
    # end
    # h3p = (w, p) -> h3(w, p; sgn_a4=+1)
    # h3m = (w, p) -> h3(w, p; sgn_a4=-1)
    # # Perform a least-square fit of h₃(w) to the imaginary part of the RPA self-energy
    # model_h3p = curve_fit(h3p, wns, -im_sigma_over_EF, [1.0, 1.0, 1.0, 1.0])
    # model_h3m = curve_fit(h3m, wns, -im_sigma_over_EF, [1.0, 1.0, 1.0, 1.0])
    # fit_h3p(w) = h3p(w, model_h3p.param)
    # fit_h3m(w) = h3m(w, model_h3m.param)

    println("\nf3p parameters: ", round.(model_f3p.param; sigdigits=5))
    println("f3m parameters: ", round.(model_f3m.param; sigdigits=5))
    println("\ng3p parameters: ", round.(model_g3p.param; sigdigits=5))
    println("g3m parameters: ", round.(model_g3m.param; sigdigits=5))
    # println("\nh3p parameters: ", round.(model_h3p.param; sigdigits=5))
    # println("h3m parameters: ", round.(model_h3m.param; sigdigits=5))

    println("\nfit_f3p: ", round.(fit_f3p.(collect(1:5)); sigdigits=5))
    println("fit_f3m: ", round.(fit_f3m.(collect(1:5)); sigdigits=5))
    println("\nfit_g3p: ", round.(fit_g3p.(collect(1:5)); sigdigits=5))
    println("fit_g3m: ", round.(fit_g3m.(collect(1:5)); sigdigits=5))
    # println("\nfit_h3p: ", round.(fit_h3p.(collect(1:5)); sigdigits=5))
    # println("fit_h3m: ", round.(fit_h3m.(collect(1:5)); sigdigits=5))

    println("\nr² for f3p: ", round(r2_f3p; sigdigits=5))
    println("r² for f3m: ", round(r2_f3m; sigdigits=5))
    println("\nr² for g3p: ", round(r2_g3p; sigdigits=5))
    println("r² for g3m: ", round(r2_g3m; sigdigits=5))

    println("\nf2 - f3: ", round.(f2.(collect(1:5)) - fit_f3p(collect(1:5)), sigdigits=5))

    wmax_plot = 15
    wns_fine = collect(LinRange(0, wmax_plot, 1000))
    # println("\n-ImΣ / EF:\n$(-im_sigma_over_EF)\n")

    # Plot a comparison of minus the imaginary part of the RPA self-energy
    # to the simple, continued-fraction, and multipole fits
    fig = figure(; figsize=(6, 4))
    plot(wns_fine, A .* wns_fine; linestyle="--", color="gray")
    plot(wns_fine, B ./ wns_fine; linestyle="--", color="gray")
    wns_spl, minus_im_sigma_over_EF_spl =
        spline(wns, -im_sigma_over_EF; xmax=wns[searchsortedfirst(wns, wmax_plot)])
    plot(
        wns_spl,
        minus_im_sigma_over_EF_spl;
        color=color[1],
        label="\$-\\mathrm{Im}\\Sigma_{\\mathrm{RPA}}(k = 0, i\\omega_n)\$",
        zorder=1,
    )
    plot(
        wns_fine,
        fit_f3p.(wns_fine);
        color=color[1],
        linestyle="--",
        label="\$a^\\star_2 = $(round(model_f3p.param[1] / Omega_t^2; sigdigits=3))\\Omega^2_t\$",
        zorder=101,
    )
    println("Omega_t = $Omega_t")
    a2s = [-2, -5, -10, -50, Inf, 50, 10, 5, 2]
    for (i, a2_plot) in enumerate(a2s)
        if isinf(a2_plot)
            plot(
                wns_fine,
                f2.(wns_fine);
                color=reverse(collect(values(cdict2)))[i + 1],
                label="\$a_2 = \\mp \\infty\$ \$(f_2(i\\omega_n))\$",
                zorder=100,
            )
        else
            plot(
                wns_fine,
                [f3p.(w, [a2_plot * Omega_t^2]) for w in wns_fine];
                color=reverse(collect(values(cdict2)))[i + 1],
                label="\$a_2 = $(Int(a2_plot))\\Omega^2_t\$",
                zorder=3,
            )
        end
    end
    xlim(0, wmax_plot)
    ylim(-0.02, 0.42)
    xlabel("\$\\omega_n\$")
    ylabel("\$f_3(a_2, i\\omega_n)\$")
    legend(; loc="upper right", ncol=1, fontsize=10)
    plt.tight_layout()
    savefig("results/self_energy_fits/f2_and_f3_fit_comparisons.pdf")

    # Plot a comparison of minus the imaginary part of the RPA self-energy
    # to the simple, continued-fraction, and multipole fits
    fig = figure(; figsize=(6, 4))
    plot(wns_fine, A .* wns_fine; color=color[1], linestyle="--")
    plot(wns_fine, B ./ wns_fine; color=color[1], linestyle="--")
    wns_spl, minus_im_sigma_over_EF_spl =
        spline(wns, -im_sigma_over_EF; xmax=wns[searchsortedfirst(wns, wmax_plot)])
    plot(wns_spl, minus_im_sigma_over_EF_spl; color=color[1], label="Exact")
    plot(
        wns_fine,
        get_rpa_analytic_tail(wns_fine, para);
        color="gray",
        linestyle="-",
        label="\$\\frac{16 \\sqrt{2}}{3\\pi}\\frac{(\\alpha r_s)^2}{\\omega^{3/2}_n}\$",
    )
    # plot(
    #     wns_fine,
    #     f2_leading_order.(wns_fine);
    #     color=color[6],
    #     linestyle="--",
    #     label="\$\\frac{0.48 r^2_s \\omega_n}{\\omega^2_n + 3.03 r_s}\$",
    # )
    plot(
        wns_fine,
        f2.(wns_fine);
        color=color[5],
        linestyle="-",
        label="\$f_2(i\\omega_n)\$",
    )
    plot(wns_fine, fit_f3p.(wns_fine); color=color[2], label="\$f_3(i\\omega_n)\$")
    plot(wns_fine, fit_g3p.(wns_fine); color=color[3], label="\$g_3(i\\omega_n)\$")
    # plot(wns_fine, fit_f3m.(wns_fine); color=color[4], label="\$f^-_3(i\\omega_n)\$")
    # plot(wns_fine, fit_g3m.(wns_fine); color=color[6], label="\$g^-_3(i\\omega_n)\$")
    # plot(wns_fine, fit_h3p.(wns_fine); color=color[7], label="\$h^+_3(i\\omega_n)\$")
    # plot(wns_fine, fit_h3m.(wns_fine); color=color[8], label="\$h_3^-(i\\omega_n)\$")
    xlim(0, wmax_plot)
    ylim(0, 0.32)
    # ylim(0, 2.2 * max(maximum(-im_sigma_over_EF), maximum(f2.(wns_fine))))
    xlabel("\$\\omega_n\$")
    ylabel("\$-\\mathrm{Im}\\,\\Sigma_\\mathrm{RPA}(k = 0, i\\omega_n)\$")
    legend(; loc="upper right")
    plt.tight_layout()
    savefig("results/self_energy_fits/im_sigma_rpa_fits.pdf")
    # savefig("results/self_energy_fits/im_sigma_rpa_fits_converged.pdf")

    # Plot ωₙ|ImΣ|
    fig = figure(; figsize=(6, 4))
    # wmax_big = 15
    wmax_big = maximum(wns)
    wns_big = collect(LinRange(0, wmax_big, 1000))
    wns_big_spl, minus_im_sigma_over_EF_big_spl =
        spline(wns, -im_sigma_over_EF; xmax=wns[searchsortedfirst(wns, wmax_big)])
    plot(
        wns_big_spl,
        wns_big_spl .* minus_im_sigma_over_EF_big_spl;
        color=color[1],
        label="Exact",
        zorder=1000,
    )
    plot(
        wns_big,
        B * one.(wns_big);
        color=color[1],
        linestyle="--",
        label="\$B(r_s)\$",
        zorder=100,
    )
    # plot(
    #     wns_big,
    #     wns_big .* get_rpa_analytic_tail(wns_big, para);
    #     color="gray",
    #     linestyle="--",
    #     label="\$\\frac{16 \\sqrt{2}}{3\\pi}\\frac{(\\alpha r_s)^2}{\\sqrt{\\omega_n}}\$",
    # )
    plot(
        wns_big,
        wns_big .* f2.(wns_big);
        color=color[5],
        label="\$f_2(i\\omega_n)\$",
        # zorder=6,
    )
    plot(
        wns_big,
        wns_big .* fit_f3p.(wns_big);
        color=color[2],
        label="\$f_3(i\\omega_n)\$",
        # zorder=5,
    )
    plot(
        wns_big,
        wns_big .* fit_g3p.(wns_big);
        color=color[3],
        label="\$g_3(i\\omega_n)\$",
        # zorder=4,
    )
    # xlim(0, wmax_big)
    xlim(0, maximum(wns_big))
    ylim(0.38, nothing)
    # ylim(-0.04, nothing)
    # ylim(0, 1.5 * maximum([ds_dw; df2_dw; df3_dw; dg3_dw]))
    xlabel("\$\\omega_n\$")
    ylabel(
        "\$-\\omega_n\\partial_{\\omega_n}\\mathrm{Im}\\,\\Sigma_\\mathrm{RPA}(k = 0, i\\omega_n)\$",
    )
    legend(; loc="best")
    plt.tight_layout()
    savefig("results/self_energy_fits/wn_times_im_sigma_rpa_large.pdf")
    # savefig("results/self_energy_fits/wn_times_im_sigma_rpa.pdf")
    # savefig("results/self_energy_fits/ds_dw_converged.pdf")

    # Plot the derivative of -ImΣ
    fig = figure(; figsize=(6, 4))
    # Get ds/dw from central difference method
    _wns, ds_dw = central_diff(-im_sigma_over_EF, wns)
    __wns_fine, df2_dw = central_diff(f2.(wns_fine), wns_fine)
    __wns_fine, df3_dw = central_diff(fit_f3p.(wns_fine), wns_fine)
    __wns_fine, dg3_dw = central_diff(fit_g3p.(wns_fine), wns_fine)

    # iwlow = searchsortedfirst(wns, 0.25)
    iwlow = 1
    A_rel_err = 100 * abs((ds_dw[iwlow] - A) / A)
    println(
        "\nRelative error between the exact/measured low-frequency moment (w = $(wns[iwlow])): $A_rel_err",
    )

    plot(_wns, ds_dw; color=color[1], label="Exact")
    plot(_wns, A * one.(_wns); color=color[1], linestyle="--", label="\$A(r_s)\$")
    plot(
        __wns_fine,
        df2_dw;
        color=color[5],
        linestyle="-",
        label="\$f^\\prime_2(i\\omega_n)\$",
    )
    plot(__wns_fine, df3_dw; color=color[2], label="\$f^\\prime_3(i\\omega_n)\$")
    plot(__wns_fine, dg3_dw; color=color[3], label="\$g^\\prime_3(i\\omega_n)\$")
    xlim(0, 1)
    ylim(-0.07, 1.1 * maximum([ds_dw; df2_dw; df3_dw; dg3_dw]))
    xlabel("\$\\omega_n\$")
    ylabel(
        "\$-\\partial_{\\omega_n}\\mathrm{Im}\\,\\Sigma_\\mathrm{RPA}(k = 0, i\\omega_n)\$",
    )
    legend(; loc="best")
    plt.tight_layout()
    savefig("results/self_energy_fits/ds_dw.pdf")
    # savefig("results/self_energy_fits/ds_dw_converged.pdf")

    plt.close("all")
    return
end

main()
