using CodecZlib
using CompositeGrids
using DataStructures
using DelimitedFiles
using ElectronLiquid
using ElectronGas
using GreenFunc
using Interpolations
using JLD2
using Measurements
using Parameters
using PyCall
using PyPlot
using SOSEM
using Lehmann
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
    return error("Not yet implemented!")
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

function get_sigma_rpa_tau(para::Parameter.Para; ktarget=0.0, atol=1e-3)
    # Make sure we are using parameters for the bare UEG theory
    @assert para.Λs == para.Λa == 0.0

    # Small params
    Euv, rtol = 1000 * para.EF, 1e-11
    maxK, minK = 30 * para.kF, 1e-8 * para.kF
    Nk, order = 11, 8

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

    # Get DLR grid
    sigma_dlr_dynamic = to_dlr(sigma_tau_dynamic)
    dlr = sigma_dlr_dynamic.mesh[1].dlr
    kgrid = sigma_dlr_dynamic.mesh[2]
    taus = dlr.τ

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
    sigma_tau_dynamic = sigma_tau_dynamic[:, ikval]

    return dlr, kval, taus, sigma_tau_dynamic
end

function main()
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
    rslist = [1.0, 5.0, 10.0]
    beta = 1000.0

    # We fit Σ_RPA(k = 0, iw)
    ktarget = 0.0

    fig = figure(; figsize=(6, 4))
    for (i, rs) in enumerate(rslist)
        para = Parameter.rydbergUnit(1 / beta, rs, 3)
        @assert para.Λs == para.Λa == 0.0
        EF, kF, β = para.EF, para.kF, para.β

        # Small params
        Euv, rtol = 1000 * para.EF, 1e-11
        maxK, minK = 30 * kF, 1e-8 * kF
        Nk, order = 11, 8

        # Grids for k and n
        qgrid = CompositeGrid.LogDensedGrid(
            :gauss,
            [0.0, maxK],
            [0.0, 2.0 * kF],
            Nk,
            minK,
            order,
        )

        # Get the RPA(+FL) self-energy and corresponding DLR grid from the ElectronGas package
        dlr, kval, taus, sigma_tau_dyn = get_sigma_rpa_tau(para; ktarget=ktarget)

        taulist = β * [0.0001, 0.001, 0.01, 1 - 0.01, 1 - 0.001, 1 - 0.0001]
        itaus = [searchsortedfirst(taus, t) for t in taulist]
        sigma_taulist = [sigma_tau_dyn[it] for it in itaus]
        println(taulist)
        println(sigma_taulist)

        plot(taulist ./ β, sigma_taulist; label="\$r_s = $rs\$", color=color[i])

        # # RPA Wtilde0(q, iωₙ) / V(q)
        # wtilde_0_over_v_wn_q, _ = Interaction.RPAwrapped(Euv, rtol, qgrid.grid, para)
        # # Interaction.RPAwrapped(Euv, rtol, qgrid.grid, para; bugfix=true)
    end
    if length(rslist) ≤ 6
        legend(; loc="best")
    end
    xlabel("\$\\tau / \\beta\$")
    ylabel("\$\\Sigma(k = $ktarget, \\tau)\$")
    plt.tight_layout()
    savefig("test_sigma_rpa_kt.pdf")
    return
end

main()
