using PyCall
using Printf
using ElectronGas, Parameters
using Lehmann, GreenFunc, CompositeGrids

@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

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
    return error("Not yet implemented!")
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

"""
    function testchargereg(V::Float64, F::Float64, Π::Float64)

Return V Π / (1 - (V - F)Π), which is the dynamic part of the test-charge test-charge interaction W_f divided by V.

#Arguments:
- Vinv: inverse bare interaction
- F: Landau parameter
- Π: polarization
"""
function testchargereg(Vinv::Float64, F::Float64, Π::Float64)
    # (V - F) Π / (1 - (V - F)Π) * (V / (V - F))
    K = 0
    if Vinv ≈ Inf
        K = 0
    else
        # K = bubbledysonreg(Vinv, F, Π) / (1 - F * Vinv)
        K = Π / (Vinv - Π * (1 - F * Vinv))
    end
    @assert !isnan(K) "nan at Vinv=$Vinv, F=$F, Π=$Π"
    return K
end

"""
    function testcharge(V::Float64, F::Float64, Π::Float64)

Return V^2 Π / (1 - (V - F)Π), which is the dynamic part of the test-charge test-charge interaction W_f.

#Arguments:
- Vinv: inverse bare interaction
- F: Landau parameter
- Π: polarization
"""
function testcharge(Vinv::Float64, F::Float64, Π::Float64)
    K = 0
    if Vinv ≈ Inf
        K = 0
    else
        K = Π / (Vinv - Π * (1 - F * Vinv)) / Vinv
    end
    @assert !isnan(K) "nan at Vinv=$Vinv, F=$F, Π=$Π"
    return K
end

function testchargecorrection(
    q::Float64,
    n::Int,
    param;
    pifunc=Polarization0_ZeroTemp,
    landaufunc=landauParameterTakada,
    Vinv_Bare=coulombinv,
    regular=false,
    massratio=1.0,
    kwargs...,
)
    Fs::Float64, _ = landaufunc(q, n, param; massratio=massratio, kwargs...)
    Ks::Float64 = 0.0
    Vinvs::Float64, _ = Vinv_Bare(q, param)
    @unpack spin = param

    if abs(q) < Convention.EPS
        q = Convention.EPS
    end

    Πs::Float64 = spin * pifunc(q, n, param; kwargs...) * massratio
    Ks = regular ? testchargereg(Vinvs, Fs, Πs) : testcharge(Vinvs, Fs, Πs)
    return Ks
end

"""
    function WtildeFs(q, n, param; pifunc = Polarization0_ZeroTemp, landaufunc = landauParameterTakada, Vinv_Bare = coulombinv, regular = false, kwargs...)

Dynamic part of the (spin symmetric) test-charge test-charge interaction.

#Arguments:
 - q: momentum
 - n: matsubara frequency given in integer s.t. ωn=2πTn
 - param: other system parameters
 - pifunc: caller to the polarization function 
 - landaufunc: caller to the Landau parameter (exchange-correlation kernel)
 - Vinv_Bare: caller to the bare Coulomb interaction
 - regular: regularized RPA or not

# Return:
If set to be regularized, it returns the dynamic part of effective interaction divided by ``v_q``
```math
    \\frac{(v_q^{+}) Π_0} {1 - (v_q^{+} - f_q^{+}) Π_0}.
```
otherwise, return
```math
    \\frac{(v_q^{+})^2 Π_0} {1 - (v_q^{+} - f_q^{+}) Π_0}.
```
"""
function WtildeFs(
    q,
    n,
    param;
    pifunc=Polarization.Polarization0_ZeroTemp,
    landaufunc=Interaction.landauParameterTakada,
    Vinv_Bare=Interaction.coulombinv,
    regular=false,
    kwargs...,
)
    return testchargecorrection(
        q,
        n,
        param;
        pifunc=pifunc,
        landaufunc=landaufunc,
        Vinv_Bare=Vinv_Bare,
        regular=regular,
        kwargs...,
    )
end

"""
    function WtildeFswrapped(Euv, rtol, sgrid::SGT, param; int_type=:ko,
        pifunc=Polarization0_ZeroTemp, landaufunc=landauParameterTakada, Vinv_Bare=coulombinv, kwargs...) where {SGT}

Return dynamic part of the (spin symmetric) test-charge test-charge interaction Green's function as a MeshArray in ImFreq and q-grid mesh.

#Arguments:
 - Euv: Euv of DLRGrid
 - rtol: rtol of DLRGrid
 - sgrid: momentum grid
 - param: other system parameters
 - pifunc: caller to the polarization function
 - landaufunc: caller to the Landau parameter (exchange-correlation kernel)
 - Vinv_Bare: caller to the bare Coulomb interaction
"""
function WtildeFswrapped(
    Euv,
    rtol,
    sgrid::SGT,
    param;
    int_type=:ko,
    pifunc=Polarization.Polarization0_ZeroTemp,
    landaufunc=Interaction.landauParameterTakada,
    Vinv_Bare=Interaction.coulombinv,
    kwargs...,
) where {SGT}
    # TODO: innerstate should be in the outermost layer of the loop. Hence, the functions such as KO and Vinv_Bare should be fixed with inner state as argument.
    @unpack β = param
    wn_mesh = GreenFunc.ImFreq(β, BOSON; Euv=Euv, rtol=rtol, symmetry=:ph)
    wtilde_Fs_wn_k = GreenFunc.MeshArray(wn_mesh, sgrid; dtype=ComplexF64)
    for (ki, k) in enumerate(sgrid)
        for (ni, n) in enumerate(wn_mesh.grid)
            wtilde_Fs_wn_k[ni, ki] = WtildeFs(
                k,
                n,
                param;
                pifunc=pifunc,
                landaufunc=landaufunc,
                Vinv_Bare=Vinv_Bare,
                kwargs...,
            )
        end
    end
    return wtilde_Fs_wn_k
end

function main()
    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # System parameters
    dim = 3
    mass2 = 1e-8
    # beta = 40.0 
    beta = 1000.0
    rs = 1.0

    # int_type = :ko
    int_type = :ko_const

    param = Parameter.rydbergUnit(1.0 / beta, rs)
    kF, β = param.kF, param.β
    Euv, rtol = 10000 * param.EF, 1e-11

    # # Converge total number of qgrid-points (tuned by Nk & order)
    # Nks = [8, 11, 16, 20]
    # orders = [4, 8, 12, 16]
    # maxK = 30kF
    # minK = 1e-8kF

    # # Converged Nk and order
    Nk = 20
    order = 16

    # # Converge maxK (10kF to 100kF)
    minK = 1e-8kF
    maxKs = 1000kF * collect(1:100)
    # maxKs = [30kF * collect(1:10); 1000kF]

    # # Converged maxK
    # maxK = 300kF

    # # Converge minK (1e-4kF to 1e-13kF)
    # minKs = 1e-10kF * (10 .^ collect(9:-1:0))
    # # For log plot
    # minus_log_minKkFs = round.(-log.(10, minKs / kF); sigdigits=13)

    # # Converged minK (inessential)
    # minK = 1e-8kF

    nqs = []
    c1_rpas = []
    c1_rpa_fls = []
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    # for (Nk, order) in zip(Nks, orders)  # Test 1: Converge Nq
        for maxK in maxKs                    # Test 2: Converge maxK
        # for minK in minKs                    # Test 3: Converge minK

        # Grids for k and n
        qgrid = CompositeGrid.LogDensedGrid(
            :gauss,
            [0.0, maxK],
            [0.0, 2.0 * kF],
            Nk,
            minK,
            order,
        )
        push!(nqs, length(qgrid.grid))
        # dlr = DLRGrid(Euv, β, rtol, false, :ph)

        # Get Landau parameter F⁰ₛ from Perdew & Wang compressibility fit
        Fs = get_Fs(rs)

        # Either use constant Fs from P&W or q-dependent Takada fit
        landaufunc =
            int_type == :ko_const ? Interaction.landauParameterConst :
            Interaction.landauParameterTakada

        # Get KO_dyn(q, iωₙ) / V(q)
        KO_dyn_over_v_wn_q, _ = Interaction.KOwrapped(
            Euv,
            rtol,
            qgrid.grid,
            param;
            regular=true,
            int_type=int_type,
            landaufunc=landaufunc,
            Fs=Fs,
        )

        # Get KO_dyn(q, τ = 0) / V(q) from KO_dyn(q, iωₙ) / V(q)
        KO_dyn_over_v_dlr_q = to_dlr(KO_dyn_over_v_wn_q)
        KO_dyn_over_v_tau_q = to_imtime(KO_dyn_over_v_dlr_q)
        # NOTE: Sigma is spin-symmetric => idx1 = 1, and idx2 = 1 for τ = 0
        KO_dyn_over_v_q_inst = real(KO_dyn_over_v_tau_q[1, 1, :])

        # RPA Wtilde0(q, iωₙ) / V(q)
        wtilde_0_over_v_wn_q, _ =
            Interaction.RPAwrapped(Euv, rtol, qgrid.grid, param; regular=true)

        # Get Wtilde0(q, τ = 0) / V(q) from WtildeFs(q, iωₙ) / V(q)
        wtilde_0_over_v_dlr_q = to_dlr(wtilde_0_over_v_wn_q)
        wtilde_0_over_v_tau_q = to_imtime(wtilde_0_over_v_dlr_q)
        # NOTE: Wtilde0 is spin-symmetric => idx1 = 1
        wtilde_0_over_v_q_inst = real(wtilde_0_over_v_tau_q[1, 1, :])

        # Integrate RPA(+FL) 4πe²(W_dyn(q, τ = 0) / V(q)) over q ∈ ℝ⁺ for W ∈ {W_0, W_KO}
        c1_rpa =
            -(2 * param.e0^2 / π) *
            CompositeGrids.Interp.integrate1D(wtilde_0_over_v_q_inst, qgrid)
        c1_rpa_fl =
            -(2 * param.e0^2 / π) *
            CompositeGrids.Interp.integrate1D(KO_dyn_over_v_q_inst, qgrid)
        push!(c1_rpas, c1_rpa)
        push!(c1_rpa_fls, c1_rpa_fl)
    end

    # # Test 1
    # println(
    #     "RPA(+FL) Percent change Nq = $(nqs[end]) vs Nq = $(nqs[end-1]): ",
    #     (c1_rpas[end] - c1_rpas[end - 1]) / c1_rpas[end - 1] * 100,
    #     ", ",
    #     (c1_rpa_fls[end] - c1_rpa_fls[end - 1]) / c1_rpa_fls[end - 1] * 100,
    # )

    # Test 2
    println(
    "RPA(+FL) Percent change 300kF vs 270kF: ",
    (c1_rpas[end] - c1_rpas[end - 1]) / c1_rpas[end - 1] * 100,
    ", ",
    (c1_rpa_fls[end] - c1_rpa_fls[end - 1]) / c1_rpa_fls[end - 1] * 100,
    )

    # # Test 1
    # ax.plot(nqs, c1_rpas, "o-")
    # ax2.plot(nqs, c1_rpa_fls, "o-")
    # ax.set_xlabel("\$N_q\$")
    # ax2.set_xlabel("\$N_q\$")

    # Test 2
    ax.plot(maxKs / kF, c1_rpas, "o-")
    ax2.plot(maxKs / kF, c1_rpa_fls, "o-")
    ax.set_xlabel("\$q_{\\mathrm{max}} / k_F\$")
    ax2.set_xlabel("\$q_{\\mathrm{max}} / k_F\$")
    # ax.set_xticks(maxKs / kF)
    # ax2.set_xticks(maxKs / kF)

    # # Test 3
    # ax.plot(minus_log_minKkFs, c1_rpas, "o-")
    # ax2.plot(minus_log_minKkFs, c1_rpa_fls, "o-")
    # ax.set_xlabel("\$-\\log_{10} \\left(q_{\\mathrm{min}} / k_F\\right)\$")
    # ax2.set_xlabel("\$-\\log_{10} \\left(q_{\\mathrm{min}} / k_F\\right)\$")

    # rpa_fl_label =
    #     int_type == :ko_const ? "RPA+FL (const. \$F^0_s(r_s)\$)" :
    #     "RPA+FL (Takada \$F^0_s(r_s, q)\$)"
    ax.set_ylabel("\$C^{(1)l}_{\\mathrm{RPA}} \\,/\\, \\epsilon^2_{\\mathrm{Ry}}\$")
    ax2.set_ylabel("\$C^{(1)l}_{\\mathrm{RPA+FL}} \\,/\\, \\epsilon^2_{\\mathrm{Ry}}\$")
    plt.tight_layout()
    fig.savefig(
        "convergence_tests/$(int_type)/c1_rpa_rs=$(rs)_beta=$(beta)_$(int_type)" *
        # "_Nq_convergence.pdf",    # Test 1
        # "_maxK_convergence.pdf",    # Test 2
        "_maxK_convergence_big.pdf",    # Test 2
        # "_minK_convergence.pdf",  # Test 3
    )
    fig2.savefig(
        "convergence_tests/$(int_type)/c1_rpa_fl_rs=$(rs)_beta=$(beta)_$(int_type)" *
        # "_Nq_convergence.pdf",    # Test 1
        # "_maxK_convergence.pdf",    # Test 2
        "_maxK_convergence_big.pdf",    # Test 2
        # "_minK_convergence.pdf",  # Test 3
    )
    return
end

main()