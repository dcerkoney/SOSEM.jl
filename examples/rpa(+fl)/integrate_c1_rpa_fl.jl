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
    rslist = [1.0, 2.0, 5.0]
    # rslist = [1.0, 2.0, 3.0, 4.0, 5.0]

    # int_type = :ko
    int_type = :ko_const

    # Plot WtildeFs(q, τ = 0) vs rs
    fig, ax = plt.subplots()
    for (i, rs) in enumerate(rslist)
        param = Parameter.rydbergUnit(1.0 / beta, rs)
        kF, β = param.kF, param.β
        Euv, rtol = 1000 * param.EF, 1e-11

        # Nk, order = 8, 4
        # maxK, minK = 10kF, 1e-7kF
        # Nk, order = 11, 8
        maxK, minK = 20kF, 1e-8kF
        Nk, order = 16, 12
        # maxK, minK = 10kF, 1e-9kF
        # Euv, rtol = 1000 * param.EF, 1e-11
        # maxK, minK = 20param.kF, 1e-9param.kF
        # Nk, order = 16, 12

        # Grids for k and n
        qgrid = CompositeGrid.LogDensedGrid(
            :gauss,
            [0.0, maxK],
            [0.0, 2.0 * kF],
            Nk,
            minK,
            order,
        )
        # dlr = DLRGrid(Euv, β, rtol, false, :ph)

        # Get Landau parameter F⁰ₛ from Perdew & Wang compressibility fit
        Fs = get_Fs(rs)

        # Either use constant Fs from P&W or q-dependent Takada fit
        landaufunc =
            int_type == :ko_const ? Interaction.landauParameterConst :
            Interaction.landauParameterTakada

        # Get WtildeFs(q, iωₙ) / V(q)
        wtilde_Fs_over_v_wn_q = WtildeFswrapped(
            Euv,
            rtol,
            qgrid.grid,
            param;
            regular=true,
            int_type=int_type,
            landaufunc=landaufunc,
            Fs=Fs,
        )

        # Get WtildeFs(q, τ = 0) / V(q) from WtildeFs(q, iωₙ) / V(q)
        wtilde_Fs_over_v_dlr_q = to_dlr(wtilde_Fs_over_v_wn_q)
        wtilde_Fs_over_v_tau_q = to_imtime(wtilde_Fs_over_v_dlr_q)
        wtilde_Fs_over_v_q_inst = real(wtilde_Fs_over_v_tau_q[1, :])

        # RPA Wtilde0(q, iωₙ) / V(q)
        wtilde_0_over_v_wn_q, _ =
            Interaction.RPAwrapped(Euv, rtol, qgrid.grid, param; regular=true)

        # Get Wtilde0(q, τ = 0) / V(q) from WtildeFs(q, iωₙ) / V(q)
        wtilde_0_over_v_dlr_q = to_dlr(wtilde_0_over_v_wn_q)
        wtilde_0_over_v_tau_q = to_imtime(wtilde_0_over_v_dlr_q)
        # NOTE: Wtilde0 is spin-symmetric => idx1 = 1
        wtilde_0_over_v_q_inst = real(wtilde_0_over_v_tau_q[1, 1, :])

        # Infinitesimal time δ_τ from DLR grid
        # qwin = UnitRange(1, length(qgrid.grid))
        qwin = qgrid.grid .≤ 3.0 * kF
        qs = qgrid.grid[qwin]
        taus = wtilde_Fs_over_v_dlr_q.mesh[1].dlr.τ

        delta_tau_str = @sprintf "%.0g" taus[1]
        Fs_str = @sprintf "%.2g" Fs

        # Plot RPA(+FL) (Wtilde / V) at this rs
        ax.plot(
            qs / kF,
            wtilde_0_over_v_q_inst[qwin];
            color="C$i",
            linestyle="--",
            label="(RPA) \$r_s = $rs,\\, \\delta_\\tau = $delta_tau_str\$",
        )
        rpa_fl_label =
            int_type == :ko_const ?
            "(RPA+FL) \$r_s = $rs,\\, F_s = $Fs_str, \\delta_\\tau = $delta_tau_str\$" :
            "(RPA+FL) \$r_s = $rs,\\, \\delta_\\tau = $delta_tau_str\$"
        ax.plot(qs / kF, wtilde_Fs_over_v_q_inst[qwin]; color="C$i", label=rpa_fl_label)

        # Integrate RPA(+FL) 4πe²(Wtilde(q, τ = 0) / V(q)) over q ∈ ℝ⁺
        c1_rpa =
            -(2 * param.e0^2 / π) *
            CompositeGrids.Interp.integrate1D(wtilde_0_over_v_q_inst, qgrid)
        c1_rpa_fl =
            -(2 * param.e0^2 / π) *
            CompositeGrids.Interp.integrate1D(wtilde_Fs_over_v_q_inst, qgrid)
        println("(rs = $rs) C⁽¹⁾_{RPA} = $c1_rpa, C⁽¹⁾_{RPA+FL} = $c1_rpa_fl")
    end
    ax.legend(; loc="best")
    ax.set_xlabel("\$q / k_F\$")
    ax.set_ylabel("\$\\widetilde{W}_f(q, \\tau = \\delta_\\tau) / V(q)\$")
    plt.tight_layout()
    fig.savefig("rpa_and_rpa_fl_wtilde_over_V_q_inst_rs=$(rslist)_$int_type.pdf")
    return
end

main()