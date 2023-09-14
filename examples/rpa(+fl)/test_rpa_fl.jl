using SOSEM
using PyCall
using Printf
using ElectronGas, Parameters
using Lehmann, GreenFunc, CompositeGrids

@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

const ktarget = 0.0

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
    function TTreg(V::Float64, F::Float64, Π::Float64)

Return V Π / (1 - (V - F)Π), which is the dynamic part of the test-charge test-charge interaction W_f divided by V.

#Arguments:
- Vinv: inverse bare interaction
- F: Landau parameter
- Π: polarization
"""
function TTreg(Vinv::Float64, F::Float64, Π::Float64)
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
    function TT(V::Float64, F::Float64, Π::Float64)

Return V^2 Π / (1 - (V - F)Π), which is the dynamic part of the test-charge test-charge interaction W_f.

#Arguments:
- Vinv: inverse bare interaction
- F: Landau parameter
- Π: polarization
"""
function TT(Vinv::Float64, F::Float64, Π::Float64)
    K = 0
    if Vinv ≈ Inf
        K = 0
    else
        K = Π / (Vinv - Π * (1 - F * Vinv)) / Vinv
    end
    @assert !isnan(K) "nan at Vinv=$Vinv, F=$F, Π=$Π"
    return K
end

function TTcorrection(
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
    Ks = regular ? TTreg(Vinvs, Fs, Πs) : TT(Vinvs, Fs, Πs)
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
    return TTcorrection(
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

function get_c1_rpa(para::Parameter.Para)
    # Small params
    kF = para.kF
    Euv, rtol = 1000 * para.EF, 1e-11
    maxK, minK = 30 * kF, 1e-8 * kF
    # Nk, order = 20, 12
    Nk, order = 11, 8

    # qgrid for integration
    qgrid =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, maxK], [0.0, 2.0 * kF], Nk, minK, order)

    # RPA Wtilde0(q, iωₙ) / V(q)
    wtilde_0_over_v_wn_q, _ = Interaction.RPAwrapped(Euv, rtol, qgrid.grid, para)
    # Interaction.RPAwrapped(Euv, rtol, qgrid.grid, param; bugfix=true)
    @assert maximum(imag(wtilde_0_over_v_wn_q[1, 1, :])) ≤ 1e-10

    # Get Wtilde0(q, τ = 0) / V(q) from WtildeFs(q, iωₙ) / V(q)
    wtilde_0_over_v_dlr_q = to_dlr(wtilde_0_over_v_wn_q)
    wtilde_0_over_v_tau_q = to_imtime(wtilde_0_over_v_dlr_q)
    # NOTE: Wtilde0 is spin-symmetric => idx1 = 1
    wtilde_0_over_v_q_inst = real(wtilde_0_over_v_tau_q[1, 1, :])

    # Integrate RPA(+FL) 4πe²(Wtilde(q, τ = 0) / V(q)) over q ∈ ℝ⁺
    rpa_integrand = wtilde_0_over_v_q_inst
    c1_rpa = -(2 * para.e0^2 / π) * CompositeGrids.Interp.integrate1D(rpa_integrand, qgrid)

    # println("\nrs = $rs:" * "\nC⁽¹⁾_{RPA} = $c1_rpa")
    return c1_rpa
end

function get_c1_rpa_fl(para::Parameter.Para; int_type=:ko_moroni, w_type=:tt)
    # Small params
    kF = para.kF
    Euv, rtol = 1000 * para.EF, 1e-11
    maxK, minK = 30 * kF, 1e-8 * kF
    # Nk, order = 20, 12
    Nk, order = 11, 8

    # Get Landau parameter F⁰ₛ from Perdew & Wang compressibility fit
    rs = round(para.rs; sigdigits=13)
    Fs = get_Fs(rs)
    if int_type == :ko_const
        println("Fs = $Fs, fs = $(-Fs / para.NF)")
    end
    @assert Fs ≤ 0

    # Either use constant Fs from P&W, q-dependent Takada ansatz, or Corradini fit to Moroni DMC data
    if int_type == :ko_const
        landaufunc = Interaction.landauParameterConst
    elseif int_type == :ko_takada
        landaufunc = Interaction.landauParameterTakada
    elseif int_type == :ko_moroni
        landaufunc = Interaction.landauParameterMoroni
    else
        throw(UndefVarError(landaufunc))
    end

    # qgrid for integration
    qgrid =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, maxK], [0.0, 2.0 * kF], Nk, minK, order)

    if w_type == :ee
        Wdyn = Interaction.KOwrapped
    elseif w_type == :et
        Wdyn = Interaction.ETwrapped
    elseif w_type == :tt
        Wdyn = Interaction.TTwrapped
    else
        throw(UndefVarError(w_type))
    end

    # Get Wtilde_{ET/KO}(q, iωₙ) / (V(q) - fs) or Wtilde_{TT}(q, iωₙ) / V(q)
    wdyn_over_v_wn_q, _ = Wdyn(
        Euv,
        rtol,
        qgrid.grid,
        para;
        regular=true,
        int_type=int_type,
        landaufunc=landaufunc,
        Fs=-Fs,  # NOTE: NEFT uses opposite sign convention!
    )
    @assert maximum(imag(wdyn_over_v_wn_q[1, 1, :])) ≤ 1e-10

    # Get Wdyn(q, τ) / (V(q) (- fs)) from Wdyn(q, iωₙ) / (V(q) (- fs))
    wdyn_over_v_dlr_q = to_dlr(wdyn_over_v_wn_q)
    wdyn_over_v_tau_q = to_imtime(wdyn_over_v_dlr_q)

    # Get Wdyn_s(q, τ = 0) / (V(q) (- fs)), keeping only the
    # spin-symmetric part of Wdyn (we take fa := 0)
    wdyn_s_over_v_q_f_inst = real(wdyn_over_v_tau_q[1, 1, :])

    Vinvs = [Interaction.coulombinv(q, para)[1] for q in qgrid.grid]  # 1 / Vs(q)
    if w_type == :tt
        # Wdyn_s / Vs = Wdyn_s(...; regular=true) for TT
        rpa_fl_integrand = wdyn_s_over_v_q_f_inst
    else
        # Wdyn_s / Vs = (Wdyn_s / (Vs - fs)) * (1 - fs Vinvs) 
        #                  = Wdyn_s(...; regular=true) * (1 - fs Vinvs) for ET/KO
        local fs_int_type
        if int_type == :ko_moroni
            # NOTE: The Moroni DMC data for fs is q-dependent!
            fs_int_type =
                [Interaction.landauParameterMoroni(q, 0, para)[1] for q in qgrid.grid]
        elseif int_type == :ko_takada
            # NOTE: The Takada ansatz for fs is q-dependent!
            fs_int_type =
                [Interaction.landauParameterTakada(q, 0, para)[1] for q in qgrid.grid]
        elseif int_type == :ko_const
            # fs = -Fs / NF
            fs_int_type = -Fs / para.NF
        else
            error("Not yet implemented!")
        end
        rpa_fl_integrand = @. wdyn_s_over_v_q_f_inst * (1.0 - fs_int_type * Vinvs)
    end
    # Integrate RPA+FL 4πe²(Wtilde(q, τ = 0) / V(q)) over q ∈ ℝ⁺ (regular = true)
    c1_rpa_fl =
        -(2 * para.e0^2 / π) * CompositeGrids.Interp.integrate1D(rpa_fl_integrand, qgrid)

    # println("rs = $rs:" * "\nC⁽¹⁾_{RPA+FL} = $c1_rpa_fl")
    return c1_rpa_fl
end

function main()
    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # System parameters
    dim = 3
    mass2 = 1e-8
    beta = 1000.0

    rslist = [0.1, 1.0, 5.0]
    # rs = 1.0
    # rs = 5.0

    # w_type = :ee
    w_type = :et

    int_types = [:rpa, :ko_moroni]

    # int_types = [:rpa, :ko_const, :ko_takada, :ko_moroni]
    # int_type = :rpa
    # int_type = :ko_moroni
    # int_type = :ko_takada
    # int_type = :ko_const

    for rs in rslist
        println("\n(rs = $rs):\n")
        para = Parameter.rydbergUnit(1.0 / beta, rs)
        EF, kF, β = para.EF, para.kF, para.β

        # Small params
        Euv, rtol = 1000 * para.EF, 1e-11
        maxK, minK = 30 * para.kF, 1e-8 * para.kF
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
        # dlr = DLRGrid(Euv, β, rtol, false, :ph)

        for int_type in int_types
            # Get Landau parameter F⁰ₛ from Perdew & Wang compressibility fit
            Fs = get_Fs(rs)
            if int_type == :ko_const
                println("Fs = $Fs, fs = $(-Fs / para.NF)")
            end
            @assert Fs ≤ 0

            # Either use constant Fs from P&W, q-dependent Takada ansatz, or Corradini fit to Moroni DMC data
            if int_type == :rpa
                landaufunc = Interaction.landauParameter0
            elseif int_type == :ko_const
                landaufunc = Interaction.landauParameterConst
            elseif int_type == :ko_takada
                landaufunc = Interaction.landauParameterTakada
            elseif int_type == :ko_moroni
                landaufunc = Interaction.landauParameterMoroni
            else
                throw(UndefVarError(landaufunc))
            end

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
                w_type=w_type,
                landaufunc=landaufunc,
                Fs=-Fs,  # NOTE: landauParameterConst uses opposite sign convention!
            )

            # sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(para; int_type=int_type)
            zfactor, kF = SelfEnergy.zfactor(para, sigma_tau_dynamic)
            massratio, kF = SelfEnergy.massratio(para, sigma_tau_dynamic, sigma_tau_instant)
            println("int_type = $int_type, w_type = $w_type")
            println("zfactor = $zfactor, massratio = $massratio\n")
        end

        # Plot sigma comparison
        wmax_plot = 50
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        ic = 0

        rpa_c1 = get_c1_rpa(para)
        B_rpa = rpa_c1 / EF^2

        srpa, _ = SelfEnergy.G0W0(
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
        srpa_dlr = to_dlr(srpa)

        # Get the static and dynamic components of the Matsubara self-energy
        srpa_dyn = to_imfreq(srpa_dlr)

        # Get grids
        dlr = srpa_dlr.mesh[1].dlr
        kgrid = srpa_dlr.mesh[2]
        wns = dlr.ωn[dlr.ωn .≥ 0]  # retain positive Matsubara frequencies only
        wns_over_EF = wns / EF

        # Evaluate at a single kpoint near k = ktarget
        ikval = searchsortedfirst(kgrid, ktarget)
        kval = kgrid[ikval]
        if abs(kval - ktarget) > 1e-3
            @warn "kval = $kval is not within atol = $atol of ktarget = $ktarget!"
        end
        # println(
        #     "Obtaining self-energy at grid point k = $kval near target k-point ktarget = $ktarget",
        # )

        # Self-energies at k = 0 and k = kF for positive frequencies
        srpa_dyn_kval = srpa_dyn[:, ikval][dlr.ωn .≥ 0]
        B_rpa_meas = -wns_over_EF .* imag.(srpa_dyn_kval) / EF

        max_sigma = 0
        it = :ko_moroni
        for wt in [:tt, :et, :ee]
            sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(
                para;
                Euv=Euv,
                rtol=rtol,
                Nk=Nk,
                minK=minK,
                maxK=maxK,
                order=order,
                int_type=it,
                w_type=wt,
                landaufunc=Interaction.landauParameterMoroni,
            )

            # Get sigma in DLR basis
            sigma_dlr_dynamic = to_dlr(sigma_tau_dynamic)

            # Get the static and dynamic components of the Matsubara self-energy
            sigma_wn_dynamic = to_imfreq(sigma_dlr_dynamic)
            # sigma_wn_static = sigma_tau_instant[1, :]

            # Get grids
            dlr = sigma_dlr_dynamic.mesh[1].dlr
            kgrid = sigma_dlr_dynamic.mesh[2]
            wns = dlr.ωn[dlr.ωn .≥ 0]  # retain positive Matsubara frequencies only
            wns_over_EF = wns / EF

            # Evaluate at a single kpoint near k = ktarget
            ikval = searchsortedfirst(kgrid, ktarget)
            kval = kgrid[ikval]
            if abs(kval - ktarget) > 1e-3
                @warn "kval = $kval is not within atol = $atol of ktarget = $ktarget!"
            end
            # println(
            #     "Obtaining self-energy at grid point k = $kval near target k-point ktarget = $ktarget",
            # )

            # Self-energies at k = 0 and k = kF for positive frequencies
            sigma_wn_dynamic_kval = sigma_wn_dynamic[:, ikval][dlr.ωn .≥ 0]
            # sigma_wn_static_kval = sigma_wn_static[ikval]

            # Get first-order RPA+FL moment at k = ktarget
            rpa_fl_c1 = get_c1_rpa_fl(para; int_type=it, w_type=wt)
            B_rpa_fl  = rpa_fl_c1 / EF^2
            println("First-order RPA moment (Rydberg): ", rpa_c1)
            println("First-order RPA+FL moment (Rydberg): ", rpa_fl_c1)

            # P1: -ImSigma(wn)
            ax.plot(
                wns_over_EF,
                -imag.(sigma_wn_dynamic_kval) / EF;
                # wt == :et ? "-." : "-";
                color="C$ic",
                label="$wt",
            )
            ax.plot(
                wns_over_EF,
                B_rpa_fl ./ wns_over_EF,
                "--";
                color="C$ic",
                label="\$B_{\\mathrm{RPA+FL}} / \\omega_n\$, $wt",
            )
            # if wt != :ee
            # P2: -wnImSigma(wn)
            ax2.plot(
                wns_over_EF,
                -wns_over_EF .* imag.(sigma_wn_dynamic_kval) / EF;
                color="C$ic",
                label="$wt",
            )
            ax2.plot(
                wns_over_EF,
                B_rpa_fl * one.(wns_over_EF),
                "--";
                color="C$ic",
                label="\$B_{\\mathrm{RPA+FL}}\$, $wt",
            )
            # end
            ic += 1
            max_sigma = max(max_sigma, maximum(-imag.(sigma_wn_dynamic_kval)) / EF)
        end
        # P1: -ImSigma(wn)
        ax.plot(wns_over_EF, -imag.(srpa_dyn_kval) / EF; color="k", label="RPA")
        ax.plot(
            wns_over_EF,
            B_rpa ./ wns_over_EF,
            "--";
            color="k",
            label="\$B_{\\mathrm{RPA}} / \\omega_n\$",
        )
        ax.set_xlim(0, wmax_plot)
        ax.set_ylim(0, 2.0 * max_sigma)
        ax.set_xlabel("\$\\omega_n\$")
        ax.set_ylabel(
            "\$-\\mathrm{Im}\\Sigma^{dyn}_{\\mathrm{RPA+FL}}(k = 0, i\\omega_n)\$",
        )
        ax.legend(; loc="best")
        plt.tight_layout()
        fig.savefig("sigma_rpa_fl_comparison_rs=$rs.pdf")
        # P2: -wnImSigma(wn)
        ax2.plot(wns_over_EF, B_rpa_meas; color="k", label="RPA")
        ax2.plot(
            wns_over_EF,
            B_rpa * one.(wns_over_EF),
            "--";
            color="k",
            label="\$B_{\\mathrm{RPA}}\$",
        )
        ax2.set_xlabel("\$\\omega_n\$")
        ax2.set_ylabel("\$B(r_s, i\\omega_n)\$")
        ax2.legend(; loc="best")
        plt.tight_layout()
        fig2.savefig("B_rpa_fl_comparison_rs=$rs.pdf")
    end

    return

    # # Get WtildeFs(q, iωₙ) / V(q)
    # wtilde_Fs_over_v_wn_q = WtildeFswrapped(
    #     Euv,
    #     rtol,
    #     qgrid.grid,
    #     para;
    #     regular=true,
    #     int_type=int_type,
    #     landaufunc=landaufunc,
    #     Fs=-Fs,  # NOTE: landauParameterConst uses opposite sign convention!
    # )

    # Get Wtilde_KO(q, iωₙ) / (V(q) - fs)
    wtilde_KO_over_v_wn_q, _ = Interaction.KOwrapped(
        Euv,
        rtol,
        qgrid.grid,
        para;
        regular=true,
        int_type=int_type,
        landaufunc=landaufunc,
        Fs=-Fs,  # NOTE: landauParameterConst uses opposite sign convention!
    )

    # Get Wtilde_KO(q, τ) / (V(q) - fs) from Wtilde_KO(q, iωₙ) / (V(q) - fs)
    wtilde_KO_over_v_dlr_q = to_dlr(wtilde_KO_over_v_wn_q)
    wtilde_KO_over_v_tau_q = to_imtime(wtilde_KO_over_v_dlr_q)

    # Get Wtilde_KO_s(q, τ = 0) / (V(q) - fs), keeping only the
    # spin-symmetric part of wtilde_KO (we define fa := 0)
    wtilde_KO_s_over_v_q_f_inst = real(wtilde_KO_over_v_tau_q[1, 1, :])

    # RPA Wtilde0(q, iωₙ) / V(q)
    wtilde_0_over_v_wn_q, _ =
        Interaction.RPAwrapped(Euv, rtol, qgrid.grid, para; bugfix=true)
    # Interaction.RPAwrapped(Euv, rtol, qgrid.grid, para)
    # Interaction.RPAwrapped(Euv, rtol, qgrid.grid, para; regular=true)

    # Get Wtilde0(q, τ = 0) / V(q) from WtildeFs(q, iωₙ) / V(q)
    wtilde_0_over_v_dlr_q = to_dlr(wtilde_0_over_v_wn_q)
    wtilde_0_over_v_tau_q = to_imtime(wtilde_0_over_v_dlr_q)

    # NOTE: Wtilde0 is spin-symmetric => idx1 = 1
    wtilde_0_over_v_q_inst = real(wtilde_0_over_v_tau_q[1, 1, :])

    @assert maximum(imag(wtilde_0_over_v_wn_q[1, 1, :])) ≤ 1e-10
    @assert maximum(imag(wtilde_KO_over_v_wn_q[1, 1, :])) ≤ 1e-10

    # Get Wtilde_KO(q, iωₙ)
    wtilde_KO_wn_q, _ = Interaction.KOwrapped(
        Euv,
        rtol,
        qgrid.grid,
        para;
        regular=false,
        int_type=int_type,
        landaufunc=landaufunc,
        Fs=-Fs,  # NOTE: landauParameterConst uses opposite sign convention!
    )

    # Get Wtilde_KO(q, τ) from Wtilde_KO(q, iωₙ)
    wtilde_KO_dlr_q = to_dlr(wtilde_KO_wn_q)
    wtilde_KO_tau_q = to_imtime(wtilde_KO_dlr_q)

    # Get Wtilde_KO_s(q, τ = 0), keeping only the
    # spin-symmetric part of wtilde_KO (we define fa := 0)
    wtilde_KO_s_q_inst = real(wtilde_KO_tau_q[1, 1, :])

    local fs_int_type
    if int_type == :ko_moroni
        # NOTE: The Moroni DMC data for fs is q-dependent!
        fs_int_type = [Interaction.landauParameterMoroni(q, 0, para)[1] for q in qgrid.grid]
    elseif int_type == :ko_takada
        # NOTE: The Takada ansatz for fs is q-dependent!
        fs_int_type = [Interaction.landauParameterTakada(q, 0, para)[1] for q in qgrid.grid]
    elseif int_type == :ko_const
        # fs = -Fs / NF
        fs_int_type = -Fs / para.NF
    else
        error("Not yet implemented!")
    end

    # 1 / Vs(q)
    Vinvs = [Interaction.coulombinv(q, para)[1] for q in qgrid.grid]

    # Wtilde_KO_s / Vs = (Wtilde_KO_s / (Vs - fs)) * (1 - fs Vinvs) 
    #                  = Wtilde_KO_s(...; regular=true) * (1 - fs Vinvs)
    rpa_fl_integrand = @. wtilde_KO_s_over_v_q_f_inst * (1.0 - fs_int_type * Vinvs)

    # Integrate RPA+FL 4πe²(Wtilde(q, τ = 0) / V(q)) over q ∈ ℝ⁺ (regular = true)
    c1_rpa_fl =
        -(2 * para.e0^2 / π) * CompositeGrids.Interp.integrate1D(rpa_fl_integrand, qgrid)

    # Integrate RPA+FL q² Wtilde(q, τ = 0) over q ∈ ℝ⁺ (regular = false)
    c1_rpa_fl_v2 =
        -(1 / 2π^2) *
        CompositeGrids.Interp.integrate1D(wtilde_KO_s_q_inst .* qgrid .* qgrid, qgrid)

    println(
        "($int_type) rs = $rs:" *
        "\nC⁽¹⁾_{RPA+FL} = $c1_rpa_fl" *
        "\n(regular = false) C⁽¹⁾_{RPA+FL} = $c1_rpa_fl_v2",
        # # NOTE: Agrees with regular = true for int_type = :ko!
    )
    return
end

main()
