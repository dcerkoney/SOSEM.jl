using ElectronGas, Parameters
using Lehmann, GreenFunc, CompositeGrids

const rs = 1.0
const beta = 1000.0
const mass2 = 1e-8
const dim = 3

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
    Fs::Float64, Fa::Float64 = landaufunc(q, n, param; massratio=massratio, kwargs...)
    Ks::Float64, Ka::Float64 = 0.0, 0.0
    Vinvs::Float64, Vinva::Float64 = Vinv_Bare(q, param)
    @unpack spin = param

    if abs(q) < EPS
        q = EPS
    end

    Πs::Float64 = spin * pifunc(q, n, param; kwargs...) * massratio
    Πa::Float64 = spin * pifunc(q, n, param; kwargs...) * massratio
    if regular
        Ks = testchargereg(Vinvs, Fs, Πs)
        Ka = testchargereg(Vinva, Fa, Πa)
    else
        Ks = testcharge(Vinvs, Fs, Πs)
        Ka = testcharge(Vinva, Fa, Πa)
    end

    return Ks, Ka
end

"""
    function WtildeF(q, n, param; pifunc = Polarization0_ZeroTemp, landaufunc = landauParameterTakada, Vinv_Bare = coulombinv, regular = false, kwargs...)

Dynamic part of test-charge test-charge interaction.
Returns the spin symmetric part and asymmetric part separately.

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
    \\frac{(v_q^{\\pm}) Π_0} {1 - (v_q^{\\pm} - f_q^{\\pm}) Π_0}.
```
otherwise, return
```math
    \\frac{(v_q^{\\pm})^2 Π_0} {1 - (v_q^{\\pm} - f_q^{\\pm}) Π_0}.
```
"""
function WtildeF(
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

function main()
    θ = 1e-3
    param = Parameter.defaultUnit(θ, rs)
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

    # test
    kgrid =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, maxK], [0.0, 2.0 * kF], Nk, minK, order)
    # kF_label = searchsortedfirst(kgrid.grid, kF)

    dlr = DLRGrid(Euv, β, rtol, false, :ph)
    wns = dlr.ωn
    
    WtildeF_kgrid = WtildeF(kgrid.grid, 0, param)

    # Get sigma in DLR basis
    sigma_dlr_dynamic = to_dlr(sigma_tau_dynamic)

    # Get the static and dynamic components of the Matsubara self-energy
    sigma_wn_static = sigma_tau_instant[1, :]
    sigma_wn_dynamic = to_imfreq(sigma_dlr_dynamic)

    # Get grids
    dlr = sigma_dlr_dynamic.mesh[1].dlr
    kgrid = sigma_dlr_dynamic.mesh[2]
    wns = dlr.ωn[dlr.ωn .≥ 0]  # retain positive Matsubara frequencies only


    integrand = real(G_ins) .* kgrid.grid .* kgrid.grid
    return c1_rpa_fl = -CompositeGrids.Interp.integrate1D(integrand, kgrid) / 2π^2
end

main()