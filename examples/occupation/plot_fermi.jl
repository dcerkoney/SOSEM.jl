using ElectronLiquid
using Lehmann
using PyCall

# For saving/loading numpy data
@pyimport matplotlib.pyplot as plt
@pyimport mpl_toolkits.axes_grid1.inset_locator as il

# UEG parameters
p1 = ParaMC(; rs=3.0, beta=5.0)
p2 = ParaMC(; rs=3.0, beta=25.0)
p3 = ParaMC(; rs=3.0, beta=100.0)

for param in [p1, p2, p3]
    # HWHM of -f'(ϵ) = ln((√2 + 1) / (√2 - 1)) / β (in units of EF; beta = β EF)
    fermi_fwhm_over_ef = log((sqrt(2) + 1) / (sqrt(2) - 1)) / param.beta

    # ϵ_± = ϵF ± Δ_HWHM
    e_ef_minus = 1 - fermi_fwhm_over_ef
    e_ef_plus  = 1 + fermi_fwhm_over_ef

    # k_± = √(2 m ϵ_±) = kF √(1 ± Δ_HWHM / ϵF)
    # k_kf_minus = 1 - fermi_fwhm_over_ef / 2  # ≈ √(1 - Δ_HWHM / ϵF)    (to linear order in Δ)
    # k_kf_plus  = 1 + fermi_fwhm_over_ef / 2  # ≈ √(1 + Δ_HWHM / ϵF)  (to linear order in Δ)
    k_kf_minus = sqrt(e_ef_minus)  # = √(1 - Δ_HWHM / ϵF)
    k_kf_plus  = sqrt(e_ef_plus)   # = √(1 + Δ_HWHM / ϵF)

    # Use maximum broadening (sqrt is a non-linear transformation, so abs(k - k_+) ≠ abs(k - k_-)!)
    dk_max = max(abs(1 - k_kf_minus), abs(1 - k_kf_plus))
    k_kf_lesser  = 1 - dk_max
    k_kf_greater = 1 + dk_max

    # kkfs = collect(LinRange(0.0, 2.0, 500))
    # eefs = kkfs .^ 2 / (2 * param.me)

    eefs = collect(LinRange(0.0, 2.0, 500))
    kkfs = collect(LinRange(0.0, 2.0, 500))

    es  = eefs * param.EF
    ks  = kkfs * param.kF
    eks = ks .^ 2 / (2 * param.me)

    # Fermi function
    fes = -Spectral.kernelFermiT.(-1e-8, es .- param.μ, param.β)
    fks = -Spectral.kernelFermiT.(-1e-8, eks .- param.μ, param.β)

    # Plot f(ϵ) and FWHM_ϵ(f(ϵ))
    fig1, ax1 = plt.subplots()
    ax1.axvline(1.0; linestyle="--", linewidth=1, color="gray")
    ax1.axvspan(k_kf_lesser, k_kf_greater; color="0.9", label="\$\\mathrm{FWHM}_{k}(f_{k})\$")
    ax1.plot(kkfs, fks)
    fig1.tight_layout()
    fig1.savefig("fermi/fk_rs=$(param.rs)_beta=$(param.beta).pdf")

    # Plot f(k) and FWHM_k(f(k))
    fig2, ax2 = plt.subplots()
    ax2.axvline(1.0; linestyle="--", linewidth=1, color="gray")
    ax2.axvspan(e_ef_minus, e_ef_plus; color="0.9", label="\$\\mathrm{FWHM}_{ϵ}(f_{ϵ})\$")
    ax2.plot(eefs, fes)
    fig2.tight_layout()
    fig2.savefig("fermi/fe_rs=$(param.rs)_beta=$(param.beta).pdf")
    plt.close("all")
end
