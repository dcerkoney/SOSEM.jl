module Propagators

using ..ElectronGas
using ...FeynmanDiagram
using ..Lehmann
using LinearAlgebra
using ..UEG
using ..UEG_MC: @todo

####################################################
# Bare Green's function and interaction evaluation #
####################################################

# Unscreened Coulomb interaction (for outer V lines of non-local SOSEM)
@inline function CoulombBareinstant(q, p::ParaMC)
    return KOinstant(q, p.e0, p.dim, 0.0, 0.0, p.kF)
end

"""Evaluate a bare Green's function line."""
function eval(id::BareGreenId, K, _, varT, p::ParaMC)
    β, me, μ, massratio = p.β, p.me, p.μ, p.massratio

    # External time difference
    τin, τout = varT[id.extT[1]], varT[id.extT[2]]
    τ = τout - τin

    # Dashed line = Θ(τ)
    if id.order[3] == 1
        return (sign(τ) + 1) / 2.0
    end

    # Get energy
    k = norm(K)
    ϵ = k^2 / (2me * massratio) - μ
    # ϵ = kF / me * (k - kF)

    # Since τout = -δ for some SOSEM observables, it is possible to generate 
    # out-of-bounds time differences => use anti-periodicity on [0, β)
    s = 1.0
    if τ < 0.0
        τ += β
        s = -s
    elseif τ >= β
        τ -= β
        s = -s
    end

    # Check for counterterms; note that we have:
    # \partial^(n)_\mu g(Ek - \mu, \tau) = (-1)^n * Spectral.kernelFermiT_dωn
    # NOTE: We have an overall sign difference relative to the Negle & Orland convention
    green = -s
    order = id.order[1]
    if order == 0
        if τ == 0.0
            green *= Spectral.kernelFermiT(-1e-8, ϵ, β)
        else
            green *= Spectral.kernelFermiT(τ, ϵ, β)
        end
    elseif order == 1
        green *= -Spectral.kernelFermiT_dω(τ, ϵ, β)
    elseif order == 2
        green *= Spectral.kernelFermiT_dω2(τ, ϵ, β)
    elseif order == 3
        green *= -Spectral.kernelFermiT_dω3(τ, ϵ, β)
    else
        @todo
    end
    return green
end

"""Evaluate a statically screened Coulomb interaction line."""
function eval(id::BareInteractionId, K, _, varT, p::ParaMC)
    # TODO: Implement check for bare interaction using: is_bare = (order[end] = 1)
    e0, ϵ0, mass2 = p.e0, p.ϵ0, p.mass2
    qd = sqrt(dot(K, K))
    if id.order[4] == 1
        # Bare Coulomb interaction (from EOM)
        # @debug "Bare V, T = $(id.extT)" maxlog = 5
        return CoulombBareinstant(qd, p)
    elseif id.order[2] == 0
        # Screened Coulomb interaction
        return Coulombinstant(qd, p)
    else
        # Counterterms for screened interaction
        invK = 1.0 / (qd^2 + mass2)
        return e0^2 / ϵ0 * invK * (mass2 * invK)^id.order[2]
    end
end

end  # module Propagators
