module Propagators

using ..ElectronGas
using ...FeynmanDiagram
using ..Lehmann
using LinearAlgebra
using ..UEG

"""First-order counter-term for G"""
function green2(Ek, τ, beta)
    if τ ≈ 0.0
        τ = -1.0e-10
    end

    s = 1.0
    if τ < 0.0
        τ += beta
        s = -s
    elseif τ >= beta
        τ -= beta
        s = -s
    end

    if Ek > 0.0
        c = exp(-beta * Ek)
        green = exp(-Ek * τ) / (1.0 + c)^2 * (τ - (beta - τ) * c)
    else
        c = exp(beta * Ek)
        green = exp(Ek * (beta - τ)) / (1.0 + c)^2 * (τ * c - (beta - τ))
    end

    return green *= s
end

"""Second/third-order counter-term for G"""
function green3(Ek, τ, beta=β)
    if τ ≈ 0.0
        τ = -1.0e-10
    end

    s = 1.0
    if τ < 0.0
        τ += beta
        s = -s
    elseif τ >= beta
        τ -= beta
        s = -s
    end

    if (Ek > 0.0)
        c = exp(-beta * Ek)
        green =
            exp(-Ek * τ) / (1.0 + c)^3.0 *
            (τ^2 / 2 - (beta^2 / 2 + beta * τ - τ^2) * c + (beta - τ)^2 * c^2 / 2.0)
    else
        c = exp(beta * Ek)
        green =
            exp(Ek * (beta - τ)) / (1.0 + c)^3 *
            (τ^2 * c^2 / 2.0 - (beta^2 / 2.0 + beta * τ - τ^2) * c + (beta - τ)^2 / 2.0)
    end

    green *= s
    return green
end

####################################################
# Bare Green's function and interaction evaluation #
####################################################

# TODO: Avoid this hack by generating all SOSEM obesrvables with extT = (1, 2)
@inline function remap_extT(taupair::Tuple{Int,Int}, extT2::Int)
    t1, t2 = taupair[1], taupair[2]
    if t1 == 2
        t1 = extT2
    elseif t1 == extT2
        t1 = 2
    end
    if t2 == 2
        t2 = extT2
    elseif t2 == extT2
        t2 = 2
    end
    return (t1, t2)
end

# Unscreened Coulomb interaction (for outer V lines of non-local SOSEM)
@inline function CoulombBareinstant(q, p::ParaMC)
    return KOinstant(q, p.e0, p.dim, 0.0, 0.0, p.kF)
end

"""Evaluate a bare Green's function line."""
function eval(id::BareGreenId, K, _, varT, additional::Tuple{ParaMC,W}) where {W}
    p, extT = additional
    β, me, μ, massratio = p.β, p.me, p.μ, p.massratio

    # HACK: Swap outgoing times into correct indices: extT[2] ↦ 2, 2 ↦ extT[2] (cost: ~ 1 ns)
    remapped_taupair = remap_extT(id.extT, extT[2])
    τin, τout = varT[remapped_taupair[1]], varT[remapped_taupair[2]]
    @debug "Remapped propagator time indices: $(id.extT) ↦ $remapped_taupair" maxlog = 1
    # τin, τout = varT[id.extT[1]], varT[id.extT[2]]

    # External time
    τ = τout - τin

    # Dashed line = Θ(τ)
    if id.order[3] == 1
        return (sign(τ) + 1) / 2.0
    end

    # Get energy
    k = norm(K)
    ϵ = k^2 / (2me * massratio) - μ
    # ϵ = kF / me * (k - kF)

    # Check for counterterms
    green = 0.0
    order = id.order[1]
    if order == 0
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
        if τ ≈ 0.0
            green = s * Spectral.kernelFermiT(-1e-8, ϵ, β)
        else
            green = s * Spectral.kernelFermiT(τ, ϵ, β)
        end
    elseif order == 1
        green = green2(ϵ, τ, β)
    elseif order == 2
        green = green3(ϵ, τ, β)
    elseif order == 3
        green = green3(ϵ, τ, β)
    else
        error("not implemented!")
    end

    # We have an overall sign difference relative to the Negle & Orland convention
    return -green
end

"""Evaluate a statically screened Coulomb interaction line."""
function eval(id::BareInteractionId, K, _, varT, additional::Tuple{ParaMC,W}) where {W}
    # additional = (mcparam, extT)
    p = additional[1]

    # TODO: Implement check for bare interaction using: is_bare = (order[end] = 1)
    e0, ϵ0, mass2 = p.e0, p.ϵ0, p.mass2
    qd = sqrt(dot(K, K))
    # Bare Coulomb interaction
    if id.order[4] == 1
        @debug "Bare V, T = $(id.extT)" maxlog = 5
        return CoulombBareinstant(qd, p)
        # Screened Coulomb interaction
    elseif id.order[2] == 0
        return Coulombinstant(qd, p)
        # Counterterms for screened interaction
    else
        invK = 1.0 / (qd^2 + mass2)
        return e0^2 / ϵ0 * invK * (mass2 * invK)^id.order[2]
    end
end

end  # module Propagators
