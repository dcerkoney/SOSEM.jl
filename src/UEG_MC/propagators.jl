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

# NOTE: The current conventions for Parquet.vertex3 force us to remap external times
#       in order to measure different observables & partitions simultaneously.
#       Depending on whether an observable involves the left/right discontinuity side,
#       the outgoing external time nt - 1 / nt is mapped to 2/3, respectively.
#       Argument `Tout_index` should be either 2 or 3.
@inline function remap_extT(taupair::Tuple{Int,Int}, extT_out::Int, extT_out_index::Int)
    t1, t2 = taupair[1], taupair[2]
    # Swap t1
    if t1 == extT_out_index
        t1 = extT_out
    elseif t1 == extT_out
        t1 = extT_out_index
    end
    # Swap t2
    if t2 == extT_out_index
        t2 = extT_out
    elseif t2 == extT_out
        t2 = extT_out_index
    end
    return (t1, t2)
end

# Unscreened Coulomb interaction (for outer V lines of non-local SOSEM)
@inline function CoulombBareinstant(q, p::ParaMC)
    # Test fictitious Yukawa screening for bare interactions (mass2 = 1e-6)
    # return KOinstant(q, p.e0, p.dim, 1e-6, 0.0, p.kF)
    return KOinstant(q, p.e0, p.dim, 0.0, 0.0, p.kF)
end

"""Evaluate a bare Green's function line."""
function eval(id::BareGreenId, K, _, varT, additional::Tuple{ParaMC,W,Int}) where {W}
    p, extT, extT_out_index = additional
    β, me, μ, massratio = p.β, p.me, p.μ, p.massratio

    # HACK: Swap outgoing times into correct indices (cost: ~ 1 ns): 
    #        • extT[2] ↦ 2, 2 ↦ extT[2]
    #        • extT[3] ↦ 3, 3 ↦ extT[3]
    remapped_taupair = remap_extT(id.extT, extT[2], extT_out_index)
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
    green = 0.0
    order = id.order[1]
    if order == 0
        if τ == 0.0
            green = s * Spectral.kernelFermiT(-1e-8, ϵ, β)
        else
            green = s * Spectral.kernelFermiT(τ, ϵ, β)
        end
    elseif order == 1
        green = -s * Spectral.kernelFermiT_dω(τ, ϵ, β)
    elseif order == 2
        green = s * Spectral.kernelFermiT_dω2(τ, ϵ, β)
    elseif order == 3
        green = -s * Spectral.kernelFermiT_dω3(τ, ϵ, β)
    else
        @todo
    end
    # We have an overall sign difference relative to the Negle & Orland convention
    return -green
end

"""Evaluate a statically screened Coulomb interaction line."""
function eval(id::BareInteractionId, K, _, varT, additional::Tuple{ParaMC,W,Int}) where {W}
    # additional = (mcparam, extT, Tout_index)
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
