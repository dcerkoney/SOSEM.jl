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
    #   if (isfinite(green) == false)
    #     ABORT("Step:" << Para.Counter << ", Green is too large! Tau=" << Tau
    #                   << ", Ek=" << Ek << ", Green=" << green << ", Mom"
    #                   << ToString(Mom));
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

"""Evaluate a bare Green's function line."""
function eval(id::BareGreenId, K, siteidx, varT, p::ParaMC)
    kF, β, me, μ, massratio, extT = p.kF, p.β, p.me, p.μ, p.massratio, p.additional

    # HACK: Swap times into correct indices: extT[2] ↦ 2, 2 ↦ extT[2] (cost: ~ 1 ns)
    # remapped_taupair = remap_extT(id.extT, extT[2])
    # τin, τout = varT[remapped_taupair[1]], varT[remapped_taupair[2]]
    # @debug "Remapped propagator time indices: $(id.extT) ↦ $remapped_taupair" maxlog = 1
    τin, τout = varT[id.extT[1]], varT[id.extT[2]]
    
    k = norm(K)

    # SOSEM observables are never just Fock diagrams
    if p.isFock
        fock =
            SelfEnergy.Fock0_ZeroTemp(k, p.basic) - SelfEnergy.Fock0_ZeroTemp(kF, p.basic)
        ϵ = k^2 / (2me * massratio) - μ + fock
    else
        ϵ = k^2 / (2me * massratio) - μ
        # ϵ = kF / me * (k - kF)
    end

    ϵ = k^2 / (2me * massratio) - μ
    # ϵ = kF / me * (k - kF)
    
    # NOTE: Is this not too restrictive? Try varying the cutoff
    if k < 0.4 * kF || k > kF * 1.3
        return 0.0
    end

    # External time
    τ = τout - τin

    # Dashed line = Θ(τ)
    if id.order[3] == 1
        return (sign(τ) + 1) / 2.0
    end

    # Check for counterterms
    green = 0.0
    order = id.order[1]
    if order == 0
        if τ ≈ 0.0
            green = Spectral.kernelFermiT(-1e-8, ϵ, β)
        else
            green = Spectral.kernelFermiT(τ, ϵ, β)
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

"""Evaluate a bare Green's function line."""
function eval(id::BareInteractionId, K, siteidx, varT, p::ParaMC)
    # TODO: Implement check for bare interaction using: is_bare = (order[end] = -1)
    # eval(id::InteractionId, K, varT) = e0^2 / ϵ0 / (dot(K, K) + mass2)
    e0, ϵ0, mass2 = p.e0, p.ϵ0, p.mass2
    # dim, e0, ϵ0, mass2 = p.dim, p.e0, p.ϵ0, p.mass2
    qd = sqrt(dot(K, K))
    if id.order[2] == 0
        @assert id.type == Instant
        # return e0^2 / ϵ0 / (dot(K, K) + mass2)
        return Coulombinstant(qd, p)
        # if id.type == Instant
        #     if interactionTauNum(id.para) == 1
        #         # return e0^2 / ϵ0 / (dot(K, K) + mass2)
        #         return Coulombinstant(qd, p)
        #     elseif interactionTauNum(id.para) == 2
        #         # println(id.extT)
        #         return interactionStatic(p, qd, varT[id.extT[1]], varT[id.extT[2]])
        #     else
        #         error("not implemented!")
        #     end
        # elseif id.type == Dynamic
        #     return interactionDynamic(p, qd, varT[id.extT[1]], varT[id.extT[2]])
        # else
        #     error("not implemented!")
        # end
    else # counterterm for the interaction
        order = id.order[2]
        @assert id.type == Instant
        invK = 1.0 / (qd^2 + mass2)
        return e0^2 / ϵ0 * invK * (mass2 * invK)^order
        # if id.type == Instant
        #     if interactionTauNum(id.para) == 1
        #         if dim == 3
        #             invK = 1.0 / (qd^2 + mass2)
        #             return e0^2 / ϵ0 * invK * (mass2 * invK)^order
        #             # elseif dim == 2
        #             #     invK = 1.0 / sqrt(qd^2 + mass2)
        #             #     return e0^2 / ϵ0 * invK * (mass2 * invK)^order
        #         else
        #             error("not implemented!")
        #         end
        #     else
        #         # return counterR(qd, varT[id.extT[1]], varT[id.extT[2]], id.order[2])
        #         return 0.0 #for dynamical interaction, the counter-interaction is always dynamic!
        #     end
        # elseif id.type == Dynamic
        #     return counterR(p, qd, varT[id.extT[1]], varT[id.extT[2]], id.order[2])
        # else
        #     error("not implemented!")
        # end
    end
end

end # module Propagators