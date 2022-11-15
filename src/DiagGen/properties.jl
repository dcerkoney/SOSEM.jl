"""Signifies whether the Γⁱ₃ insertion is to the left or the right of G₂."""
@enum Gamma3InsertionSide begin
    left
    right
end

"""
Signifies whether the SOSEM observable is non-zero to 
the left or the right of the discontinuity at τ = 0.
"""
@enum DiscontSide begin
    negative
    positive
    both  # for direct self-energy measurements
end

"""Classes of observables for different second-order moment (SOSEM) measurements."""
@enum Observables begin
    sigma20  # second-order self-energy
    sigma2
    c1a      # local SOSEM class:          C⁽¹⁾ˡ = C⁽¹ᵃ⁾
    c1bL0    # non-local SOSEM classes:
    c1bR0    #                             C⁽¹⁾ⁿˡ 
    c1bL     #                                 = C⁽¹ᵇ⁾ᴸ
    c1bR     #                                 + C⁽¹ᵇ⁾ᴿ
    c1c      #                                 + C⁽¹ᶜ⁾
    c1d      #                                 + C⁽¹ᵈ⁾
end
const bare_observable_to_exact_k0 = Dict(
    c1a   => Inf,  # The bare local moment is divergent
    c1bL0 => (1 / 4 - pi^2 / 16),
    c1bR0 => (1 / 4 - pi^2 / 16),
    c1c   => -1,
    c1d   => pi^2 / 8,
)
const observable_to_lowest_loop_order = Dict(
    sigma20 => 2,
    sigma2  => 3,  # Γⁱ₃ > Γ₀ insertion ⟹ one additional loop
    c1a     => 2,
    c1bL0   => 2,
    c1bR0   => 2,
    c1bL    => 3,  # Γⁱ₃ > Γ₀ insertion ⟹ one additional loop
    c1bR    => 3,  # Γⁱ₃ > Γ₀ insertion ⟹ one additional loop
    c1c     => 2,
    c1d     => 2,
)
const observable_to_dash_indices = Dict(
    sigma20 => Int[],
    sigma2  => Int[],
    c1a     => [3],    # local SOSEM class has only one dash configuration
    c1bL0   => [1],
    c1bR0   => [3],
    c1bL    => [1],
    c1bR    => [3],    # crossing-symmetric (non-local) counterpart to c1a => same dash index (3) 
    c1c     => [2],
    c1d     => [1, 3],
)
const observable_to_discont_side = Dict(
    sigma20 => both,
    sigma2  => both,
    c1a     => positive,
    c1bL0   => positive,
    c1bR0   => positive,
    c1bL    => positive,
    c1bR    => positive,
    c1c     => negative,
    c1d     => positive,
)
const observable_to_obs_sign = Dict(
    sigma20 => 0,  # Direct measurement not yet implemented
    sigma2  => 0,  # Direct measurement not yet implemented
    c1a     => 1,
    c1bL0   => 1,
    c1bR0   => 1,
    c1bL    => 1,
    c1bR    => 1,
    c1c     => -1,
    c1d     => 1,  # Since (Θ₋₁(τ))² = Θ(-τ)
)
const observable_to_name = Dict(
    sigma20 => "sigma20",
    sigma2  => "sigma2",
    c1a     => "c1a",
    c1bL0   => "c1bL0",
    c1bR0   => "c1bR0",
    c1bL    => "c1bL",
    c1bR    => "c1bR",
    c1c     => "c1c",
    c1d     => "c1d",
)
const observable_to_string = Dict(
    sigma20 => "Σ₂[G, V, Γⁱ₃ = Γ₀]",
    sigma2  => "Σ₂[G, V, Γⁱ₃ > Γ₀]",
    c1a     => "C⁽¹ᵃ⁾[G, V, Γⁱ₃ ≥ Γ₀]",
    c1bL0   => "C⁽¹ᵇ⁾ᴸ[G, V, Γⁱ₃ = Γ₀]",
    c1bR0   => "C⁽¹ᵇ⁾ᴿ[G, V, Γⁱ₃ = Γ₀]",
    c1bL    => "C⁽¹ᵇ⁾ᴸ[G, V, Γⁱ₃ > Γ₀]",
    c1bR    => "C⁽¹ᵇ⁾ᴿ[G, V, Γⁱ₃ > Γ₀]",
    c1c     => "C⁽¹ᶜ⁾[G, V, Γⁱ₃ = Γ₀]",
    c1d     => "C⁽¹ᵈ⁾[G, V, Γⁱ₃ = Γ₀]",
)
const bare_observable_to_string = Dict(
    c1a   => "C₂⁽¹ᵃ⁾",   # Divergent, but we provide a string representation anyway
    c1bL0 => "C₂⁽¹ᵇ⁾ᴸ",
    c1bR0 => "C₂⁽¹ᵇ⁾ᴿ",
    c1c   => "C₂⁽¹ᶜ⁾",
    c1d   => "C₂⁽¹ᵈ⁾",
)
"""Overload print operator for string representations of observables."""
Base.print(io::IO, obs::Observables) = print(io, observable_to_string[obs])

"""Print the string representation of a (non-local) bare observable."""
@inline function get_observable_name(obs::DiagGen.Observables)
    return observable_to_name[obs]
end

"""Print the string representation of a (non-local) bare observable."""
@inline function get_bare_string(obs::DiagGen.Observables)
    @assert obs in [c1a, c1bL0, c1bR0, c1c, c1d]
    return bare_observable_to_string[obs]
end

"""
Returns the exact value of a specified low-order SOSEM observable to O(V²) at k = 0
(after nondimensionalization by E²_{TF})).
"""
@inline function get_exact_k0(observable::DiagGen.Observables)
    return bare_observable_to_exact_k0[observable]
end

"""
Returns the lowest valid loop order for the given SOSEM observable.
"""
@inline function _get_lowest_loop_order(observable::DiagGen.Observables)
    return observable_to_lowest_loop_order[observable]
end

"""
Returns the side of the discontinuity at τ = 0 giving 
a non-zero contribution for this SOSEM observable.
"""
@inline function _get_discont_side(observable::Observables)
    return observable_to_discont_side[observable]
end

"""
Return the sign of the outgoing external time τ for a given SOSEM observable 
(each observable contributes from one side of the discontinuity at τ = 0 only).
"""
@inline function _get_obs_sign(observable::Observables)
    # Direct self-energy measurement not yet implemented
    if observable in [sigma20, sigma2]
        @todo
    end
    return observable_to_obs_sign[observable]
end

"""
Return the sign of the outgoing external time τ for a given SOSEM observable 
(each observable contributes from one side of the discontinuity at τ = 0 only).
"""
@inline function _get_extT_sign(side::DiscontSide)
    if side == negative
        # Observable non-zero when τ = 0⁻
        return -1
    elseif side == positive
        # Observable non-zero when τ = 0⁺
        return 1
    else
        # Direct self-energy measurement: not yet implemented
        @todo
    end
end

"""Deduce whether this observable has a Γⁱ₃ insertion."""
@inline function _has_gamma3(observable::Observables)
    return observable in [c1bL, c1bR]
end

"""Returns the indices for dashed Green's function line(s), if any, for the given observable."""
@inline function _get_dash_indices(observable::Observables)
    return observable_to_dash_indices[observable]
end

"""Deduce the insertion side for observables with Γⁱ₃ insertions."""
@inline function _get_insertion_side(observable::Observables)
    # These are the only two observables with Γⁱ₃ insertions
    @assert _has_gamma3(observable)
    # If the dashed line is to the left, the Γⁱ₃ insertion is on the right side (and vice-versa)
    if observable == c1bL
        return right::Gamma3InsertionSide
    else # observable == c1bR
        return left::Gamma3InsertionSide
    end
end

"""Build the DiagramId for a second-order moment."""
function getID(param::DiagParaF64)
    return SigmaId(
        param,
        Dynamic;
        k=DiagTree.getK(param.totalLoopNum, 1),
        t=(1, param.totalTauNum),
    )
end

"""
Construct diagram parameters for a second-order self-energy moment (SOSEM).
Each SOSEM is derived from a self-energy diagram with two bare Coulomb (V)
lines Σ₂[G, V, Γⁱ₃], where Γⁱ₃ is the improper three-point vertex.
"""
function _getparam(
    n_loop_tot::Int;
    filter=[NoHartree],
    interaction=[Interaction(ChargeCharge, Instant)],
)
    # Instantaneous bare interaction (interactionTauNum = 1) 
    # => innerLoopNum = totalTauNum = n_loop_tot
    return DiagParaF64(;
        type=SigmaDiag,
        hasTau=true,
        firstTauIdx=1,
        innerLoopNum=n_loop_tot,
        totalTauNum=n_loop_tot,
        interaction=interaction,
        filter=filter,
    )
end

"""Construct diagram parameters for bare propagators."""
function propagator_param(
    type,
    n_loop_inner,
    firstTauIdx,
    firstLoopIdx;
    filter=[NoHartree],
    interaction=[Interaction(ChargeCharge, Instant)],
)
    # The bare interaction is instantaneous (interactionTauNum = 1),
    # so innerLoopNum = totalTauNum = n_loop_inner
    return DiagParaF64(;
        type=type,
        hasTau=true,
        innerLoopNum=n_loop_inner,
        firstTauIdx=firstTauIdx,
        firstLoopIdx=firstLoopIdx,
        totalLoopNum=n_loop_inner + 3,  # = n_loop_tot + 1
        interaction=interaction,
        filter=filter,
    )
end

"""
Method overwrite to avoid invalid derivatives on dashed Green's function lines. These lines are
just theta functions in τ, so they have no chemical potential renormalization counterterms.
Note that by convention, we take order[3] as the entry indicating a dashed line.
"""
function DiagTree.hasOrderHigher(diag::Diagram{W}, ::Type{ID}) where {W,ID<:PropagatorId}
    # For bare propagators, a derivative of different id vanishes,
    # and derivatives of dashed or bare Coulomb lines also vanish
    return diag.id isa ID && all(diag.id.order[[3, 4]] .== 0)
end
