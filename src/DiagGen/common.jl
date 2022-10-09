using FeynmanDiagram
using Parameters

macro todo()
    return :(error("Not yet implemented!"))
end

"""Verbosity level for printing information to stdout."""
@enum Verbosity begin
    quiet    # = 0
    info     # = 1
    verbose  # = 2
end

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
    c1bR0    #                           C⁽¹⁾ⁿˡ 
    c1bL     #                                 = C⁽¹ᵇ⁾ᴸ
    c1bR     #                                 + C⁽¹ᵇ⁾ᴿ
    c1c      #                                 + C⁽¹ᶜ⁾
    c1d      #                                 + C⁽¹ᵈ⁾
end
const bare_observable_to_exact_k0 = Dict(
    c1a => Inf,  # The bare local moment is divergent
    c1bL0 => (1 / 4 - pi^2 / 16),
    c1bR0 => (1 / 4 - pi^2 / 16),
    c1c => -1,
    c1d => pi^2 / 8,
)
const observable_to_dash_indices = Dict(
    sigma20 => Int[],
    sigma2 => Int[],
    c1a => [3],   # local SOSEM class has only one dash configuration
    c1bL0 => [1],
    c1bR0 => [3],
    c1bL => [1],
    c1bR => [3],  # crossing-symmetric (non-local) counterpart to c1a => same dash index (3) 
    c1c => [2],
    c1d => [1, 3],
)
const observable_to_discont_side = Dict(
    sigma20 => both,
    sigma2 => both,
    c1a => positive,
    c1bL0 => positive,
    c1bR0 => positive,
    c1bL => positive,
    c1bR => positive,
    c1c => negative,
    c1d => positive,
)
const observable_to_obs_sign = Dict(
    sigma20 => 0,  # Direct measurement not yet implemented
    sigma2 => 0,   # Direct measurement not yet implemented
    c1a => 1,
    c1bL0 => 1,
    c1bR0 => 1,
    c1bL => 1,
    c1bR => 1,
    c1c => -1,
    c1d => 1,      # Since (Θ₋₁(τ))² = Θ(-τ)
)
const observable_to_string = Dict(
    sigma20 => "Σ₂[G, V, Γⁱ₃ = Γ₀]",
    sigma2 => "Σ₂[G, V, Γⁱ₃ > Γ₀]",
    c1a => "C⁽¹ᵃ⁾[G, V, Γⁱ₃ ≥ Γ₀]",
    c1bL0 => "C⁽¹ᵇ⁾ᴸ[G, V, Γⁱ₃ = Γ₀]",
    c1bR0 => "C⁽¹ᵇ⁾ᴿ[G, V, Γⁱ₃ = Γ₀]",
    c1bL => "C⁽¹ᵇ⁾ᴸ[G, V, Γⁱ₃ > Γ₀]",
    c1bR => "C⁽¹ᵇ⁾ᴿ[G, V, Γⁱ₃ > Γ₀]",
    c1c => "C⁽¹ᶜ⁾[G, V, Γⁱ₃ = Γ₀]",
    c1d => "C⁽¹ᵈ⁾[G, V, Γⁱ₃ = Γ₀]",
)
const bare_observable_to_string = Dict(
    c1a => "C₂⁽¹ᵃ⁾",  # Divergent, but we provide a string representation anyway
    c1bL0 => "C₂⁽¹ᵇ⁾ᴸ",
    c1bR0 => "C₂⁽¹ᵇ⁾ᴿ",
    c1c => "C₂⁽¹ᶜ⁾",
    c1d => "C₂⁽¹ᵈ⁾",
)
"""Overload print operator for string representations of observables."""
Base.print(io::IO, obs::Observables) = print(io, observable_to_string[obs])

"""Print the string representation of a (non-local) bare observable."""
function bare_string(obs::DiagGen.Observables)
    @assert obs in [c1a, c1bL0, c1bR0, c1c, c1d]
    return bare_observable_to_string[obs]
end

"""
Returns the exact value of a specified low-order SOSEM observable to O(V²) at k = 0.
"""
@inline function get_exact_k0(observable::DiagGen.Observables)
    return bare_observable_to_exact_k0[observable]
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

"""Returns the indices for dashed Green's function line(s), if any, for the given observable."""
# @inline function _get_gamma3_index(observable::Observables)
#     @assert _has_gamma3(observable)
#     if observable == c1bL
#         return 1
#     else # observable == c1bR
#         return 
# end

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

"""Settings for diagram generation of Σ₂[G, V, Γⁱ₃] and derived second-order moments."""
@with_kw struct Settings
    observable::Observables = sigma20
    n_order::Int = 2             # Total order (n_v + loop + CTs)
    n_expand::Int = n_order - 2  # Expansion order ξ (loop + CTs)
    verbosity::Verbosity = quiet
    expand_bare_interactions::Bool = false
    name::Symbol = Symbol(string(observable))  # Derive SOSEM name from observable
end

"""Call print on args when above the verbosity threshold v."""
function vprint(s::Settings, v::Verbosity, args...)
    return s.verbosity >= v ? print(args...) : function (args...) end
end

"""Call println on args when above the verbosity threshold v."""
function vprintln(s::Settings, v::Verbosity, args...)
    return s.verbosity >= v ? println(args...) : function (args...) end
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
function _getparam(n_loop_tot::Int)
    # Instantaneous bare interaction (interactionTauNum = 1) 
    # => innerLoopNum = totalTauNum = n_loop_tot (total loop number)
    return DiagParaF64(;
        type=SigmaDiag,
        hasTau=true,
        firstTauIdx=1,
        innerLoopNum=n_loop_tot,
        totalTauNum=n_loop_tot,
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],
    )
end

"""Construct diagram parameters for bare propagators."""
function propagator_param(type, n_loop, firstTauIdx, firstLoopIdx, filter=[NoHartree])
    # The bare interaction is instantaneous (interactionTauNum = 1),
    # so innerLoopNum = totalTauNum = n_loop (inner loop number)
    return DiagParaF64(;
        type=type,
        hasTau=true,
        innerLoopNum=n_loop,
        firstTauIdx=firstTauIdx,
        firstLoopIdx=firstLoopIdx,
        totalLoopNum=n_loop + 3,  # = n_loop_tot + 1
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],
        filter=filter,
    )
end

@with_kw struct GData
    names::Tuple{Symbol,Symbol,Symbol}
    # Incoming times τᵢₙ for Gᵢ (i=2,3) are generally unspecifiable (depend on Γⁱ₃
    # subdiagram and whether the Γⁱ₃ insertion is to the left or the right of G₂)
    taus::Tuple{ProprTauType,ProprOptTauType,ProprOptTauType}
    ks::Tuple{VFloat64,VFloat64,VFloat64}
    indices::Vector{Int}  # Expansion order index conventions
    dash_indices::Vector{Int}
end

@with_kw struct VData
    names::Tuple{Symbol,Symbol}
    taus::Tuple{ProprTauType,ProprTauType}
    ks::Tuple{VFloat64,VFloat64}
    orders::Vector{Int}  # (before differentiations)
end

@with_kw struct Gamma3Data
    name::Symbol
    side::Gamma3InsertionSide     # Is the Γⁱ₃ insertion to the left/right?
    taus::Gamma3OptTauType        # τₒᵤₜ is generally unspecifiable (depends on subdiagram)
    ks::Tuple{VFloat64,VFloat64}  # Leg (bosonic and incoming fermionic) momenta
    index::Int                    # Expansion order index convention
end

"""
Bundles names and external/internal variables for Σ₂[G, V, Γⁱ₃]
(before reexpansion in powers of V[λ]).
"""
@with_kw struct Config
    # Diagram parameters for the SOSEM observable
    param::DiagParaF64
    # Expansion order info
    n_order::Int  # Total order (n_v + loop + CTs)
    n_expand::Int # Expansion order ξ (loop + CTs)
    n_loop::Int   # Inner loop order
    n_expandable::Int
    # There are 3 G lines and 2 outer V lines in every SOSEM observable
    n_g::Int = 3
    n_v::Int = 2
    # Data (names, variables, and indices) for G and V lines
    G::GData
    V::VData
    # Optional data for Γⁱ₃ if this SOSEM diagram contains a vertex insertion
    has_gamma3::Bool = false
    Gamma3::Union{Nothing,Gamma3Data} = nothing
    # Discontinuity side and overall sign
    discont_side::DiscontSide
    obs_sign::Int
    # Sign of the outgoing external time
    extT_sign::Int
    # External time indices
    extT::Tuple{Int,Int}
    # A generic ID for intermediate diagram construction steps
    generic_id = GenericId(propagator_param(GreenDiag, 0, 1, 1))
end

"""Construct a Config struct via diagram parameters with/without Γⁱ₃ insertion."""
function Config(
    settings::Settings,
    n_loop=settings.n_expand;
    g_names=(:G₁, :G₂, :G₃),
    v_names=(:V₁, :V₂),
    gamma3_name=:Γ₃,
)
    if _has_gamma3(settings.observable)
        if n_loop ≤ 0
            throw(
                ArgumentError(
                    "n_loop > 0 required for observable " *
                    "$(settings.observable) with Γⁱ₃ insertion!",
                ),
            )
        end
        return _Config(settings, n_loop, g_names, v_names, gamma3_name)
    else
        return _Config(settings, n_loop, g_names, v_names)
    end
end

"""Construct a Config struct with trivial Γⁱ₃ insertion (Γⁱ₃ = Γ₀)."""
function _Config(settings::Settings, n_loop, g_names, v_names)
    # Expansion order info
    n_g = 3
    n_v = 2
    n_order = settings.n_order    # Total order (n_v + loop + CTs)
    n_expand = settings.n_expand  # Expansion order ξ (loop + CTs)
    n_loop_tot = n_loop + n_v     # Total loop order
    n_expandable = n_g

    # Get diagram parameters
    param = _getparam(n_loop_tot)

    # Total size of the SOSEM loop basis dimension (= n_loop + n_v + 1)
    nk = param.totalLoopNum
    # Biggest tau index (= n_loop + n_v)
    nt = param.totalTauNum

    # Basis momenta for loops of Σ₂
    k = DiagTree.getK(nk, 1)
    k1 = DiagTree.getK(nk, 2)
    k3 = DiagTree.getK(nk, 3)
    # Derived momenta
    k2 = k1 + k3 - k
    q1 = k1 - k
    q2 = k3 - k
    # Propagator momenta/times
    g_ks = (k1, k2, k3)
    v_ks = (q1, q2)
    g_taus = ((1, nt), (nt, 1), (1, nt))
    v_taus = ((1, 1), (nt, nt))
    # External times
    extT = (1, nt)

    # Optionally mark the two V lines as unscreened
    v_orders = [0, 0, 0, 0]
    if !settings.expand_bare_interactions
        v_orders[end] = 1
    end

    # Indices of (dashed) G lines in the expansion order list
    indices_g = collect(1:n_g)
    indices_g_dash = _get_dash_indices(settings.observable)

    # Discontinuity side and sign for this observable
    discont_side = _get_discont_side(settings.observable)
    obs_sign = _get_obs_sign(settings.observable)
    extT_sign = _get_extT_sign(discont_side)

    # Bundle data for each line/vertex object
    g_data = GData(g_names, g_taus, g_ks, indices_g, indices_g_dash)
    v_data = VData(v_names, v_taus, v_ks, v_orders)

    # Config struct for low-order case
    return Config(;
        param=param,
        n_order=n_order,
        n_expand=n_expand,
        n_loop=n_loop,
        n_expandable=n_expandable,
        G=g_data,
        V=v_data,
        discont_side=discont_side,
        obs_sign=obs_sign,
        extT_sign=extT_sign,
        extT=extT,
    )
end

"""Construct a Config struct with nontrivial Γⁱ₃ insertion (Γⁱ₃ > Γ₀)."""
function _Config(settings::Settings, n_loop, g_names, v_names, gamma3_name)
    # Expansion order info
    n_g = 3
    n_v = 2
    n_order = settings.n_order    # Total order (n_v + loop + CTs)
    n_expand = settings.n_expand  # Expansion order ξ (loop + CTs)
    n_loop_tot = n_loop + n_v     # Total loop order
    n_expandable = n_g + 1        # ( = n_g + n_gamma3)

    # Get diagram parameters
    param = _getparam(n_loop_tot)

    # Total size of the SOSEM loop basis dimension (= n_order + 1)
    nk = param.totalLoopNum
    # Biggest tau index
    nt = param.totalTauNum

    # Basis momenta for loops of Σ₂
    k = DiagTree.getK(nk, 1)
    k1 = DiagTree.getK(nk, 2)
    k3 = DiagTree.getK(nk, 3)
    # Derived momenta
    k2 = k1 + k3 - k
    q1 = k1 - k
    q2 = k3 - k
    # Propagator momenta
    g_ks = (k1, k2, k3)
    v_ks = (q1, q2)

    # Incoming times τᵢₙ for Gᵢ (i=2,3) and outgoing fermionic time τₒᵤₜ for Γⁱ₃ are not initially
    # specifiable; they depend on whether the Γⁱ₃ insertion is to the left or the right of G₂.
    # Due to the convention of Parquet.vertex3, the bosonic and incoming fermionic times for
    # Γⁱ₃ and outgoing external time of Σ₂ must depend on the Γⁱ₃ insertion side.
    local v_taus, g_taus, gamma3_ks, extT
    gamma3_side = _get_insertion_side(settings.observable)
    if gamma3_side == right
        v_taus = ((1, 1), (nt, nt))
        g_taus = ((1, nt), (nt, 2), (nothing, nt))
        gamma3_ks = (-q1, k2)
        extT = (1, nt)
    else # gamma3_side == left
        v_taus = ((nt, nt), (1, 1))
        g_taus = ((nt, 2), (nothing, nt), (nt, 1))
        gamma3_ks = (q2, k1)
        extT = (nt, 1)
    end
    # Using the above conventions, the times for Γⁱ₃ are the same for both insertion sides
    gamma3_taus = (1, 2, nothing)

    # Optionally mark the two V lines as unscreened
    v_orders = [0, 0, 0, 0]
    if !settings.expand_bare_interactions
        v_orders[end] = 1
    end

    # Indices of (dashed) G lines in the expansion order list
    indices_g = collect(1:n_g)
    indices_g_dash = _get_dash_indices(settings.observable)
    # Γⁱ₃ is last in the expansion order list
    index_gamma3 = n_g + 1

    # Discontinuity side and sign for this observable
    discont_side = _get_discont_side(settings.observable)
    obs_sign = _get_obs_sign(settings.observable)
    extT_sign = _get_extT_sign(discont_side)

    # Bundle data for each line/vertex object
    g_data = GData(g_names, g_taus, g_ks, indices_g, indices_g_dash)
    v_data = VData(v_names, v_taus, v_ks, v_orders)
    gamma3_data = Gamma3Data(gamma3_name, gamma3_side, gamma3_taus, gamma3_ks, index_gamma3)

    # Config struct for high-order case
    return Config(;
        param=param,
        n_order=n_order,
        n_expand=n_expand,
        n_loop=n_loop,
        n_expandable=n_expandable,
        G=g_data,
        V=v_data,
        has_gamma3=true,
        Gamma3=gamma3_data,
        discont_side=discont_side,
        obs_sign=obs_sign,
        extT_sign=extT_sign,
        extT=extT,
    )
end

"""Print and/or plot a diagram tree to the given depth if verbosity is sufficiently high."""
function checktree(d::Diagram, s::Settings; plot=false, maxdepth=6)
    if s.verbosity > quiet
        print_tree(d)
    end
    if plot
        plot_tree(d; maxdepth=maxdepth)
    end
end

"""
Get all counterterm partitions (n1, n2, n3) satisfying the following constraints:

    (1) n_min <= n1 + n2 + n3 <= n_max
    (2) n1 > 0 if either n2 > 0 or n3 > 0

By convention, we interpret: n1 = n_loop, n2 = n_∂μ, n3 = n_∂λ (normal order, G order, W order).
"""
function counterterm_partitions(n_max::Int, n_min::Int=0; renorm_mu=false)
    if n_max < n_min
        return Tuple{Int,Int,Int}[]
    end
    # Include the bare partition, since the integral is non-trivial for SOSEM observables
    partitions = [(0, 0, 0); partition(n_max; renorm_mu=renorm_mu)]
    return sort([p for p in partitions if p[1] + p[2] + p[3] >= n_min])
end

"""
Get all counterterm partitions (n1, n2, n3) at fixed order n = n1 + n2 + n3.
"""
function counterterm_partitions_fixed_order(n::Int; renorm_mu=false)
    return counterterm_partitions(n, n; renorm_mu=renorm_mu)
end

"""
Partition the total expansion order `n` into loop and counterterm orders, 
n ↦ (n_loop, n_ct_mu, n_ct_lambda). 

If `renorm_mu` is false, then n_ct_mu = 0.
"""
function partition(n::Int; renorm_mu=false)
    if renorm_mu
        return UEG.partition(n)
    else
        return sort([(n1, 0, n3) for (n1, n3) in counterterm_split(n) if n1 + n3 ≤ n])
    end
end

"""
Generate weak compositions of size 2 of an integer n,
(i.e., the cycle (n, 0), (n-1, 1), ..., (0, n)), where 
(n_order, n_ct_lambda) = (i, j) and n_ct_lambda ≤ n - 1.
"""
function counterterm_split(n::Int)
    splits = []
    n1::Int = n > 0 ? 1 : 0
    n2::Int = n > 0 ? n - 1 : 0
    while n2 >= 0
        push!(splits, (n1, n2))
        n1 += 1
        n2 -= 1
    end
    return splits
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
