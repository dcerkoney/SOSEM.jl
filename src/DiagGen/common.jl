"""Verbosity level for printing information to stdout."""
@enum Verbosity begin
    quiet    # = 0
    info     # = 1
    verbose  # = 2
end

"""Settings for diagram generation/integration of Σ₂[G, V, Γⁱ₃] and derived second-order moments."""
struct Settings{O}
    observable::O
    min_order::Int
    max_order::Int
    verbosity::Verbosity
    expand_bare_interactions::Int
    filter::Vector{Filter}
    interaction::Vector{Interaction}
    name::Symbol
end
function Settings{O}(
    obs::O;
    min_order::Int=_get_lowest_loop_order(obs),  # Minimum allowed total order ξ (loops + CTs)
    max_order::Int=_get_lowest_loop_order(obs),  # Maximum allowed total order ξ (loops + CTs)
    verbosity::Verbosity=quiet,
    expand_bare_interactions::Int=0,
    filter::Vector{Filter}=[NoHartree],
    interaction::Vector{Interaction}=[Interaction(ChargeCharge, Instant)],
    name::Symbol=Symbol(string(obs)),  # Derive SOSEM name from observable
) where {O<:ObsType}
    return Settings{O}(
        obs,
        min_order,
        max_order,
        verbosity,
        expand_bare_interactions,
        filter,
        interaction,
        name,
    )
end
function Base.isequal(a::Settings, b::Settings)
    typeof(a) != typeof(b) && return false
    for field in fieldnames(typeof(a))
        getproperty(a, field) != getproperty(b, field) && return false
    end
    return true
end
Base.:(==)(a::Settings, b::Settings) = Base.isequal(a, b)

"""Split settings for a composite observable into a list of settings for each atomic observable."""
function atomize(settings::Settings{CompositeObservable})
    return [
        reconstruct(Settings, settings; observable=o, name=Symbol(string(o))) for
        o in settings.observable.observables
    ]
end
"""
Split settings for a composite observable with observable-dependent expansion
schemes for bare interactions into a list of settings for each atomic observable.
"""
function atomize(
    settings::Settings{CompositeObservable},
    expand_bare_interactions_list::Vector{Int},
)
    return [
        reconstruct(
            Settings,
            settings;
            observable=o,
            name=Symbol(string(o)),
            expand_bare_interactions=expand_bare_interactions_list[i],
        ) for (i, o) in enumerate(settings.observable.observables)
    ]
end
# Settings(obs::CompositeObservable; kwargs...) = Settings.(obs.observables; kwargs...)

"""Call print on args when above the verbosity threshold v."""
function vprint(s::Settings, v::Verbosity, args...)
    return s.verbosity >= v ? print(args...) : function (args...) end
end

"""Call println on args when above the verbosity threshold v."""
function vprintln(s::Settings, v::Verbosity, args...)
    return s.verbosity >= v ? println(args...) : function (args...) end
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
    orders::Tuple{Vector{Int},Vector{Int}}  # (before differentiations)
end

@with_kw struct Gamma3Data
    name::Symbol
    side::Gamma3InsertionSide     # Is the Γⁱ₃ insertion to the left/right?
    taus::Gamma3OptTauType        # τₒᵤₜ is generally unspecifiable (depends on subdiagram)
    ks::Tuple{VFloat64,VFloat64}  # Leg (bosonic and incoming fermionic) momenta
    index::Int                    # Expansion order index convention
end

"""
Configuration bundling physical properties and fixed variables for a Σ₂[G, V, Γⁱ₃]-derived observable.
"""
@with_kw struct Config
    # Diagram parameters for the SOSEM observable
    param::DiagParaF64
    # There are 3 G lines and 2 outer V lines in every SOSEM observable
    n_g::Int = 3
    n_v::Int = 2
    # Expansion order info
    min_order::Int                    # Minimum allowed total order ξ (loops + CTs)
    max_order::Int                    # Maximum allowed total order ξ (loops + CTs)
    n_loop::Int                       # Loop order
    n_expand::Int = n_loop - n_v      # Expansion order for internal lines/vertices
    n_expandable::Int
    # Data (names, variables, and indices) for G and V lines
    G::GData
    V::VData
    # Optional data for Γⁱ₃ if this SOSEM diagram contains a vertex insertion
    has_gamma3::Bool = false
    Gamma3::Union{Nothing,Gamma3Data} = nothing
    # Discontinuity side and overall sign
    discont_side::DiscontSide
    obs_sign::Int
    # Sign and variable pool index of the outgoing external time
    Tout_sign::Int
    # External time indices
    extT::Tuple{Int,Int}
    # A generic ID for intermediate diagram construction steps
    generic_id::GenericId
end

"""Construct a Config struct via diagram parameters with/without Γⁱ₃ insertion."""
function Config(
    settings::Settings{Observable},
    n_loop=settings.max_order;
    g_names=(:G₁, :G₂, :G₃),
    v_names=(:V₁, :V₂),
    gamma3_name=:Γ₃,
)
    if _has_gamma3(settings.observable)
        if n_loop ≤ 2
            throw(
                ArgumentError(
                    "n_loop > 2 required for observable " *
                    "$(settings.observable) with Γⁱ₃ insertion!",
                ),
            )
        end
        return _Config(settings, n_loop, g_names, v_names, gamma3_name)
    else
        return _Config(settings, n_loop, g_names, v_names)
    end
end
function Config(settings::Settings{CompositeObservable}, kwargs...)
    return Config.(atomize(settings), kwargs...)
end

"""Construct a Config struct with trivial Γⁱ₃ insertion (Γⁱ₃ = Γ₀)."""
function _Config(settings::Settings{Observable}, n_loop, g_names, v_names)
    # Expansion order info
    n_g          = 3
    n_v          = 2
    n_expand     = n_loop - n_v  # Expansion order for internal lines/vertices
    n_expandable = n_g

    # Get diagram parameters
    param = _getparam(n_loop; filter=settings.filter, interaction=settings.interaction)

    # Total size of the SOSEM loop basis dimension (= n_loop + n_v + 1)
    nk = param.totalLoopNum
    # Biggest tau index (= n_loop + n_v, +1 for Tout switch)
    # nt = param.totalTauNum
    @assert param.totalTauNum == nk

    # Basis momenta for loops of Σ₂
    k  = DiagTree.getK(nk, 1)
    k1 = DiagTree.getK(nk, 2)
    k3 = DiagTree.getK(nk, 3)
    # Derived momenta
    k2 = k1 + k3 - k
    q1 = k1 - k
    q2 = k3 - k

    # By convention, the outgoing external time is τₒᵤₜ = 1 (2)
    # for observables with negative (positive) discontinuity side
    Tout = _get_discont_side(settings.observable) == negative ? 1 : 2

    # Propagator momenta/times
    g_ks   = (k1, k2, k3)
    v_ks   = (q1, q2)
    g_taus = ((3, Tout), (Tout, 3), (3, Tout))
    v_taus = ((3, 3), (Tout, Tout))
    # External times (τᵢₙ, τₒᵤₜ) ∈ {(3, 1), (3, 2)}
    extT = (3, Tout)

    # We fix both external interaction lines as bare
    # for all non-local observables without vertex corrections
    # (i.e., all except c1bL and c1bR)
    v_orders = ([0, 0, 0, 1], [0, 0, 0, 1])
    if settings.expand_bare_interactions > 0
        @warn(
            "No support for expanded bare interactions for SOSEM observable " *
            "$(settings.observable) without Γⁱ₃ insertion, falling back to bare V lines..."
        )
    end

    # # Start by assuming we re-expand both V_left and V_right
    # # Mark external V lines as fixed bare, where applicable
    # if settings.expand_bare_interactions == 0
    #     # Mark both external lines as fixed bare
    #     v_orders = ([0, 0, 0, 1], [0, 0, 0, 1])  # bare V_left, bare V_right
    # elseif settings.expand_bare_interactions == 1
    #     # Mark one (left) external line as fixed bare
    #     v_orders = ([0, 0, 0, 1], [0, 0, 0, 0])  # bare V_left, expanded V_right
    # end

    # Indices of (dashed) G lines in the expansion order list
    indices_g = collect(1:n_g)
    indices_g_dash = _get_dash_indices(settings.observable)

    # Discontinuity side and sign for this observable
    discont_side = _get_discont_side(settings.observable)
    obs_sign     = _get_obs_sign(settings.observable)
    Tout_sign    = _get_Tout_sign(discont_side)

    # Bundle data for each line/vertex object
    g_data = GData(g_names, g_taus, g_ks, indices_g, indices_g_dash)
    v_data = VData(v_names, v_taus, v_ks, v_orders)

    # Generic ID for intermediate diagram objects
    generic_id = GenericId(
        propagator_param(
            GreenDiag,
            0,
            1,
            1;
            filter=param.filter,
            interaction=param.interaction,
        ),
    )

    # Config struct for low-order case
    return Config(;
        param=param,
        min_order=settings.min_order,
        max_order=settings.max_order,
        n_loop=n_loop,
        n_expand=n_expand,
        n_expandable=n_expandable,
        G=g_data,
        V=v_data,
        discont_side=discont_side,
        obs_sign=obs_sign,
        Tout_sign=Tout_sign,
        extT=extT,
        generic_id=generic_id,
    )
end

"""Construct a Config struct with nontrivial Γⁱ₃ insertion (Γⁱ₃ > Γ₀)."""
function _Config(settings::Settings{Observable}, n_loop, g_names, v_names, gamma3_name)
    # Expansion order info
    n_g          = 3
    n_v          = 2
    n_expand     = n_loop - n_v     # Expansion order for internal lines/vertices
    n_expandable = n_g + 1          # ( = n_g + n_gamma3)

    @debug "Generating config for observable with Γⁱ₃ insertion (Γⁱ₃ > Γ₀)..."

    # Get diagram parameters
    param = _getparam(n_loop; filter=settings.filter, interaction=settings.interaction)

    # Total size of the SOSEM loop basis dimension (= n_loop + n_v + 1)
    nk = param.totalLoopNum
    # Biggest tau index (= n_loop + n_v, +1 for Tout switch)
    # nt = param.totalTauNum
    @assert param.totalTauNum == nk

    @debug "nk = $nk, nt = $(param.totalTauNum - 1) (plus one fake extT)"

    # Basis momenta for loops of Σ₂
    k  = DiagTree.getK(nk, 1)
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
    local v_taus, g_taus, gamma3_ks, gamma3_taus, extT
    gamma3_side = _get_insertion_side(settings.observable)

    # By convention, the outgoing external time is τₒᵤₜ = 1 (2)
    # for observables with negative (positive) discontinuity side
    Tout = _get_discont_side(settings.observable) == negative ? 1 : 2
    # Since c1c has no gamma insertions, discont side must be positive
    @assert Tout == 2

    if gamma3_side == right  # c1bL
        v_taus = ((3, 3), (Tout, Tout))
        g_taus = ((3, Tout), (Tout, 4), (nothing, Tout))
        gamma3_ks = (-q1, k2)
        gamma3_taus = (3, 4, nothing) # (τᵢₙ, τ4, (*))
        extT = (3, Tout)
    else # gamma3_side == left
        # Config for c1bR not yet implemented (requires τ-remapping after DiagGen)
        @todo
    end

    # Mark external V lines as fixed bare, where applicable
    if settings.expand_bare_interactions == 0
        # Mark both external lines as fixed bare
        v_orders = ([0, 0, 0, 1], [0, 0, 0, 1])  # bare V_left, bare V_right
    elseif settings.expand_bare_interactions == 1
        # Mark external line not connecting with gamma3 as fixed bare
        if gamma3_side == right  # c1bL
            v_orders = ([0, 0, 0, 0], [0, 0, 0, 1])  # expand V_left, bare V_right
        else
            v_orders = ([0, 0, 0, 1], [0, 0, 0, 0])  # bare V_left, expand V_right
        end
    else
        # Re-expanding both V_left and V_right in V_λ
        v_orders = ([0, 0, 0, 0], [0, 0, 0, 0])
    end

    # # Mark external V lines as fixed bare, where applicable
    # v_orders = [0, 0, 0, 0]
    # if settings.expand_bare_interactions == 0
    #     v_orders[end] = 1
    # end

    # Indices of (dashed) G lines in the expansion order list
    indices_g = collect(1:n_g)
    indices_g_dash = _get_dash_indices(settings.observable)

    # Γⁱ₃ is last in the expansion order list by convention
    index_gamma3 = n_g + 1

    # Discontinuity side and sign for this observable
    discont_side = _get_discont_side(settings.observable)
    obs_sign     = _get_obs_sign(settings.observable)
    Tout_sign    = _get_Tout_sign(discont_side)

    # Bundle data for each line/vertex object
    g_data      = GData(g_names, g_taus, g_ks, indices_g, indices_g_dash)
    v_data      = VData(v_names, v_taus, v_ks, v_orders)
    gamma3_data = Gamma3Data(gamma3_name, gamma3_side, gamma3_taus, gamma3_ks, index_gamma3)

    # Generic ID for intermediate diagram objects
    generic_id = GenericId(
        propagator_param(
            GreenDiag,
            0,
            1,
            1;
            filter=param.filter,
            interaction=param.interaction,
        ),
    )

    # Config struct for high-order case
    return Config(;
        param=param,
        min_order=settings.min_order,
        max_order=settings.max_order,
        n_loop=n_loop,
        n_expand=n_expand,
        n_expandable=n_expandable,
        G=g_data,
        V=v_data,
        has_gamma3=true,
        Gamma3=gamma3_data,
        discont_side=discont_side,
        obs_sign=obs_sign,
        Tout_sign=Tout_sign,
        extT=extT,
        generic_id=generic_id,
    )
end

"""Print and/or plot a diagram tree to the given depth if verbosity is sufficiently high."""
function checktree(d::Diagram, s::Settings; print=true, plot=false, maxdepth=6)
    if s.verbosity > quiet && print
        print_tree(d)
    end
    if plot
        plot_tree(d; maxdepth=maxdepth)
    end
end
