"""Verbosity level for printing information to stdout."""
@enum Verbosity begin
    quiet    # = 0
    info     # = 1
    verbose  # = 2
end

"""Settings for diagram generation/integration of Σ₂[G, V, Γⁱ₃] and derived second-order moments."""
@with_kw struct Settings
    observable::Observables = sigma20
    min_order::Int = _get_lowest_loop_order(observable)  # Minimum allowed total order ξ (loops + CTs)
    max_order::Int = _get_lowest_loop_order(observable)  # Maximum allowed total order ξ (loops + CTs)
    verbosity::Verbosity = quiet
    expand_bare_interactions::Bool = false
    filter::Vector{Filter} = [NoHartree]
    interaction::Vector{Interaction} = [Interaction(ChargeCharge, Instant)]
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
    # Sign of the outgoing external time
    extT_sign::Int
    # External time indices
    extT::Tuple{Int,Int}
    # A generic ID for intermediate diagram construction steps
    generic_id::GenericId
end

"""Construct a Config struct via diagram parameters with/without Γⁱ₃ insertion."""
function Config(
    settings::Settings,
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

"""Construct a Config struct with trivial Γⁱ₃ insertion (Γⁱ₃ = Γ₀)."""
function _Config(settings::Settings, n_loop, g_names, v_names)
    # Expansion order info
    n_g          = 3
    n_v          = 2
    n_expand     = n_loop - n_v  # Expansion order for internal lines/vertices
    n_expandable = n_g

    # Get diagram parameters
    param = _getparam(n_loop; filter=settings.filter, interaction=settings.interaction)

    # Total size of the SOSEM loop basis dimension (= n_loop + n_v + 1)
    nk = param.totalLoopNum
    # Biggest tau index (= n_loop + n_v)
    nt = param.totalTauNum

    # Basis momenta for loops of Σ₂
    k  = DiagTree.getK(nk, 1)
    k1 = DiagTree.getK(nk, 2)
    k3 = DiagTree.getK(nk, 3)
    # Derived momenta
    k2 = k1 + k3 - k
    q1 = k1 - k
    q2 = k3 - k
    # Propagator momenta/times
    g_ks   = (k1, k2, k3)
    v_ks   = (q1, q2)
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
    obs_sign     = _get_obs_sign(settings.observable)
    extT_sign    = _get_extT_sign(discont_side)

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
        extT_sign=extT_sign,
        extT=extT,
        generic_id=generic_id,
    )
end

"""Construct a Config struct with nontrivial Γⁱ₃ insertion (Γⁱ₃ > Γ₀)."""
function _Config(settings::Settings, n_loop, g_names, v_names, gamma3_name)
    # Expansion order info
    n_g          = 3
    n_v          = 2
    n_expand     = n_loop - n_v     # Expansion order for internal lines/vertices
    n_expandable = n_g + 1          # ( = n_g + n_gamma3)

    # Get diagram parameters
    param = _getparam(n_loop; filter=settings.filter, interaction=settings.interaction)

    # Total size of the SOSEM loop basis dimension (= max_order + 1)
    nk = param.totalLoopNum
    # Biggest tau index
    nt = param.totalTauNum

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
    obs_sign     = _get_obs_sign(settings.observable)
    extT_sign    = _get_extT_sign(discont_side)

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
        extT_sign=extT_sign,
        extT=extT,
        generic_id=generic_id,
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
