"""
Construct diagram and expression trees for a non-local second-order moment (SOSEM) diagram derived 
from Σ₂[G, V, Γⁱ₃ = Γ₀] (or Σ₂ itself) at O(Vⁿ) for a statically-screened interaction V[λ].
"""
function build_nonlocal_fixed_order(s::Settings)
    @assert s.min_order == s.max_order
    DiagTree.uidreset()
    # Construct the self-energy diagram tree without counterterms
    sign = _get_obs_sign(s.observable)
    diagparam, diagtree = build_diagtree(s; factor=sign)
    # @debug "\nDiagTree:\n" * repr_tree(diagtree)
    # Compile to expression tree
    exprtree = ExprTree.build([diagtree])
    return diagparam, diagtree, exprtree
end

"""
Construct diagram and expression trees for the full non-local second-order moment (SOSEM) diagram
derived from Σ₂[G, V, Γⁱ₃ = Γ₀] (or Σ₂ itself) at O(Vⁿ) for a statically-screened interaction V[λ].
"""
function build_full_nonlocal_fixed_order(s_list::Vector{Settings})
    @assert all(s.min_order == s.max_order for s in s_list)
    DiagTree.uidreset()
    # Build trees for each (atomic) observable; we can merge them all except for c1c
    diagparams = DiagParaF64[]
    diagtrees_rest = DiagramF64[]
    extTs_rest = Tuple{Int,Int}[]
    local diagtree_c1c
    for (i, s) in enumerate(s_list)
        sign = _get_obs_sign(s.observable)
        diagparam, obstree = build_diagtree(s; factor=sign * c1nl_ueg.factors[i])
        if s.observable == c1c
            push!(diagparams, diagparam)
            diagtree_c1c = obstree
        else
            push!(diagparams, diagparam)
            push!(diagtrees_rest, obstree)
            push!(extTs_rest, Config(s).extT)
        end
    end
    # The external times and diagram parameters of all 
    # observables must be equal to measure simultaneously;
    # hence, we can combine all observables but c1c, which
    # requires its own partition
    @assert alleq(diagparams)
    @assert alleq(extTs_rest)
    # Sum trees for the 3 observables with positive discontinuity side
    diagparam = diagparams[1]
    diagtree_rest = DiagramF64(getID(diagparam), Sum(), diagtrees_rest; name=c1nl_ueg.name)
    # Two diagram trees, one for each extT type
    diagtrees = [diagtree_c1c, diagtree_rest]
    # Build two-root expression trees (1) c1c (2) rest
    exprtree = ExprTree.build([diagtree_c1c, diagtree_rest])
    return diagparam, diagtrees, exprtree
end

"""
Construct diagram and expression trees for a non-local second-order moment (SOSEM) diagram derived from 
Σ₂[G, V, Γⁱ₃ = Γ₀] (or Σ₂ itself) between orders n_min and n_max in a statically-screened interaction V[λ].
"""
function build_nonlocal(s::Settings)
    DiagTree.uidreset()
    diagparams = Vector{DiagParaF64}()
    diagtrees = Vector{DiagramF64}()
    exprtrees = Vector{ExprTreeF64}()
    # Loop over all orders for the given SOSEM observable settings
    sign = _get_obs_sign(s.observable)
    for n_loop in (s.min_order):(s.max_order)
        # Build diagram tree for this loop order
        @debug "Loop order n = $n_loop:"
        diagparam, diagtree = build_diagtree(s; n_loop=n_loop, factor=sign)

        # Compile to expression tree and save results for this loop order
        exprtree = ExprTree.build([diagtree])
        push!(diagparams, diagparam)
        push!(exprtrees, exprtree)
        push!(diagtrees, diagtree)
    end
    return diagparams, diagtrees, exprtrees
end

"""
Construct a list of all expression trees for non-local second-order moment (SOSEM) diagrams derived from
Σ₂[G(μ), V(λ), Γⁱ₃ = Γ₀] (or Σ₂ itself) between orders n_min and n_max in ξ (loop + CT orders),
including counterterms in μ and/or λ.
"""
function build_nonlocal_with_ct(s::Settings; renorm_mu=false, renorm_lambda=true)
    DiagTree.uidreset()
    valid_partitions = Vector{PartitionType}()
    diagparams = Vector{DiagParaF64}()
    diagtrees = Vector{DiagramF64}()
    exprtrees = Vector{ExprTreeF64}()
    # Loop over all counterterm partitions for the given SOSEM observable settings
    sign = _get_obs_sign(s.observable)
    for p in counterterm_partitions(s; renorm_mu=renorm_mu, renorm_lambda=renorm_lambda)
        # Build diagram tree for this partition
        @debug "Partition (n_loop, n_ct_mu, n_ct_lambda): $p"
        diagparam, diagtree = build_diagtree(s; factor=sign, n_loop=p[1])

        # Build tree with counterterms (∂λ(∂μ(DT))) via automatic differentiation
        dμ_diagtree = DiagTree.derivative([diagtree], BareGreenId, p[2]; index=1)
        dλ_dμ_diagtree = DiagTree.derivative(dμ_diagtree, BareInteractionId, p[3]; index=2)
        if isempty(dλ_dμ_diagtree)
            @warn("Ignoring partition $p with no diagrams")
            continue
        end
        @debug "\nDiagTree:\n" * repr_tree(dλ_dμ_diagtree)

        # Compile to expression tree and save results for this partition
        exprtree = ExprTree.build(dλ_dμ_diagtree)
        push!(valid_partitions, p)
        push!(diagparams, diagparam)
        push!(exprtrees, exprtree)
        append!(diagtrees, dλ_dμ_diagtree)
    end
    return valid_partitions, diagparams, diagtrees, exprtrees
end

"""
Construct a list of all expression trees for the full non-local second-order moment (SOSEM)
derived from Σ₂[G(μ), V(λ), Γⁱ₃ = Γ₀] (or Σ₂ itself) between orders n_min and n_max in ξ
(loop + CT orders), including counterterms in μ and/or λ.
"""
function build_full_nonlocal_with_ct(
    s_list::Vector{Settings};
    renorm_mu=false,
    renorm_lambda=true,
)
    @assert s.min_order == s.max_order
    DiagTree.uidreset()
    valid_partitions = Vector{PartitionType}()
    diagparams = Vector{DiagParaF64}()
    diagtrees_list = Vector{Vector{DiagramF64}}()
    exprtrees = Vector{ExprTreeF64}()
    # Loop over all counterterm partitions for the given SOSEM observable settings
    for p in counterterm_partitions(s; renorm_mu=renorm_mu, renorm_lambda=renorm_lambda)
        # Build diagram trees for each (atomic) observable for this partition
        @debug "Partition (n_loop, n_ct_mu, n_ct_lambda): $p"
        # Build trees for each (atomic) observable; we can merge them all except for c1c
        partn_diagparams = DiagParaF64[]
        partn_diagtrees_rest = DiagramF64[]
        partn_extTs_rest = Vector{Int}[]
        local partn_diagtree_c1c
        for (i, s) in enumerate(s_list)
            sign = _get_obs_sign(s.observable)
            partn_diagparam, partn_obstree =
                build_diagtree(s; factor=sign * c1nl_ueg.factors[i], n_loop=p[1])
            if s.observable == c1c
                push!(partn_diagparams, partn_diagparam)
                partn_diagtree_c1c = partn_obstree
            else
                push!(partn_diagparams, partn_diagparam)
                push!(partn_diagtrees_rest, partn_obstree)
                push!(partn_extTs_rest, Config(s).extT)
            end
        end
        # The external times and diagram parameters of all 
        # observables must be equal to measure simultaneously;
        # hence, we can combine all observables but c1c, which
        # requires its own partition
        @assert alleq(partn_diagparams)
        @assert alleq(partn_extTs_rest)
        # Sum trees for the 3 observables with positive discontinuity side
        partn_diagparam = partn_diagparams[1]
        diagtree_rest = DiagramF64(
            getID(partn_diagparam),
            Sum(),
            partn_diagtrees_rest;
            name=c1nl_ueg.name,
        )
        # Diagtrees for c1c and rest for this partition
        partn_diagtrees = [diagtree_c1c, diagtree_rest]

        # Build tree with counterterms (∂λ(∂μ(DT))) via automatic differentiation
        dμ_diagtrees = DiagTree.derivative(partn_diagtrees, BareGreenId, p[2]; index=1)
        dλ_dμ_diagtrees =
            DiagTree.derivative(dμ_diagtrees, BareInteractionId, p[3]; index=2)
        if isempty(dλ_dμ_diagtrees)
            @warn("Ignoring partition $p with no diagrams")
            continue
        end
        @debug "\nDiagTree:\n" * repr_tree(dλ_dμ_diagtrees)

        # Compile to expression tree and save results for this partition
        partn_exprtree = ExprTree.build(dλ_dμ_diagtrees)
        push!(valid_partitions, p)
        push!(diagparams, diagparam)
        push!(diagtrees_list, dλ_dμ_diagtrees)
        # Build two-root expression trees (1) c1c (2) rest
        push!(exprtrees, partn_exprtree)
    end
    return diagparams, diagtrees_list, exprtrees
end

"""
Generate a diagram tree for the one-crossing Σ₂[G, V, Γⁱ₃ = Γ₀] diagram
(without dashed G-lines) to O(Vⁿ) for a statically-screened interaction V[λ].
"""
function build_sigma2_nonlocal(s::Settings)
    @assert isempty(s.indices_g_dash) && return build_nonlocal_fixed_order(s)
end

"""
Construct a diagram tree for a non-local second-order 
moment (SOSEM) observable at the given loop order.
"""
function build_diagtree(s::Settings; n_loop::Int=s.max_order, factor=1.0)
    # Initialize DiagGen configuration containing diagram parameters, partition,
    # propagator and vertex momentum/time data, (expansion) order info, etc.
    cfg = Config(s, n_loop)

    # The number of (possibly indistinct) diagram trees to generate is: 
    # ((n_expandable n_expand)) = binomial(n_expandable + n_expand - 1, n_expand)
    n_trees_naive = binomial(cfg.n_expandable + cfg.n_expand - 1, cfg.n_expand)
    vprintln(
        s,
        info,
        "Generating (($(cfg.n_expandable) $(cfg.n_expand))) = $n_trees_naive expanded subtrees...",
    )
    # Generate a list of all expanded diagrams at fixed order n
    tree_count = 0
    som_diags = Vector{DiagramF64}()
    for expansion_orders in weak_integer_compositions(cfg.n_expand, cfg.n_expandable)
        # Filter out invalid expansions due to dashed lines and/or subdiagram properties
        if _is_invalid_expansion(cfg, expansion_orders)
            continue
        end
        # Build the subdiagram corresponding to the current expansion orders
        this_diag, tree_count = _build_subdiagram(cfg, expansion_orders, tree_count)
        push!(som_diags, this_diag)
        tree_count += 1
    end
    vprintln(s, info, "Done! (discarded $(n_trees_naive - tree_count) invalid expansions)")

    # Now construct the self-energy diagram tree and parameters
    diagparam = cfg.param
    diagtree = DiagramF64(getID(diagparam), Sum(), som_diags; factor=factor, name=s.name)
    return diagparam, diagtree
end

"""Check if a (weak) composition of expansion orders is valid for a given observable/settings."""
@inline function _is_invalid_expansion(cfg::Config, expansion_orders::Vector{Int})
    # We can't spend any orders on dashed line(s), if any exist
    is_invalid =
        !isempty(cfg.G.dash_indices) && any(expansion_orders[cfg.G.dash_indices] .!= 0)
    if cfg.has_gamma3
        # We must spend at least one order on the 3-point vertex insertion
        is_invalid = is_invalid || expansion_orders[cfg.Gamma3.index] == 0
    end
    return is_invalid
end

"""Construct a second-order self-energy (moment) subdiagram"""
function _build_subdiagram(cfg::Config, expansion_orders::Vector{Int}, tree_count)
    subdiag = cfg.has_gamma3 ? _build_subdiagram_gamma : _build_subdiagram_gamma0
    return subdiag(cfg, expansion_orders; tree_count)
end

"""
Construct a second-order self-energy (moment) subdiagram with bare Γⁱ₃ insertion:

D = (G₁ ∘ G₂ ∘ G₃ ∘ V₁ ∘ V₂)
"""
function _build_subdiagram_gamma0(cfg::Config, expansion_orders::Vector{Int}; tree_count)
    # The first available tau index for G(1,n) is 2 when n_expand > 0, 
    # and FeynmanDiagram.firstTauIdx(Ver3Diag) = 3 otherwise
    first_fti = cfg.n_expand > 0 ? 2 : FeynmanDiagram.firstTauIdx(Ver3Diag)
    # The firstTauIdx for each G line depends on the expansion order of the previous G.
    g_ftis, g_max_ti = Parquet.findFirstTauIdx(
        expansion_orders,
        repeat([GreenDiag], cfg.n_g),
        first_fti,  # = 2 when n > 2, otherwise = 3
        1,
    )

    # First loop indices for each Green's function
    # NOTE: The default value for a self-energy observable is FeynmanDiagram.firstLoopIdx(SigmaDiag) = 2.
    #       Here we add an offset of n_v = 2 due to the outer two bare interactions.
    first_fli = FeynmanDiagram.firstLoopIdx(SigmaDiag, cfg.n_v)
    @assert first_fli == 4
    g_flis, g_max_li = Parquet.findFirstLoopIdx(expansion_orders, first_fli)

    @debug """
    \nSubtree #$(tree_count+1):
        • Expansion orders:\t\t\t$expansion_orders
        • First tau indices for G_i's:\t\t$g_ftis (maxTauIdx = $g_max_ti)
        • First momentum loop indices for G_i's:\t$g_flis (maxLoopIdx = $g_max_li)
    """

    # TODO: Add counterterms---for n[i] expansion order of line i, spend n_cti
    #       orders on counterterm derivatives in all possible ways (0 < n_cti < n[i]).
    #       E.g., if n[i] = 4, we have: ni_left, n_cti = weakintsplit(n[i])

    # Green's function and bare interaction params
    g_params = [
        reconstruct(
            cfg.param;
            type=GreenDiag,
            innerLoopNum=expansion_orders[i],
            firstLoopIdx=g_flis[i],
            firstTauIdx=g_ftis[i],
        ) for i in 1:(cfg.n_g)
    ]
    # v_ftis = [taupair[1] for taupair in cfg.V.taus]
    v_params = [
        reconstruct(
            cfg.param;
            type=Ver4Diag,
            innerLoopNum=0,
            firstLoopIdx=1,  # =0
            firstTauIdx=cfg.V.taus[i][1],
        ) for i in 1:(cfg.n_v)
    ]

    # Re-expanded Green's function and bare interaction lines
    g_lines = [
        Parquet.green(g_params[i], cfg.G.ks[i], cfg.G.taus[i]; name=cfg.G.names[i]) for
        i in 1:(cfg.n_g)
    ]

    # We optionally mark the outer two bare interactions as fixed via `order[end] = 1`
    v_ids = [
        BareInteractionId(
            v_params[i],
            ChargeCharge,
            Instant,
            cfg.V.orders;
            k=cfg.V.ks[i],
            t=cfg.V.taus[i],
            permu=Di,
        ) for i in 1:(cfg.n_v)
    ]
    v_lines = [DiagramF64(v_ids[i]; name=cfg.V.names[i]) for i in 1:(cfg.n_v)]

    # Dash Green's function line(s), if applicable
    g_lines[cfg.G.dash_indices] .=
        DiagTree.derivative(g_lines[cfg.G.dash_indices], BareGreenId, 1; index=3)

    # Build the full subdiagram
    subdiagram = DiagramF64(cfg.generic_id, Prod(), [g_lines; v_lines])
    return subdiagram, tree_count
end

"""
Construct a second-order self-energy (moment) subdiagram with higher-order Γⁱ₃ insertion.

For (1) left and (2) right Γⁱ₃ insertions, we have:

(1) D = (G₁ ∘ G₃ ∘ V₁ ∘ V₂) ∘ (Γⁱ₃ ∘ G₂),
(2) D = (G₁ ∘ G₂ ∘ V₁ ∘ V₂) ∘ (Γⁱ₃ ∘ G₃).
"""
function _build_subdiagram_gamma(cfg::Config, expansion_orders::Vector{Int}; tree_count)
    # The firstTauIdx for each expandable item depends on the expansion order of the previous item.
    # We must expand and group by Γⁱ₃ subdiagram first, so it comes first in the expansion order list.
    gamma3_fti = 1

    # g_ftis, max_ti = Parquet.findFirstTauIdx(
    #     expansion_orders,
    #     [Ver3Diag; repeat([GreenDiag], cfg.n_g)],
    #     FeynmanDiagram.firstTauIdx(Ver3Diag),  # = 1
    #     1,
    # )

    # First loop indices for each Green's function
    # NOTE: The default value for a self-energy observable is FeynmanDiagram.firstLoopIdx(SigmaDiag) = 2.
    #       Here we add an offset of n_v = 2 due to the outer two bare interactions.
    first_fli = FeynmanDiagram.firstLoopIdx(SigmaDiag, cfg.n_v)  # = 4
    @assert first_fli == 4  # k1, k2, k3 are already present in loop basis
    # Expansion orders starting with Gamma_3
    expansion_reorders = [expansion_orders[end]; expansion_orders[1:(end - 1)]]
    g_flis, max_li = Parquet.findFirstLoopIdx(expansion_reorders, first_fli)

    # First, we compute the product (Γⁱ₃ ∘ Gᵢ), i = 2 (3) for a SOSEM with left (right) Γⁱ₃ insertion.
    # Hence, the first items in the loop/time index lists correspond to those of Γⁱ₃.
    gamma3_fli = popfirst!(g_flis)
    # gamma3_fti = popfirst!(g_ftis)

    @debug """
    \nSubtree #$(tree_count+1):
        • Expansion orders (Gamma last):\t\t\t$expansion_orders
        • G expansion orders:\t\t\t$(expansion_orders[1:(cfg.n_g)])
        • First momentum loop indices for G_i's:\t$g_flis (maxLoopIdx = $max_li)
        • Gamma_3 expansion order:\t\t$(expansion_orders[cfg.Gamma3.index])
        • First tau index for Gamma_3:\t\t$gamma3_fti
        • First momentum loop index for Gamma_3:\t$gamma3_fli
    """

    # TODO: Add counterterms---for n[i] expansion order of line i, spend n_cti
    #       orders on counterterm derivatives in all possible ways (0 < n_cti < n[i]).
    #       E.g., if n[i] = 4, we have: ni_left, n_cti = weakintsplit(n[i])

    # Bare vertex function params
    gamma3_param = reconstruct(
        cfg.param;
        type=Ver3Diag,
        innerLoopNum=expansion_orders[cfg.Gamma3.index],
        firstLoopIdx=gamma3_fli,
        firstTauIdx=gamma3_fti,
    )

    # Expand 3-point vertex and group by external times (summing over internal spins)
    gamma3_df =
        mergeby(Parquet.vertex3(gamma3_param, cfg.Gamma3.ks; name=cfg.Gamma3.name), :extT)

    # Grab outgoing time of each Gamma_3 group for Gᵢ
    taus_gamma3_out = [group.extT for group in eachrow(gamma3_df)]

    # Index of the Green's function attached to Gamma_3 at the right
    idx_g_gamma3 = (cfg.Gamma3.side == left) ? 2 : 3

    # Inner loop over variable outgoing extT from Gamma_3 and Gᵢ to build
    # (Γⁱ₃ ∘ Gᵢ), where i = 2 (3) for a left (right) Γⁱ₃ insertion
    g_ftis_rest = []
    gamma3_gi_diags = []
    gamma3_gi_name = Symbol(cfg.Gamma3.name, "∘", cfg.G.names[idx_g_gamma3])
    for (i, gamma3_diag) in enumerate(gamma3_df.diagram)
        # Build Green function attached to Gamma3
        delta_tau_gi = innerTauNum(Ver3Diag, expansion_orders[idx_g_gamma3], 1)
        first_Tidx_gi = delta_tau_gi + 1
        gi_param = reconstruct(
            cfg.param;
            type=GreenDiag,
            innerLoopNum=expansion_orders[idx_g_gamma3],
            firstLoopIdx=g_flis[1],
            firstTauIdx=first_Tidx_gi,
        )
        gi = Parquet.green(
            gi_param,
            cfg.G.ks[idx_g_gamma3],
            (taus_gamma3_out[i][end], cfg.G.taus[idx_g_gamma3][end]);
            name=cfg.G.names[idx_g_gamma3],
        )
        # Build product diagram
        this_gamma3_gi = DiagramF64(
            GenericId(gamma3_param),
            Prod(),
            [gamma3_diag, gi];
            name=gamma3_gi_name,
        )
        push!(gamma3_gi_diags, this_gamma3_gi)
    end

    # Construct the diagram tree for Gamma_3 * Gᵢ
    gamma3_gi =
        DiagramF64(GenericId(gamma3_param), Sum(), gamma3_gi_diags; name=gamma3_gi_name)
    plot_tree(gamma3_gi; maxdepth=10)

    # Number of inner times spent on Gᵢ
    delta_tau_gi = innerTauNum(GreenDiag, expansion_orders[idx_g_gamma3], 1)

    # Indices of the two Green's function lines, which remain to be built
    idx_rest = deleteat!(collect(1:(cfg.n_g)), idx_g_gamma3)

    # Find the first time indices of the remaining Green's functions
    first_Tidx_rest = taus_gamma3_out[i][end] + delta_tau_gi + 1
    g_ftis_rest, max_ti = Parquet.findFirstTauIdx(
        expansion_orders[idx_rest],
        repeat([GreenDiag], cfg.n_g - 1),
        first_Tidx_rest,
        1,
    )
    @debug """
        • First tau indices for remaining G_i's:\t\t$g_ftis_rest (maxTauIdx = $max_ti)
    """

    # Params for remaining Green's functions
    g_params_rest = [
        reconstruct(
            cfg.param;
            type=GreenDiag,
            innerLoopNum=expansion_orders[idx_rest[i]],
            firstLoopIdx=g_flis[idx_rest[i]],
            firstTauIdx=g_ftis_rest[i],
        ) for i in eachindex(idx_rest)
    ]

    # Re-expanded Green's function and bare interaction lines
    g_lines_rest = [
        Parquet.green(
            g_params_rest[i],
            cfg.G.ks[idx_rest[i]],
            cfg.G.taus[idx_rest[i]];
            name=cfg.G.names[idx_rest[i]],
        ) for i in eachindex(idx_rest)
    ]

    # Adapt dash indices to the 2-element list g_lines_rest (max index is 2). Since the dash
    # is never on the Green's function line to the right of Gamma_3, we ignore that line.
    dash_indices_rest = copy(cfg.G.dash_indices)
    dash_indices_rest[dash_indices_rest .> 2] .= 2

    # Dash Green's function line(s), if applicable. 
    g_lines_rest[dash_indices_rest] .=
        DiagTree.derivative(g_lines_rest[dash_indices_rest], BareGreenId, 1; index=3)

    # Bare interaction params
    v_params = [
        reconstruct(
            cfg.param;
            type=Ver4Diag,
            innerLoopNum=0,
            firstLoopIdx=1,
            firstTauIdx=cfg.V.taus[i][1],
        ) for i in 1:(cfg.n_v)
    ]
    # Mark the outer two bare interactions as fixed via `order[end] = 1`
    v_ids = [
        BareInteractionId(
            v_params[i],
            ChargeCharge,
            Instant,
            cfg.V.orders;
            k=cfg.V.ks[i],
            t=cfg.V.taus[i],
            permu=Di,
        ) for i in 1:(cfg.n_v)
    ]
    v_lines = [DiagramF64(v_ids[i]; name=cfg.V.names[i]) for i in 1:(cfg.n_v)]

    # Build the full subdiagram
    subdiagram = DiagramF64(cfg.generic_id, Prod(), [gamma3_gi; g_lines_rest; v_lines])
    return subdiagram, tree_count
end
