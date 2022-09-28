"""
Construct a DiagramTree for a non-local second-order moment (SOSEM) diagram derived from
Σ₂[G, V, Γⁱ₃ = Γ₀] (or Σ₂ itself), re-expanded to O(Vⁿ) in a statically-screened interaction V[λ].
"""
function build_nonlocal_with_ct(s::Settings)
    DiagTree.uidreset()

    # TODO: reexpress bare interaction lines via Parquet.vertex4 to allow for
    #       optional re-expansion via the statically screened interaction
    if s.expand_bare_interactions
        @todo
    end

    # Initialize DiagGen configuration containing diagram parameters propagator
    # and vertex momentum/time data, (expansion) order info, etc.
    cfg = Config(s)

    # The number of (possibly indistinct) diagram trees to generate is: 
    # ((n_expandable n_expand)) = binomial(n_expandable + n_expand - 1, n_expand)
    n_trees_naive = binomial(cfg.n_expandable + cfg.n_expand - 1, cfg.n_expand)
    vprintln(
        s,
        info,
        "Generating (($cfg.n_expandable $(cfg.n_expand))) = $n_trees_naive expanded trees...",
    )
    # Generate a list of all expanded diagrams at fixed order n
    tree_count = 0
    som_diags = Vector{DiagramF64}()
    for expansion_orders in weak_integer_compositions(cfg.n_expand, cfg.n_expandable)
        # Filter out invalid expansions due to dashed lines and/or subdiagram properties
        if is_invalid_expansion_with_ct(cfg, expansion_orders)
            continue
        end
        # Build the subdiagram corresponding to the current expansion orders
        this_diag, tree_count = build_subdiagram_with_ct(cfg, expansion_orders, tree_count)
        push!(som_diags, this_diag)
        tree_count += 1
    end
    vprintln(s, info, "Done! (discarded $(n_trees_naive - tree_count) invalid expansions)")

    # Now construct the self-energy diagram tree
    diagram_tree = DiagramF64(getID(cfg.params), Sum(), som_diags; name=s.name)
    print_tree(diagram_tree)

    # Compile to expression tree
    expression_tree = ExprTree.build([diagram_tree])

    return diagram_tree, expression_tree
end

"""Check if a (weak) composition of expansion orders is valid for a given observable/settings."""
function is_invalid_expansion_with_ct(cfg::Config, expansion_orders::Vector{Int})
    # We can't spend any orders on dashed line(s), if any exist
    is_invalid =
        !isempty(cfg.G.dash_indices) && any(expansion_orders[cfg.G.dash_indices] .!= 0)
    if cfg.has_gamma3
        # Beyond bare order we must spend at least one order on a 3-point vertex insertion
        is_invalid = is_invalid || expansion_orders[cfg.Gamma3.index] == 0
    end
    return is_invalid
end

"""Construct a second-order self-energy (moment) subdiagram"""
function build_subdiagram_with_ct(cfg::Config, expansion_orders::Vector{Int}, tree_count)
    subdiag =
        cfg.has_gamma3 ? build_subdiagram_gamma_with_ct : build_subdiagram_gamma0_with_ct
    return subdiag(cfg, expansion_orders; tree_count)
end

"""
Construct a second-order self-energy (moment) subdiagram with bare Γⁱ₃ insertion:

D = (G₁ ∘ G₂ ∘ G₃ ∘ V₁ ∘ V₂)
"""
function build_subdiagram_gamma0_with_ct(
    cfg::Config,
    expansion_orders::Vector{Int};
    tree_count,
)
    # The firstTauIdx for each G line depends on the expansion order of the previous G.
    # NOTE: the default offset for Green's functions is FeynmanDiagram.firstTauIdx(GreenDiag) = 3
    g_ftis, g_max_fti = Parquet.findFirstTauIdx(
        expansion_orders,
        repeat([GreenDiag], cfg.n_g),
        FeynmanDiagram.firstTauIdx(GreenDiag),
        1,
    )

    # First loop indices for each Green's function
    # NOTE: the default offset for Green's functions is FeynmanDiagram.firstLoopIdx(GreenDiag) = 2
    # g_flis, g_max_fli = Parquet.findFirstLoopIdx(expansion_orders, first_fli)
    g_flis, g_max_fli = Parquet.findFirstLoopIdx(
        expansion_orders,
        FeynmanDiagram.firstLoopIdx(GreenDiag, cfg.n_v), # offset = n_v = 2 due to two bare interactions
    )

    @debug """
    \nTree #$(tree_count+1):
        • Expansion orders:\t\t\t$expansion_orders
        • First tau indices for G_i's:\t\t$g_ftis (maxTauIdx = $g_max_fti)
        • First momentum loop indices for G_i's:\t$g_flis (maxLoopIdx = $g_max_fli)
    """ maxlog = 10

    # TODO: Add counterterms---for n[i] expansion order of line i, spend n_cti
    #       orders on counterterm derivatives in all possible ways (0 < n_cti < n[i]).
    #       E.g., if n[i] = 4, we have: ni_left, n_cti = weakintsplit(n[i])

    # Green's function and bare interaction params
    g_params = [
        reconstruct(
            cfg.params;
            type=GreenDiag,
            innerLoopNum=expansion_orders[i],
            firstLoopIdx=g_flis[i],
            firstTauIdx=g_ftis[i],
        ) for i in 1:(cfg.n_g)
    ]
    # v_ftis = [taupair[1] for taupair in cfg.V.taus]
    v_params = [
        reconstruct(
            cfg.params;
            type=Ver4Diag,
            innerLoopNum=0,
            firstLoopIdx=1, # =0
            firstTauIdx=cfg.V.taus[i][1],
        ) for i in 1:(cfg.n_v)
    ]

    # Re-expanded Green's function and bare interaction lines
    g_lines = [
        Parquet.green(g_params[i], cfg.G.ks[i], cfg.G.taus[i]; name=cfg.G.names[i]) for
        i in 1:(cfg.n_g)
    ]

    # We mark the outer two bare interactions as fixed via `order[end] = -1`
    v_ids = [
        BareInteractionId(
            v_params[i],
            ChargeCharge,
            Instant,
            [0, 0, 0, -1];
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
function build_subdiagram_gamma_with_ct(
    cfg::Config,
    expansion_orders::Vector{Int};
    tree_count,
)
    # The firstTauIdx for each G line depends on the expansion order of the previous G.
    # NOTE: the default offset for Green's functions is FeynmanDiagram.firstTauIdx(GreenDiag) = 3
    g_ftis, max_ti = Parquet.findFirstTauIdx(
        expansion_orders,
        [Ver3Diag; repeat([GreenDiag], cfg.n_g)],
        1,
        1,
    )

    # First loop indices for each Green's function
    # NOTE: the default offset for Green's functions is FeynmanDiagram.firstLoopIdx(GreenDiag) = 2
    first_fli = FeynmanDiagram.firstLoopIdx(GreenDiag, cfg.n_v)
    @assert first_fli == 4
    g_flis, max_li = Parquet.findFirstLoopIdx(expansion_orders, first_fli)

    # First, we compute the product (Γⁱ₃ ∘ Gᵢ), i = 2 (3) for a SOSEM with left (right) Γⁱ₃ insertion.
    # Hence, the first items in the loop/time index lists correspond to those of Γⁱ₃.
    gamma3_fti = popfirst!(g_ftis)
    gamma3_fli = popfirst!(g_flis)

    @debug """
    \nTree #$(tree_count+1):
        • Expansion orders:\t\t\t$expansion_orders
        • G expansion orders:\t\t\t$(expansion_orders[1:(cfg.n_g)])
        • First tau indices for G_i's:\t\t$g_ftis (maxTauIdx = $max_ti)
        • First momentum loop indices for G_i's:\t$g_flis (maxLoopIdx = $max_li)
        • Gamma_3 expansion order:\t\t$(expansion_orders[cfg.Gamma3.index])
        • First tau index for Gamma_3:\t\t$gamma3_fti
        • First momentum loop index for Gamma_3:\t$gamma3_fli
    """ maxlog = 10

    # TODO: Add counterterms---for n[i] expansion order of line i, spend n_cti
    #       orders on counterterm derivatives in all possible ways (0 < n_cti < n[i]).
    #       E.g., if n[i] = 4, we have: ni_left, n_cti = weakintsplit(n[i])

    # Green's function and bare interaction params
    g_params = [
        reconstruct(
            cfg.params;
            type=GreenDiag,
            innerLoopNum=expansion_orders[i],
            firstLoopIdx=g_flis[i],
            firstTauIdx=g_ftis[i],
        ) for i in 1:(cfg.n_g)
    ]
    v_params = [
        reconstruct(
            cfg.params;
            type=Ver4Diag,
            innerLoopNum=0,
            firstLoopIdx=1,
            firstTauIdx=cfg.V.taus[i][1],
        ) for i in 1:(cfg.n_v)
    ]
    gamma3_params = reconstruct(
        cfg.params;
        type=Ver3Diag,
        innerLoopNum=expansion_orders[cfg.Gamma3.index],
        firstLoopIdx=gamma3_fli,
        firstTauIdx=gamma3_fti,
    )

    # Expand 3-point vertex and group by external times (summing over internal spins)
    gamma3_df =
        mergeby(Parquet.vertex3(gamma3_params, cfg.Gamma3.ks; name=cfg.Gamma3.name), :extT)

    # Grab outgoing time of each Gamma_3 group for G_3
    taus_gamma3_out = [group.extT for group in eachrow(gamma3_df)]

    # Index of the Green's function attached to Gamma_3 at the right
    idx_g_gamma3 = (cfg.Gamma3.side == left) ? 2 : 3

    # Inner loop over variable outgoing extT from Gamma_3 and G_3 to build
    # (Γⁱ₃ ∘ Gᵢ), where i = 2 (3) for a left (right) Γⁱ₃ insertion
    gamma3_gi_diags = []
    gamma3_gi_name = Symbol(cfg.Gamma3.name, "∘", cfg.G.names[idx_g_gamma3])
    for (i, gamma3_diag) in enumerate(gamma3_df.diagram)
        gi = Parquet.green(
            g_params[idx_g_gamma3],
            cfg.G.ks[idx_g_gamma3],
            (taus_gamma3_out[i][end], cfg.G.taus[idx_g_gamma3][end]);
            name=cfg.G.names[idx_g_gamma3],
        )
        this_gamma3_gi = DiagramF64(
            GenericId(gamma3_params),
            Prod(),
            [gamma3_diag, gi];
            name=gamma3_gi_name,
        )
        push!(gamma3_gi_diags, this_gamma3_gi)
    end

    # Indices of the two Green's function lines, which remain to be built
    idx_rest = deleteat!(collect(1:(cfg.n_g)), idx_g_gamma3)

    # Construct the diagram tree for Gamma_3 * G_3
    gamma3_gi =
        DiagramF64(GenericId(gamma3_params), Sum(), gamma3_gi_diags; name=cfg.Gamma3.name)

    # Re-expanded Green's function and bare interaction lines
    g_lines_rest = [
        Parquet.green(g_params[i], cfg.G.ks[i], cfg.G.taus[i]; name=cfg.G.names[i]) for
        i in idx_rest
    ]

    # Mark the outer two bare interactions as fixed via `order[end] = -1`
    v_ids = [
        BareInteractionId(
            v_params[i],
            ChargeCharge,
            Instant,
            [0, 0, 0, -1];
            k=cfg.V.ks[i],
            t=cfg.V.taus[i],
            permu=Di,
        ) for i in 1:(cfg.n_v)
    ]
    v_lines = [DiagramF64(v_ids[i]; name=cfg.V.names[i]) for i in 1:(cfg.n_v)]

    # Adapt dash indices to the 2-element list g_lines_rest (max index is 2). Since the dash
    # is never on the Green's function line to the right of Gamma_3, we ignore that line.
    dash_indices_rest = copy(cfg.G.dash_indices)
    dash_indices_rest[dash_indices_rest .> 2] .= 2

    # Dash Green's function line(s), if applicable. 
    g_lines_rest[dash_indices_rest] .=
        DiagTree.derivative(g_lines_rest[dash_indices_rest], BareGreenId, 1; index=3)

    # Build the full subdiagram
    subdiagram = DiagramF64(cfg.generic_id, Prod(), [gamma3_gi; g_lines_rest; v_lines])
    return subdiagram, tree_count
end

"""
Generate a DiagramTree for the one-crossing Σ₂[G, V, Γⁱ₃ = Γ₀] diagram (without
dashed G-lines), re-expanded to O(Vⁿ) in a statically-screened interaction V[λ].
"""
function build_sigma2_nonlocal_with_ct(s::Settings)
    @assert isempty(s.indices_g_dash) && return build_nonlocal_with_ct(s)
end
