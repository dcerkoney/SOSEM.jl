"""
Construct diagram and expression trees for a non-local second-order moment (SOSEM) diagram derived 
from Σ₂[G, V, Γⁱ₃ = Γ₀] (or Σ₂ itself) to O(Vⁿ) for a statically-screened interaction V[λ].
"""
function build_nonlocal(s::Settings)
    DiagTree.uidreset()
    # Construct the self-energy diagram tree without counterterms
    diagparam, diagtree = build_diagtree(s)
    @debug "\nDiagTree:\n" * repr_tree(diagtree)
    # Compile to expression tree
    exprtree = ExprTree.build([diagtree])
    return diagparam, diagtree, exprtree
end

"""
Construct a list of all expression trees for non-local second-order moment (SOSEM) diagrams derived from
Σ₂[G, V, Γⁱ₃ = Γ₀] (or Σ₂ itself) to O(ξⁿ) for a statically-screened interaction V[λ] with counterterms.

If `fixed_order` is true, generate partitions at fixed order N = `s.n_order`.
Otherwise, generate all counterterm partitions up to max order N.
"""
function build_nonlocal_with_ct(
    s::Settings;
    fixed_order=true,
    renorm_mu=false,
    renorm_lambda=true,
)
    DiagTree.uidreset()
    valid_partitions = Tuple{Int,Int,Int}[]
    diagparams = Vector{DiagParaF64}()
    diagtrees = Vector{DiagramF64}()
    exprtrees = Vector{ExprTreeF64}()

    # Either generate all counterterm partitions up to max order N
    # or generate partitions at fixed order N, where N = s.n_order.
    partitions = fixed_order ? counterterm_partitions_fixed_order : counterterm_partitions
    for p in partitions(s; renorm_mu=renorm_mu, renorm_lambda=renorm_lambda)
        # Build diagram tree for this partition
        @debug "Partition (n_loop, n_ct_mu, n_ct_lambda): $p"
        diagparam, diagtree = build_diagtree(s; n_loop=p[1])

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
Generate a diagram tree for the one-crossing Σ₂[G, V, Γⁱ₃ = Γ₀] diagram
(without dashed G-lines) to O(Vⁿ) for a statically-screened interaction V[λ].
"""
function build_sigma2_nonlocal(s::Settings)
    @assert isempty(s.indices_g_dash) && return build_nonlocal(s)
end

"""
Construct a diagram tree for a non-local second-order 
moment (SOSEM) observable at the given loop order.
"""
function build_diagtree(s::Settings; n_loop::Int=s.n_order)
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
    diagtree = DiagramF64(getID(diagparam), Sum(), som_diags; name=s.name)
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
    g_ftis, max_ti = Parquet.findFirstTauIdx(
        expansion_orders,
        [Ver3Diag; repeat([GreenDiag], cfg.n_g)],
        FeynmanDiagram.firstTauIdx(Ver3Diag),  # = 1
        1,
    )

    # First loop indices for each Green's function
    # NOTE: The default value for a self-energy observable is FeynmanDiagram.firstLoopIdx(SigmaDiag) = 2.
    #       Here we add an offset of n_v = 2 due to the outer two bare interactions.
    first_fli = FeynmanDiagram.firstLoopIdx(SigmaDiag, cfg.n_v)  # = 4
    @assert first_fli == 4
    g_flis, max_li = Parquet.findFirstLoopIdx(expansion_orders, first_fli)

    # First, we compute the product (Γⁱ₃ ∘ Gᵢ), i = 2 (3) for a SOSEM with left (right) Γⁱ₃ insertion.
    # Hence, the first items in the loop/time index lists correspond to those of Γⁱ₃.
    gamma3_fti = popfirst!(g_ftis)
    gamma3_fli = popfirst!(g_flis)

    @debug """
    \nSubtree #$(tree_count+1):
        • Expansion orders:\t\t\t$expansion_orders
        • G expansion orders:\t\t\t$(expansion_orders[1:(cfg.n_g)])
        • First tau indices for G_i's:\t\t$g_ftis (maxTauIdx = $max_ti)
        • First momentum loop indices for G_i's:\t$g_flis (maxLoopIdx = $max_li)
        • Gamma_3 expansion order:\t\t$(expansion_orders[cfg.Gamma3.index])
        • First tau index for Gamma_3:\t\t$gamma3_fti
        • First momentum loop index for Gamma_3:\t$gamma3_fli
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
    v_params = [
        reconstruct(
            cfg.param;
            type=Ver4Diag,
            innerLoopNum=0,
            firstLoopIdx=1,
            firstTauIdx=cfg.V.taus[i][1],
        ) for i in 1:(cfg.n_v)
    ]
    gamma3_params = reconstruct(
        cfg.param;
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
        DiagramF64(GenericId(gamma3_params), Sum(), gamma3_gi_diags; name=gamma3_gi_name)

    # Re-expanded Green's function and bare interaction lines
    g_lines_rest = [
        Parquet.green(g_params[i], cfg.G.ks[i], cfg.G.taus[i]; name=cfg.G.names[i]) for
        i in idx_rest
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
