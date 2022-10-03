"""
Construct a DiagramTree for a non-local second-order moment (SOSEM) diagram derived from
Σ₂[G, V, Γⁱ₃ = Γ₀] (or Σ₂ itself), re-expanded to O(Vⁿ) in a statically-screened interaction V[λ].
"""
function build_nonlocal_with_ct(s::Settings)
    DiagTree.uidreset()

    # Initialize DiagGen configuration containing diagram parameters propagator
    # and vertex momentum/time data, (expansion) order info, etc.
    cfg = Config(s)

    # Normal number of expandables, plus two entries for counterterms on V₁ and V₂
    n_expandable_with_ct = cfg.n_expandable + cfg.n_v

    # The number of (possibly indistinct) diagram trees to generate is: 
    # ((n_expandable n_expand)) = binomial(n_expandable + n_expand - 1, n_expand)
    n_trees_naive = binomial(n_expandable_with_ct + cfg.n_expand - 1, cfg.n_expand)
    vprintln(
        s,
        info,
        "Generating (($(n_expandable_with_ct) $(cfg.n_expand))) = $n_trees_naive expanded trees...",
    )
    # Generate a list of all expanded diagrams at fixed order n
    tree_count = 0
    som_diags = Vector{DiagramF64}()
    for expansion_orders in weak_integer_compositions(cfg.n_expand, n_expandable_with_ct)
        # Filter out invalid expansions due to dashed lines and/or subdiagram properties
        if is_invalid_expansion(cfg, expansion_orders)
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

    # Compile to expression tree
    expression_tree = ExprTree.build([diagram_tree])

    return diagram_tree, expression_tree
end

"""Check if a (weak) composition of expansion orders is valid for a given observable/settings."""
function is_invalid_expansion(cfg::Config, expansion_orders::Vector{Int})
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
    # Split each expansion order into loop and interaction counterterm orders
    for (n_loops, n_cts) in counterterm_split(expansion_orders)
        @todo
    end
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
    # Split each expansion order into loop and interaction counterterm orders
    for (n_loops, n_cts) in counterterm_split(expansion_orders)
        @todo
    end
end

"""
Generate a DiagramTree for the one-crossing Σ₂[G, V, Γⁱ₃ = Γ₀] diagram (without
dashed G-lines), re-expanded to O(Vⁿ) in a statically-screened interaction V[λ].
"""
function build_sigma2_nonlocal_with_ct(s::Settings)
    @assert isempty(s.indices_g_dash) && return build_nonlocal_with_ct(s)
end
