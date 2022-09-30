"""
Construct a DiagramTree for the local second-order moment (SOSEM) diagram derived from the Hubbard-like
Σ₂[G, V, Γⁱ₃ = Γ₀] (or Σ₂ itself), re-expanded to O(Vⁿ) in a statically-screened interaction V[λ].
"""
function build_nonlocal(s::Settings)
    DiagTree.uidreset()

    # TODO: reexpress bare interaction lines via Parquet.vertex4 to allow for
    #       optional re-expansion via the statically screened interaction
    if s.expand_bare_interactions
        @todo
    end

    # Initialize DiagGen configuration containing diagram parameters propagator
    # and vertex momentum/time data, (expansion) order info, etc.
    cfg = Config(s)

    # Partition for Dyson equation (e.g., Χ₁ = Π₁VΠ₀ + Π₀VΠ₁ + Π₀VΠ₀VΠ₀)
    @todo
end

function build_susceptibility(cfg::Config)
    # Get polarization orders of all Dyson subterms via integer
    # combinations of the order n, then multiply by V^{n-1}.
    @todo
end

function build_poln_subdiagram(cfg::Config)

    @todo

    generalized_poln = Parquet.polarization()
    charge_poln = mergeby(generalized_poln.diagram) # 2(Π↑↑ + Π↑↓)

    # Now construct the self-energy diagram tree
    diagram_tree =
        DiagramF64(getID(cfg.params), Prod(), [g_line; v_lines; charge_poln]; name=s.name)

    # Compile to expression tree
    expression_tree = ExprTree.build([diagram_tree])

    return diagram_tree, expression_tree
end

"""
Generate a DiagramTree for the Hubbard-like Σ₂[G, V, Γⁱ₃ = Γ₀] diagram (without
dashed G-lines), re-expanded to O(Vⁿ) in a statically-screened interaction V[λ].
"""
function build_sigma2_local(s::Settings)
    @assert isempty(s.indices_g_dash) && return build_local(s)
end
