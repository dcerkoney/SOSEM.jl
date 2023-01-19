"""
Construct a DiagramTree for the local second-order moment (SOSEM) diagram derived from the Hubbard-like
Σ₂[G, V, Γⁱ₃ = Γ₀] (or Σ₂ itself), re-expanded to O(Vⁿ) in a statically-screened interaction V[λ].
"""
function build_local(s::Settings)
    DiagTree.uidreset()

    # TODO: reexpress bare interaction lines via Parquet.vertex4 to allow for
    #       optional re-expansion via the statically screened interaction
    if s.expand_bare_interactions == false
        error("Bare interactions from EOM must be re-expanded for the local moment!")
    end

    # Initialize DiagGen configuration containing diagram parameters propagator
    # and vertex momentum/time data, (expansion) order info, etc.
    cfg = Config(s)

    # Build the local moment diagrams, which are the instantaneous part of Wtilde = VχV.
    # TODO: generalize this low-order hard-coded approach
    local diagtree
    if cfg.max_order == 2
        # χ₀ = Π₀
    elseif cfg.max_order == 3
        # χ₁ = Π₁ + Π₀VΠ₀
    elseif cfg.max_order == 4
        # χ₂ = Π₂ + Π₁VΠ₀ + Π₀VΠ₁ + Π₀VΠ₀VΠ₀
    else
        @todo
    end

    # Compile to expression tree
    exprtree = ExprTree.build([diagtree])
end

function build_poln_subdiagram(cfg::Config)
    @todo
    generalized_poln = Parquet.polarization()
    charge_poln = mergeby(generalized_poln.diagram) # 2(Π↑↑ + Π↑↓)

    # Now construct the self-energy diagram tree
    poln_id = PolarId(cfg.param, Dynamic; k=extK, t=cfg.extT)
    poln = DiagramF64(poln_id, Prod(), [g_line; v_lines; charge_poln]; name=s.name)
    return poln
end

"""
Generate a DiagramTree for the Hubbard-like Σ₂[G, V, Γⁱ₃ = Γ₀] diagram (without
dashed G-lines), re-expanded to O(Vⁿ) in a statically-screened interaction V[λ].
"""
function build_sigma2_local(s::Settings)
    @assert isempty(s.indices_g_dash) && return build_local(s)
end
