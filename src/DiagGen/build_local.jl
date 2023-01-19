"""
Construct a DiagramTree for the local second-order moment (SOSEM) diagram derived from the Hubbard-like
Σ₂[G, V, Γⁱ₃ = Γ₀] (or Σ₂ itself), re-expanded to O(Vⁿ) in a statically-screened interaction V[λ].
"""
function build_local(s::Settings{Observable})
    DiagTree.uidreset()

    if s.expand_bare_interactions == false
        error(
            "Bare interactions from EOM must be re-expanded for the local moment series to be analytic!",
        )
    end

    # Initialize DiagGen configuration containing diagram parameters propagator
    # and vertex momentum/time data, (expansion) order info, etc.
    cfg = LocalConfig(s)

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
    return exprtree = ExprTree.build([diagtree])
end

"""
Construct a DiagramTree for the local second-order moment (SOSEM) diagram derived from the Hubbard-like
Σ₂[G, V, Γⁱ₃ = Γ₀] (or Σ₂ itself), re-expanded to O(Vⁿ) in a statically-screened interaction V[λ].
"""
function build_local_with_ct(s::Settings{Observable}; renorm_mu=true, renorm_lambda=true)
    DiagTree.uidreset()
    valid_partitions = Vector{PartitionType}()
    diagparams = Vector{DiagParaF64}()
    diagtrees = Vector{DiagramF64}()
    exprtrees = Vector{ExprTreeF64}()

    if s.expand_bare_interactions == false
        error(
            "Bare interactions from EOM must be re-expanded for the local moment series to be analytic!",
        )
    end

    # Loop over all counterterm partitions for the given SOSEM observable settings
    sign = _get_obs_sign(s.observable)
    for p in counterterm_partitions(s; renorm_mu=renorm_mu, renorm_lambda=renorm_lambda)
        # Build diagram tree for this partition
        @debug "Partition (n_loop, n_ct_mu, n_ct_lambda): $p"
        diagparam, diagtree = build_local_diagtree(s; factor=sign, n_loop=p[1])

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
Construct a diagram tree for a local second-order 
moment (SOSEM) observable at the given loop order.
"""
function build_local_diagtree(s::Settings{Observable}; n_loop::Int=s.max_order, factor=1.0)
    # Initialize DiagGen configuration containing diagram parameters, partition,
    # propagator and vertex momentum/time data, (expansion) order info, etc.
    cfg = LocalConfig(s, n_loop)
    # We assume that Gamma3 is last in the expansion order list
    cfg.has_gamma3 && @assert cfg.Gamma3.index == 4

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

        this_diag, tree_count = _build_local_subdiagram(cfg, expansion_orders, tree_count)
        push!(som_diags, this_diag)
        tree_count += 1
    end
    vprintln(s, info, "Done! (discarded $(n_trees_naive - tree_count) invalid expansions)")

    # Now construct the SOSEM diagram tree and parameters
    extK = DiagTree.getK(cfg.param.totalLoopNum, 1)
    sosem_id = SigmaId(cfg.param, Dynamic; k=extK, t=cfg.extT)
    diagtree = DiagramF64(sosem_id, Sum(), som_diags; factor=factor, name=s.name)
    # diagtree = DiagramF64(getID(diagparam), Sum(), som_diags; factor=factor, name=s.name)
    return cfg.param, diagtree
end

function _build_local_subdiagram(cfg::LocalConfig)
    @todo

    # Build the local moment diagrams for this partition, which are the instantaneous part of Wtilde = VχV.
    # TODO: generalize this low-order hard-coded approach
    local this_diag
    if cfg.max_order == 2
        # χ₀ = Π₀
    elseif cfg.max_order == 3
        # χ₁ = Π₁ + Π₀VΠ₀
    elseif cfg.max_order == 4
        # χ₂ = Π₂ + Π₁VΠ₀ + Π₀VΠ₁ + Π₀VΠ₀VΠ₀
    else
        @todo
    end

    poln_param = DiagParaF64(type = PolarDiag, innerLoopNum = n_loop, hasTau=true, filter = [Proper])
    generalized_poln = Parquet.polarization(poln_param)
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
function build_sigma2_local(s::Settings{Observable})
    @assert isempty(s.indices_g_dash) && return build_local(s)
end
