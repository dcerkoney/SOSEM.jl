"""
Construct a DiagramTree for the local second-order moment (SOSEM) diagram derived from the Hubbard-like
Σ₂[G, V, Γⁱ₃ = Γ₀] (or Σ₂ itself), re-expanded to O(Vⁿ) in a statically-screened interaction V[λ].
"""
function build_local(s::Settings)
    DiagTree.uidreset()

    # TODO: reexpress bare interaction lines via Parquet.vertex4 to allow for
    #       optional re-expansion via the statically screened interaction
    if s.expand_bare_interactions == 0
        error("Bare interactions from EOM must be re-expanded for the local moment!")
    end

    # Initialize DiagGen configuration containing diagram parameters propagator
    # and vertex momentum/time data, (expansion) order info, etc.
    cfg = Config(s)

    # Build the local moment diagrams, which are the instantaneous part of Wtilde = VχV.
    # TODO: generalize this low-order hard-coded approach
    local diagtree
    local pi0, pi1, pi2
    local chi0, chi1, chi2
    if cfg.max_order == 2
        # χ₀ = Π₀
        pi0 = build_pi_N_with_ct(0)
        # Build chi0
        chi0 = pi0
    end
    if cfg.max_order == 3
        # χ₁ = Π₁ + Π₀VΠ₀
        pi1 = build_pi_N_with_ct(1)
        # Build chi1
        chi1 = pi1
    end
    if cfg.max_order == 4
        # χ₂ = Π₂ + Π₁VΠ₀ + Π₀VΠ₁ + Π₀VΠ₀VΠ₀
        pi2 = build_pi_N_with_ct(2)
        # Build chi2
        chi2 = pi2
    end

    # Compile to expression tree
    return ExprTree.build([diagtree])
end

function build_local_with_ct(orders; renorm_mu=true, renorm_lambda=true, isFock=false)
    DiagTree.uidreset()
    valid_partitions = Vector{PartitionType}()
    diagparams = Vector{DiagParaF64}()
    diagtrees = Vector{DiagramF64}()
    exprtrees = Vector{ExprTreeF64}()

    # Build chi partitions at the given orders (lowest order (χ₀ = Π₀) is N = 1)
    partitions = DiagGen.counterterm_partitions(
        orders;
        n_lowest=1,
        renorm_mu=renorm_mu,
        renorm_lambda=renorm_lambda,
    )

    @debug "Partitions: $partitions"
    for p in partitions
        # Build diagram tree for this partition
        @debug "Partition (n_loop, n_ct_mu, n_ct_lambda): $p"
        diagparam, diagtree = build_poln(; n_loop=p[1])
        # Build tree with counterterms (∂λ(∂μ(DT))) via automatic differentiation
        dμ_diagtree = DiagTree.derivative([diagtree], BareGreenId, p[2]; index=1)
        dλ_dμ_diagtree = DiagTree.derivative(dμ_diagtree, BareInteractionId, p[3]; index=2)
        if isempty(dλ_dμ_diagtree)
            @warn("Ignoring partition $p with no diagrams")
        else
            isFock && DiagTree.removeHartreeFock!(dλ_dμ_diagtree)
            @debug "\nDiagTree:\n" * repr_tree(dλ_dμ_diagtree)
            # Compile to expression tree and save results for this partition
            exprtree = ExprTree.build(dλ_dμ_diagtree)
            push!(valid_partitions, p)
            push!(diagparams, diagparam)
            push!(exprtrees, exprtree)
            append!(diagtrees, dλ_dμ_diagtree)
        end
    end
    return valid_partitions, diagparams, diagtrees, exprtrees
end

function build_poln(; n_loop=0)
    # Polarization diagram parameters
    #
    # NOTE: Differentiation and Fock filter do not commute, so we need to manually zero
    #       out the Fock insertions after differentiation (via DiagTree.removeHartreeFock!)
    DiagTree.uidreset()
    diagparam = DiagParaF64(;
        type=PolarDiag,
        hasTau=true,
        innerLoopNum=n_loop,
        filter=[Proper, NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa interaction
    )
    # Build diagram tree dataframe
    diag_df = Parquet.polarization(diagparam; name=:Π)
    # Merge spins, Π_σ = Π↑↑ + Π↑↓
    diagtree = mergeby(diag_df.diagram)[1]
    @debug "\nDiagTree:\n" * repr_tree(diagtree)
    return diagparam, diagtree
end

"""
Generate a DiagramTree for the Hubbard-like Σ₂[G, V, Γⁱ₃ = Γ₀] diagram (without
dashed G-lines), re-expanded to O(Vⁿ) in a statically-screened interaction V[λ].
"""
function build_sigma2_local(s::Settings)
    @assert isempty(s.indices_g_dash) && return build_local(s)
end
