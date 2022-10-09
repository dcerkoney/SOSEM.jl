"""
Prune branches with invalid derivatives on dashed Green's function lines. Since these lines
are just theta functions in τ, they have no chemical potential renormalization counterterms.
"""
function prune_invalid_dash_derivatives!(diagram::Diagram{W}; dry_run=false) where {W}
    # FIXME: may leave empty nodes which should be removed (breaks ExprTree compilation)
    @todo
    for node in PreOrderDFS(diagram)
        prune_indices = []
        for (i, child) in enumerate(node.subdiagram)
            for line in child.subdiagram
                # Isolate dashed lines (order[3] > 0 => dashed BareGreenId)
                if line.id.order[3] == 0
                    continue
                end
                @assert line.id isa BareGreenId
                # Check for Green's function derivatives on a dashed line
                if line.id.order[1] > 0
                    @debug """
                    \nMarking node for deletion:\t$child
                    Invalid derivative in line:\t$line
                    """
                    push!(prune_indices, i)
                end
            end
        end
        if !dry_run
            # Remove all children containing invalid dashed line derivatives from this node 
            deleteat!(node.subdiagram, prune_indices)
        end
    end
end

"""
Generate weak compositions of size 2 of an integer n,
(i.e., the cycle (n, 0), (n-1, 1), ..., (0, n))
"""
function weakintsplit(n::Integer)
    splits = []
    n1::Integer = n
    n2::Integer = 0
    while n1 >= 0
        push!(splits, (n1, n2))
        n1 -= 1
        n2 += 1
    end
    return splits
end
