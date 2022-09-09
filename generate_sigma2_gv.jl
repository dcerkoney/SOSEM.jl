using AbstractTrees
using Combinatorics: with_replacement_combinations
using FeynmanDiagram

const DiagramF64 = Diagram{Float64}
const multicombinations = with_replacement_combinations

const n_order = 2
const expand_bare_interactions = false
const verbose = true

macro todo()
    :(error("Not yet implemented!"))
end

# Generate weak compositions of size 2 of an integer n,
# (i.e., the cycle (n, 0), (n-1, 1), ..., (0, n))
function weak_split(n::Integer)
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

function propr_params(type, n_expand, firstTauIdx, filter=[NoHartree,])
    # The bare interaction is instantaneous (interactionTauNum = 1),
    # so innerLoopNum = totalTauNum = n_expand.
    return DiagParaF64(
        type=type,
        hasTau=true,
        innerLoopNum=n_expand,
        totalTauNum=n_expand + 2,
        firstTauIdx=firstTauIdx,
        interaction=[Interaction(ChargeCharge, Instant),],
        filter=filter,
    )
end

"""
Generates a DiagramTree for the O(V^n) exchange self-energy 
with lowest-order 3-point vertex insertion, i.e., Sigma_2[G, V]|_n
(the O(V^2) exchange term in the bold G, bare V series re-expanded to O(V^n)).
"""
function build_sigma2_gv(; id::DiagramId, n_order=2, expand_bare_intns=false, verbose=false)
    vprintln = (verbose ? println : function (args...) end)

    if expand_bare_intns
        # TODO: reexpress bare interaction lines via Parquet.vertex4
        @todo
    end

    # Inner expansion order for bold lines
    n_expand = n_order - 2

    k = vcat([1, 0, 0], repeat([0], n_expand))     # external momentum
    k1 = vcat([0, 1, 0], repeat([0], n_expand))    # k1 = k + q1
    k3 = vcat([0, 0, 1], repeat([0], n_expand))    # k3 = k + q2
    k2 = k1 + k3 - k                               # k2 = k + q1 + q2
    q1 = k1 - k
    q2 = k3 - k
    # leg_k = [-q1, k2]                            # [Q, Kin] for Gamma_3

    # Green's function labels, times, and momenta
    g_names = [:G0_1, :G0_2, :G0_3]
    g_ks = [k1, k2, k3]
    g_taus = [(1, n_order), (n_order, 1), (1, n_order)]

    # Interaction labels, times, and momenta
    v_names = [:V_1, :V_2]
    v_qs = [q1, q2]
    v_taus = [(1, 1), (n_order, n_order)]

    # Re-expand the three Green's function lines, and optionally,
    # the two bare Coulomb interaction lines as well
    expandables = expand_bare_intns ? [g_names; v_names] : g_names
    n_expandables = length(expandables)
    const max_expandables = 5

    # The number of (possibly indistinct) diagrams to generate is: 
    # ((n_expandables n_order)) = binomial(n_expandables + n_order - 1, n_order)
    n_multicombinations = binomial(n_expandables + n_order - 1, n_order)
    vprintln("Generating ((n_expandables order)) = ", n_multicombinations, " expanded diagrams...")

    # Generate a list of all expanded diagrams at fixed order n
    count = 1
    sigma2_diags = Vector{Diagram{W}}()
    generic_id = GenericId(propr_params(GreenDiag, 0, 1))
    for expansion_indices in multicombinations(1:n_expandables, n_order)
        # Get expansion orders (weak compositions) for each line by counting the expansion indices
        expansion_orders = zeros(Int, max_expandables)
        for i in expansion_indices
            expansion_orders[i] += 1
        end
        vprintln("\tExpansion orders: ", expansion_orders)
        vprintln()

        # The firstTauIdx for each G line depends on the expansion order of the previous G.
        g_ftis = firstTauIdx(GreenDiag) .+ [0; expansion_orders[1:end-1]]

        # TODO: add counterterms---for n[i] expansion order of line i, spend n_cti
        #       orders on counterterm derivatives in all possible ways (0 < n_cti < n[i]).
        #       E.g., if n[i] = 4, we have: ni_left, n_cti = weak_split(n[i])

        # Green's function and bare interaction params
        g_params = [propr_params(GreenDiag, expansion_orders[i], g_ftis[i]) for i in 1:3]
        v_params = [propr_params(Ver4Diag, expansion_orders[3:i], i) for i in 1:2]

        # Re-expanded Green's function and bare interaction lines
        g_lines = [Parquet.green(g_params, g_ks[i], g_taus[i], name=g_names[i]) for i in 1:3]
        v_lines = [
            DiagramF64(
                BareInteractionId(v_params[i], ChargeCharge, Instant, [0, 0, 0, -1],
                    k=v_qs[i], t=v_taus[i], permu=Di),
                name=v_names[i]
            )
            for i in 1:2
        ]

        # Build this diagram and add to the list
        this_diag = DiagramF64(generic_id, Prod(), [g_lines; v_lines])
        push!(sigma2_diags, this_diag)
        count += 1
    end
    @assert count == n_multicombinations
    vprintln("done!")

    # Now construct the self-energy diagram tree. The bare interaction is
    # instantaneous (interactionTauNum = 1), so n_order = innerLoopNum = totalTauNum
    sigma2_params = DiagParaF64(
        type=SigmaDiag,
        hasTau=true,
        innerLoopNum=n_order,
        totalTauNum=n_order,
        firstTauIdx=1,
        interaction=[Interaction(ChargeCharge, Instant),],
    )
    # The (dynamic) self-energy has external momentum k and times (1, n)
    sigma2_id = SigmaId(sigma2_params, Dynamic, k=k, t=(1, n_order))
    sigma2 = DiagramF64(sigma2_id, Sum(), sigma2_diags)
    return sigma2
end

function main()
    # Parses cmdline arguments to a dict
    argdict = parse_cmdline(args)
    println(argdict)

    DiagTree.uidreset()

    # Build the diagram tree for all sigma2 diagrams at order n
    sigma2 = build_sigma2_gv(
        id=sigma2_id,
        n_order=argdict["n_order"],
        expand_bare_intns=argdict["expand_bare_interactions"],
        verbose=argdict["verbose"],
    )

    # Build expression tree
    sigma2_compiled = ExprTree.build([sigma2])

    return sigma2, sigma2_compiled
end

sigma2, sigma2_compiled = main()

# Print the diagram and expression trees
if verbose
    print_tree(sigma2)
    println(sigma2_compiled)
    for node in sigma2_compiled.node
        println(node)
    end
end

# Plot the DiagTree
if plot
    plot_tree(sigma2)
end