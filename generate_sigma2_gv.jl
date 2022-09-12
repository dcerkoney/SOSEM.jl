# Package includes
using AbstractTrees
using Combinatorics: with_replacement_combinations
using FeynmanDiagram

# Local module includes
include("weak_integer_compositions.jl")
using .IntegerCompositions

const DiagramF64 = Diagram{Float64}
const multicombinations = with_replacement_combinations

# Settings
n_order = 4
plot = true
debug = false
verbose = true
expand_bare_interactions = false

# Verbose print level
vprint = (verbose ? print : function (args...) end)
vprintln = (verbose ? println : function (args...) end)

# Debug print level
dprintln = (debug ? println : function (args...) end)

macro todo()
    :(error("Not yet implemented!"))
end

function numerical_suffix(n)
    if n == 1
        return "st"
    elseif n == 2
        return "nd"
    elseif n == 3
        return "rd"
    else
        return "th"
    end
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

function propr_params(type, n_expand, firstTauIdx, firstLoopIdx, filter=[NoHartree,])
    # The bare interaction is instantaneous (interactionTauNum = 1),
    # so innerLoopNum = totalTauNum = n_expand.
    return DiagParaF64(
        type=type,
        hasTau=true,
        innerLoopNum=n_expand,
        firstTauIdx=firstTauIdx,
        firstLoopIdx=firstLoopIdx,
        totalLoopNum=n_order + 1, # it seems we need to set this value?
        interaction=[Interaction(ChargeCharge, Instant),],
        filter=filter,
    )
end

"""
Generates a DiagramTree for the O(V^n) exchange self-energy 
with lowest-order 3-point vertex insertion, i.e., Sigma_2[G, V]|_n
(the O(V^2) exchange term in the bold G, bare V series re-expanded to O(V^n)).
"""
function build_sigma2_gv()
    # TODO: reexpress bare interaction lines via Parquet.vertex4 to allow for
    #       optional re-expansion via the statically screened interaction
    if expand_bare_interactions
        @todo
    end

    # Total re-expansion order
    n_expand = n_order - 2

    k = vcat([1, 0, 0], repeat([0], n_expand))     # external momentum
    k1 = vcat([0, 1, 0], repeat([0], n_expand))    # k1 = k + q1
    k3 = vcat([0, 0, 1], repeat([0], n_expand))    # k3 = k + q2
    # k = [1, 0, 0]     # external momentum
    # k1 = [0, 1, 0]    # k1 = k + q1
    # k3 = [0, 0, 1]    # k3 = k + q2
    k2 = k1 + k3 - k                               # k2 = k + q1 + q2
    q1 = k1 - k
    q2 = k3 - k
    # leg_k = [-q1, k2]                            # [Q, Kin] for Gamma_3

    # Green's function labels, times, and momenta
    n_g = 3
    g_names = [:G0_1, :G0_2, :G0_3]
    g_ks = [k1, k2, k3]
    g_taus = [(1, 2), (2, 1), (1, 2)]

    # Interaction labels, times, and momenta
    n_v = 2
    v_names = [:V_1, :V_2]
    v_qs = [q1, q2]
    v_taus = [(1, 1), (2, 2)]

    # Re-expand the three Green's function lines, and optionally,
    # the two bare Coulomb interaction lines as well
    expandables = expand_bare_interactions ? [g_names; v_names] : g_names
    n_expandables = length(expandables)
    max_expandables = n_g + n_v

    # The number of (possibly indistinct) diagram trees to generate is: 
    # ((n_expandables n_expand)) = binomial(n_expandables + n_expand - 1, n_expand)
    n_multicombinations = binomial(n_expandables + n_expand - 1, n_expand)
    print("Generating (($n_expandables $n_expand)) = $n_multicombinations expanded diagram trees...")
    dprintln()

    # Construct parameters and ID for the self-energy sigma2
    # The bare interaction is instantaneous (interactionTauNum = 1), 
    # so n_order = innerLoopNum = totalTauNum
    sigma2_params = DiagParaF64(
        type=SigmaDiag,
        hasTau=true,
        innerLoopNum=n_order,
        totalTauNum=n_order,
        firstTauIdx=1,
        interaction=[Interaction(ChargeCharge, Instant),],
    )
    # The (dynamic) self-energy has external momentum k and times (1, n)
    sigma2_id = SigmaId(sigma2_params, Dynamic, k=k, t=(1, 2))

    # for expansion_indices in multicombinations(1:n_expandables, n_expand)
    #     # Get expansion orders (weak compositions) for each line by counting the expansion indices
    #     expansion_orders = zeros(Int, max_expandables)
    #     for i in expansion_indices
    #         expansion_orders[i] += 1
    #     end

    # Generate a list of all expanded diagrams at fixed order n
    tree_count = 0
    sigma2_diags = Vector{DiagramF64}()
    generic_id = GenericId(propr_params(GreenDiag, 0, 1, 1))
    for expansion_orders in rpadded_weak_integer_compositions(n_expand, n_expandables, pad=n_v)
        dprintln("\nTree #$(tree_count+1):")
        dprintln("  • Expansion orders:\t\t\t\t$expansion_orders")

        # The firstTauIdx for each G line depends on the expansion order of the previous G.
        # NOTE: the default offset for Green's functions is FeynmanDiagram.firstTauIdx(GreenDiag) = 3
        g_ftis, g_max_fti = Parquet.findFirstTauIdx(
            expansion_orders[1:n_g],
            repeat([GreenDiag], n_g),
            FeynmanDiagram.firstTauIdx(GreenDiag),  # leftmost firstTauIdx
            1,                                      # = interactionTauNum
        )
        dprintln("  • First tau indices for G_i's:\t\t$g_ftis (maxTauIdx = $g_max_fti)")

        # First loop indices for each Green's function
        # NOTE: the default offset for Green's functions is FeynmanDiagram.firstLoopIdx(GreenDiag) = 2
        g_flis, g_max_fli = Parquet.findFirstLoopIdx(
            expansion_orders[1:n_g],
            FeynmanDiagram.firstLoopIdx(GreenDiag, n_v), # offset = n_v = 2 due to two bare interactions
        )
        dprintln("  • First momentum loop indices for G_i's:\t$g_flis (maxLoopIdx = $g_max_fli)")

        # TODO: Add counterterms---for n[i] expansion order of line i, spend n_cti
        #       orders on counterterm derivatives in all possible ways (0 < n_cti < n[i]).
        #       E.g., if n[i] = 4, we have: ni_left, n_cti = weak_split(n[i])

        # Green's function and bare interaction params
        # g_params = [propr_params(GreenDiag, expansion_orders[i], g_max_fti, g_max_fli) for i in 1:n_g]
        g_params = [propr_params(GreenDiag, expansion_orders[i], g_ftis[i], g_flis[i]) for i in 1:n_g]
        v_params = [propr_params(Ver4Diag, expansion_orders[i + n_g], i, i) for i in 1:n_v]

        # Re-expanded Green's function and bare interaction lines
        g_lines = [Parquet.green(g_params[i], g_ks[i], g_taus[i], name=g_names[i]) for i in 1:n_g]
        v_lines = [
            DiagramF64(
                # We mark the outer two bare interactions as fixed via `order[end] = -1`
                BareInteractionId(v_params[i], ChargeCharge, Instant, [0, 0, 0, -1],
                    k=v_qs[i], t=v_taus[i], permu=Di),
                name=v_names[i]
            )
            for i in 1:n_v
        ]

        # Build this diagram tree and add to the list
        this_diag = DiagramF64(generic_id, Prod(), [g_lines; v_lines])
        push!(sigma2_diags, this_diag)
        tree_count += 1
    end
    @assert tree_count == n_multicombinations
    dprintln()
    println("done!\n")

    # Now construct the self-energy diagram tree
    vprint("Merging subtrees...")
    sigma2 = DiagramF64(sigma2_id, Sum(), sigma2_diags, name=:Sigma_2)
    vprintln("done!\n")

    return sigma2
end

function main()
    DiagTree.uidreset()

    # Build the diagram tree for all sigma2 diagrams at order n
    sigma2 = build_sigma2_gv()
    if verbose
        println("DiagTree:")
        print_tree(sigma2)
        # Check subtree for one high-order Green's function line(s)
        for n in sigma2.subdiagram[1].subdiagram
            if n.name == :G
                order = n.id.para.innerLoopNum
                suffix = numerical_suffix(order)
                println("\n$order$suffix-order Green's function subtree:")
                print_tree(n)
                break
            end
        end
    end

    # Build expression tree
    sigma2_compiled = ExprTree.build([sigma2])
    vprintln()
    vprintln(sigma2_compiled)
    for (i, node) in enumerate(sigma2_compiled.node)
        vprintln("\u001b[32m$i\u001b[0m : $node")
    end
    vprintln()

    return sigma2, sigma2_compiled
end

sigma2, sigma2_compiled = main()

# Plot the DiagTree
if plot
    plot_tree(sigma2)
end