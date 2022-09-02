using FeynmanDiagram
using AbstractTrees
using Parameters
using StaticArrays
# using Combinatorics: with_replacement_combinations

const DiagramF64 = Diagram{Float64}

function propr_params(type, n_order, firstTauIdx)
    @assert n_order >= 2
    return DiagParaF64(
        type=type,
        hasTau=true,
        innerLoopNum=n_order,
        totalTauNum=n_order,
        # The bare interaction is instantaneous (interactionTauNum = 1)
        firstTauIdx=firstTauIdx,
        interaction=[Interaction(ChargeCharge, Instant),],
    )
end

DiagTree.uidreset()

# Expansion order (O(V^n})) for this run
n_order = 2  # First, generate the bare O(V^2) self-energy
    
k  = vcat([1, 0, 0], repeat([0], n_order - 2)) # external momentum
k1 = vcat([0, 1, 0], repeat([0], n_order - 2)) # k1 = k + q1
k3 = vcat([0, 0, 1], repeat([0], n_order - 2)) # k3 = k + q2
k2 = k1 + k3 - k                               # k2 = k + q1 + q2
q1 = k1 - k
q2 = k3 - k
# legk = [-q, k2]                              # [Q, Kin] for Gamma_3

# Bare Green's function labels, times, and momenta
g_names = [:G0_1, :G0_2, :G0_3]
g_taus  = [[1, 2], [2, 1], [1, 2]]
g_ks    = [k1, k2, k3]

# Bare interaction labels and momenta
v_names = [:V_1, :V_2]
v_taus  = [[1, 1], [2, 2]]
v_qs    = [q1, q2]

# Bare Green's function params
g_params = [propr_params(GreenDiag, n_order, g_taus[i][1]) for i in 1:3]

# Bare interaction line params
v_params = [propr_params(Ver4Diag, n_order, i) for i in 1:2]

# Hard-coded Bare Green's function and interaction lines
g_lines = [
    Diagram{Float64}(BareGreenId(g_params[i], k=g_ks[i], t=g_taus[i]),
        name=g_names[i])
    for i in 1:3
]
v_lines = [
    Diagram{Float64}(BareInteractionId(v_params[i], ChargeCharge,
        k=v_qs[i], t=v_taus[i], permu=Di), name=v_names[i])
    for i in 1:2
]

# Build the second-order self-energy diagram tree
sigma2_params = DiagParaF64(
    type=SigmaDiag,
    hasTau=true,
    innerLoopNum=n_order,
    totalTauNum=n_order,
    # The bare interaction is instantaneous (interactionTauNum = 1)
    firstTauIdx=1,
    interaction=[Interaction(ChargeCharge, Instant),],
)
id = GenericId(sigma2_params)
sigma2 = DiagramF64(id, Prod(), [g_lines; v_lines])

# print the tree
print_tree(sigma2)

# plot the tree
# plot_tree(sigma2)

# build expression tree
sigma2_exprtree = ExprTree.build([sigma2])
println(sigma2_exprtree)