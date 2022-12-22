using FeynmanDiagram
using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

# Generate diagrams at fixed order n
order = 2

# The non-local SOSEM without VCs for the UEG is a composite observable
settings_list = [
    DiagGen.Settings(;
        observable=observable,
        min_order=order,
        max_order=order,
        verbosity=DiagGen.info,
        expand_bare_interactions=false,
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
    ) for observable in DiagGen.c1nl0_ueg.observables
]
cfgs = DiagGen.Config.(settings_list)

# Build diagram and expression trees for all diagrams at order n
diagparam, diagtrees, exprtree = DiagGen.build_full_nonlocal_fixed_order(settings_list);
diagtree_c1c, diagtree_rest = diagtrees
@assert length(exprtree.root) == 2 # (1) c1c (2) rest

# Check the diagram tree
DiagGen.checktree(diagtree_c1c, settings_list[1]; plot=true, maxdepth=10)
DiagGen.checktree(diagtree_rest, settings_list[1]; plot=true, maxdepth=10)
# DiagGen.checktree(diagtrees, settings)
