using FeynmanDiagram
using SOSEM
using SOSEM.DiagGen

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

# Generate diagrams at fixed order n
order = 2

# The non-local SOSEM without VCs for the UEG is a composite observable
settings = Settings{CompositeObservable}(;
    observable=c1nl0_ueg,
    min_order=order,
    max_order=order,
    verbosity=info,
    expand_bare_interactions=false,
    filter=[NoHartree],
    interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
)
println(atomize(settings))
cfgs = Config(settings)

# Build diagram and expression trees for all diagrams at order n
diagparam, diagtrees, exprtree = build_full_nonlocal_fixed_order(settings);
diagtree_c1c, diagtree_rest = diagtrees
@assert length(exprtree.root) == 2 # (1) c1c (2) rest

# Check the diagram tree
checktree(diagtree_c1c, settings; plot=true, maxdepth=10)
checktree(diagtree_rest, settings; plot=true, maxdepth=10)
# checktree(diagtrees, settings)
