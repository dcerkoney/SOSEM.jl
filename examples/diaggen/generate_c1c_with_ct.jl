using FeynmanDiagram
using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

settings = DiagGen.Settings(;
    observable=DiagGen.c1c,
    min_order=5,
    max_order=5,
    verbosity=DiagGen.info,
    expand_bare_interactions=false,
    filter=[NoHartree],
    interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
)
cfg = DiagGen.NonlocalConfig(settings)

# Build diagram and expression trees for all sigma_2 diagrams at order n with interaction counterterms
partitions, diagparams, diagtrees, exprtrees = DiagGen.build_nonlocal_with_ct(settings);

println(partitions)
println(diagtrees)
println(exprtrees)

@assert all(t.factor == -1 for t in diagtrees)

# Check the first diagram tree
for tree in diagtrees
    DiagGen.checktree(tree, settings; print=false, plot=true, maxdepth=6)
    # DiagGen.checktree(tree, settings)
end
