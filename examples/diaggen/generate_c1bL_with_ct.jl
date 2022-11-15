using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

settings = DiagGen.Settings(;
    observable=DiagGen.c1bL,
   max_order=4,
    verbosity=DiagGen.info,
    expand_bare_interactions=false,
)

# Build diagram and expression trees for all sigma_2 diagrams at order n with interaction counterterms
partitions, diagparams, diagtrees, exprtrees =
    DiagGen.build_nonlocal_with_ct(settings; renorm_mu=true);

println(partitions)
println(diagtrees)
println(exprtrees)

# Check the first diagram tree
for tree in diagtrees
    DiagGen.checktree(tree, settings; plot=true, maxdepth=10)
    # DiagGen.checktree(tree, settings)
end
