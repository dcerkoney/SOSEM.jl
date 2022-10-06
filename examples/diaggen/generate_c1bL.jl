using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

settings = DiagGen.Settings(; observable=DiagGen.c1bL, n_order=4, verbosity=DiagGen.info)

# Build diagram and expression trees for all sigma_2 diagrams at order n
som_c1bL, som_c1bL_compiled = DiagGen.build_nonlocal(settings);

# Check the diagram tree
DiagGen.checktree(som_c1bL, settings; plot=true, maxdepth=10)
# DiagGen.checktree(som_c1bL, settings)
