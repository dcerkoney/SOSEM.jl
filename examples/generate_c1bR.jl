using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

settings = DiagGen.Settings(; observable=DiagGen.c1bR, n_order=4, verbosity=DiagGen.info)

# Build diagram and expression trees for all sigma_2 diagrams at order n
som_c1bR, som_c1bR_compiled = DiagGen.build_nonlocal(settings);

# Check the diagram tree
DiagGen.checktree(som_c1bR, settings; plot=true, maxdepth=10)
# DiagGen.checktree(som_c1bR, settings)
