using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

settings = DiagGen.Settings(; observable=DiagGen.c1bL0, n_order=4, verbosity=DiagGen.info)

# Build diagram and expression trees for all sigma_2 diagrams at order n
diagparam, som_c1bL0, som_c1bL0_compiled = DiagGen.build_nonlocal(settings)

# Check the diagram tree
DiagGen.checktree(som_c1bL0, settings; plot=true, maxdepth=10)
# DiagGen.checktree(som_c1bL0, settings)
