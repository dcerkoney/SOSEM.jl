using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

settings = DiagGen.Settings{DiagGen.Observable}(; DiagGen.sigma2,max_order=4, verbosity=DiagGen.info)

# Build diagram and expression trees for all sigma_2 diagrams at order n
diagparam, sigma2, sigma2_compiled = DiagGen.build_sigma2_nonlocal(settings)

# Check the diagram tree
DiagGen.checktree(sigma2, settings; plot=true, maxdepth=10)
# DiagGen.checktree(sigma2, settings)
