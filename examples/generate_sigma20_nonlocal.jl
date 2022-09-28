using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

settings = DiagGen.Settings(; observable=DiagGen.sigma20, n_order=4, verbosity=DiagGen.info)

# Build diagram and expression trees for all sigma_2 diagrams at order n
sigma20, sigma20_compiled = DiagGen.build_sigma2_nonlocal(settings)

# Check the diagram tree
DiagGen.checktree(sigma20, settings; plot=true, maxdepth=10)
# DiagGen.checktree(sigma20, settings)
