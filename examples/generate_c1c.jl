using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

settings = DiagGen.Settings(; observable=DiagGen.c1c, n_order=5, verbosity=DiagGen.info)
cfg = DiagGen.Config(settings)

# Build diagram and expression trees for all sigma_2 diagrams at order n
som_c1c, som_c1c_compiled = DiagGen.build_nonlocal(settings);

# Check the diagram tree
# DiagGen.checktree(som_c1c, settings; plot=true, maxdepth=6)
DiagGen.checktree(som_c1c, settings)
