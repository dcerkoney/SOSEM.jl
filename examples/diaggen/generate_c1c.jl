using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

settings = DiagGen.Settings(;
    observable=DiagGen.c1c,
    n_order=2,
    verbosity=DiagGen.info,
    expand_bare_interactions=false,
)
cfg = DiagGen.Config(settings)

# Build diagram and expression trees for all sigma_2 diagrams at order n
diagparam, som_c1c, som_c1c_compiled = DiagGen.build_nonlocal(settings);

# Check the diagram tree
# DiagGen.checktree(som_c1c, settings; plot=true, maxdepth=10)
# DiagGen.checktree(som_c1c, settings)
