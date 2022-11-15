using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

settings = DiagGen.Settings(;
    observable=DiagGen.c1d,
    min_order=3,
    max_order=3,
    verbosity=DiagGen.info,
)

# Build diagram and expression trees for all sigma_2 diagrams at order n
diagparam, som_c1d, som_c1d_compiled = DiagGen.build_nonlocal_fixed_order(settings);

# Check the diagram tree
DiagGen.checktree(som_c1d, settings; plot=true, maxdepth=10);
# DiagGen.checktree(som_c1d, settings)
