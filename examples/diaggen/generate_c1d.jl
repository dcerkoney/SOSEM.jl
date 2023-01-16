using FeynmanDiagram
using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

settings = DiagGen.Settings(;
    observable=DiagGen.c1d,
    min_order=4,
    max_order=4,
    verbosity=DiagGen.info,
    expand_bare_interactions=false,
    filter=[NoHartree],
    interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
)
cfg = DiagGen.Config(settings)

# Build diagram and expression trees for all sigma_2 diagrams at order n
diagparam, som_c1d, som_c1d_compiled = DiagGen.build_nonlocal_fixed_order(settings);

# Check the diagram tree
DiagGen.checktree(som_c1d, settings; plot=true, maxdepth=10);
# DiagGen.checktree(som_c1d, settings)

@assert som_c1d.factor == 1
