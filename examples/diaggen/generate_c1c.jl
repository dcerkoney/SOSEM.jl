using FeynmanDiagram
using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

settings = DiagGen.Settings{DiagGen.Observable}(;
    DiagGen.c1c,
    min_order=6,
    max_order=6,
    verbosity=DiagGen.info,
    expand_bare_interactions=false,
    filter=[NoHartree],
    interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
)
cfg = DiagGen.Config(settings)

# Build diagram and expression trees for all sigma_2 diagrams at order n
diagparam, som_c1c, som_c1c_compiled = DiagGen.build_nonlocal_fixed_order(settings);

# Check the diagram tree
DiagGen.checktree(som_c1c, settings; plot=true, maxdepth=10)
# DiagGen.checktree(som_c1c, settings)

@assert som_c1c.factor == -1
