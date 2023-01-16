using FeynmanDiagram
using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

settings = DiagGen.Settings(;
    observable=DiagGen.c1bL0,
    min_order=5,
    max_order=5,
    verbosity=DiagGen.info,
    expand_bare_interactions=false,
    filter=[NoHartree],
    interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
)
cfg = DiagGen.Config(settings)

# Build diagram and expression trees for all sigma_2 diagrams at order n
diagparam, som_c1bL0, som_c1bL0_compiled = DiagGen.build_nonlocal_fixed_order(settings)

# Check the diagram tree
DiagGen.checktree(som_c1bL0, settings; plot=true, maxdepth=10)
# DiagGen.checktree(som_c1bL0, settings)

@assert som_c1bL0.factor == 1
