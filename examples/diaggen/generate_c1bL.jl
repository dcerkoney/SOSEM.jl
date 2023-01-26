using FeynmanDiagram
using SOSEM

# Generate debug info
if isinteractive()
    ENV["JULIA_DEBUG"] = SOSEM
end

settings = DiagGen.Settings{DiagGen.Observable}(;
    DiagGen.c1bL,
    min_order=5,
    max_order=5,
    verbosity=DiagGen.verbose,
    expand_bare_interactions=false,
    filter=[NoHartree],
    interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],  # Yukawa-type interaction
)
cfg = DiagGen.Config(settings)

# Build diagram and expression trees for all sigma_2 diagrams at order n
diagparam, som_c1bL, som_c1bL_compiled = DiagGen.build_nonlocal_fixed_order(settings);

# Check the diagram tree
DiagGen.checktree(som_c1bL, settings; print=true, plot=true, maxdepth=6)
# DiagGen.checktree(som_c1bL, settings)

@assert som_c1bL.factor == 1
