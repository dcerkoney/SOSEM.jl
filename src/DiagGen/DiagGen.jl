"""Diagrammatic generation of second-order self-energy moments."""
module DiagGen

using AbstractTrees
using ..FeynmanDiagram
using ..Logging
using ..Parameters

include("IntegerCompositions/weak_integer_compositions.jl")
using .IntegerCompositions
export IntegerCompositions

include("common.jl")
export Verbosity, quiet, info, verbose
export Gamma3InsertionSide, left, right
export DiscontSide, negative, positive
export Observables, sigma20, sigma2, c1a, c1bL0, c1bR0, c1bL, c1bR, c1c, c1d
export Settings,
    Config, getID, propagator_params, checktree, weakintsplit, counterterm_split

# Non-local moment
include("build_nonlocal.jl")
export build_nonlocal, build_sigma2_nonlocal

# # TEST: non-local moment with counterterms
# include("build_nonlocal_with_ct.jl")
# export build_nonlocal_with_ct, build_sigma2_nonlocal_with_ct

# # Local moment
# @todo && include("build_local.jl")
# export build_local, build_sigma2_local

# # TODO: Full local and nonlocal SOSEM and direct self-energy generation
# @todo && include("build_full.jl")
# export build_full_local, build_full_nonlocal

end
