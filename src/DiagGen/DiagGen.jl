"""Diagrammatic generation of second-order self-energy moments."""
module DiagGen

using AbstractTrees
using ..ElectronLiquid
using ..FeynmanDiagram
using ..Logging
using ..Parameters
using ..SOSEM: @todo

# Convenience typedefs for diagram and expression trees
const DiagramF64 = Diagram{Float64}
const ExprTreeF64 = ElectronLiquid.ExprTreeF64

# Convenience typedefs for Settings and Config
const VFloat64 = Vector{Float64}
const OptInt = Union{Nothing,Int}
const ProprTauType = Tuple{Int,Int}
const ProprOptTauType = Tuple{OptInt,Int}
const Gamma3OptTauType = Tuple{Int,Int,OptInt}

include("IntegerCompositions/weak_integer_compositions.jl")
using .IntegerCompositions
export IntegerCompositions

include("properties.jl")
export Gamma3InsertionSide, left, right
export DiscontSide, negative, positive, both
export Observables, sigma20, sigma2, c1a, c1bL0, c1bR0, c1bL, c1bR, c1c, c1d
export get_bare_string, get_exact_k0, getID, propagator_param

include("common.jl")
export Verbosity, quiet, info, verbose
export Settings,
    Config, checktree, counterterm_partitions, counterterm_partitions_fixed_order

# Non-local moment
include("build_nonlocal.jl")
export build_nonlocal, build_nonlocal_with_ct, build_sigma2_nonlocal, build_diagtree

# # Local moment
# include("build_local.jl")
# export build_local, build_sigma2_local

# # TODO: Full local and nonlocal SOSEM and direct self-energy generation
# @todo && include("build_full.jl")
# export build_full_local, build_full_nonlocal

end  # module DiagGen
