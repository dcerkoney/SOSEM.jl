"""Diagrammatic generation of second-order self-energy moments."""
module DiagGen

using AbstractTrees
using Combinatorics
using ..ElectronLiquid
using ..FeynmanDiagram
using ..Logging
using ..Parameters
using ..SOSEM: @todo, alleq, DiagramF64, ExprTreeF64, PartitionType, MergedPartitionType

# Convenience typedefs for Settings and Config
const VFloat64 = Vector{Float64}
const OptInt = Union{Nothing,Int}
const ProprTauType = Tuple{Int,Int}
const ProprOptTauType = Tuple{OptInt,Int}
const Gamma3OptTauType = Tuple{Int,Int,OptInt}
const Interaction = FeynmanDiagram.Interaction
const Filter = FeynmanDiagram.Filter

include("IntegerCompositions/weak_integer_compositions.jl")
using .IntegerCompositions
export IntegerCompositions

include("properties.jl")
export Gamma3InsertionSide, left, right
export DiscontSide, negative, positive, both
export CompositeObservable, c1b_total, c1nl0, c1nl, c1b_total_ueg, c1nl0_ueg, c1nl_ueg
export Observable, sigma20, sigma2, c1a, c1bL0, c1bR0, c1bL, c1bR, c1c, c1d
export get_bare_string, get_exact_k0, getID, propagator_param

include("common.jl")
export Verbosity, quiet, info, verbose
export Settings, Config, checktree, atomize

include("partitions.jl")
export partition,
    integer_compositions,
    counterterm_partitions,
    counterterm_partitions_fixed_order,
    counterterm_single_split

# Non-local moment
include("build_nonlocal.jl")
export build_nonlocal,
    build_nonlocal_fixed_order,
    build_nonlocal_with_ct,
    build_sigma2_nonlocal,
    build_diagtree

# # Local moment
# include("build_local.jl")
# export build_local, build_sigma2_local

# # TODO: Full local and nonlocal SOSEM and direct self-energy generation
# @todo && include("build_full.jl")
# export build_full_local, build_full_nonlocal

end  # module DiagGen
