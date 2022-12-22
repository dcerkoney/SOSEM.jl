"""Generation and integration of second-order self-energy moment (SOSEM) diagrams."""
module SOSEM

using ElectronLiquid
using FeynmanDiagram
using Logging
using Parameters

# Convenience typedefs for diagram and expression trees
const DiagramF64 = Diagram{Float64}
const ExprTreeF64 = ElectronLiquid.ExprTreeF64
export DiagramF64, ExprTreeF64

# Convenience typedefs for counterterm partitions
const PartitionType = Tuple{Int,Int,Int}
const MergedPartitionType = Tuple{Int,Int}
export PartitionType, MergedPartitionType

macro todo()
    return :(error("Not yet implemented!"))
end

# NOTE: Backport—function allequal is not available in julia<1.8
"""Checks that all elements of an iterable x are equal."""
function alleq(x)
    return all(isequal(first(x)), x)
end
export alleq

# SOSEM diagram generation
include("DiagGen/DiagGen.jl")
using .DiagGen
export DiagGen

# SOSEM MC evaluation for the uniform electron gas (UEG)
include("UEG_MC/UEG_MC.jl")
using .UEG_MC
export UEG_MC

end  # module SOSEM
