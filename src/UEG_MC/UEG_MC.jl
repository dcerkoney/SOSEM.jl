"""Monte-Carlo (MC) evaluation of second-order self-energy moments for the uniform electron gas (UEG)."""
module UEG_MC

using DataFrames
using DataStructures: SortedDict
using DelimitedFiles
using ..DiagGen
using ElectronGas
using ..ElectronLiquid
using ..FeynmanDiagram
using JLD2
using Lehmann
using LinearAlgebra
using MCIntegration
using Measurements
using ..Parameters
using ..SOSEM: @todo, alleq, DiagramF64, ExprTreeF64, PartitionType, MergedPartitionType

# Convenience typedefs for measurement data
const MeasType       = Dict{PartitionType,T} where {T}
const MergedMeasType = Dict{MergedPartitionType,T} where {T}
const RenormMeasType = SortedDict{Int,T} where {T}
const TotalMeasType  = Dict{Int,T} where {T}

include("common.jl")
export restodict, load_fixed_order_data_jld2, aggregate_orders

# Chemical potential renormalization for Monte-Carlo with counterterms
include("renormalization.jl")
export chemicalpotential_renormalization_sosem,
    chemicalpotential_renormalization, delta_mu1, load_z_mu, fromFile, toFile

# Dimensionless Lindhard functions for the bare and statically-screened UEG theories
include("lindhard.jl")
export lindhard, screened_lindhard

# Propagator evaluation for Monte-Carlo
include("propagators.jl")
using .Propagators
export Propagators

# Nonlocal moment integration
include("integrate_nonlocal.jl")
export integrate_nonlocal # , integrate_sigma2_nonlocal

# Local moment integration
# include("integrate_local.jl")
# export integrate_local # , integrate_sigma2_local

end  # module UEG_MC
