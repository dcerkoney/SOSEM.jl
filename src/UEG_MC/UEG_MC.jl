"""Monte-Carlo (MC) evaluation of second-order self-energy moments for the uniform electron gas (UEG)."""
module UEG_MC

using ElectronGas
using ..ElectronLiquid
using ..FeynmanDiagram
using Lehmann
using LinearAlgebra
using MCIntegration
using ..Parameters
using ..SOSEM: @todo, DiagramF64, ExprTreeF64

"""UEG MC parameters necessary for plotting in post-processing"""
struct PlotParams
    order::Int
    rs::Float64
    beta::Float64
    kF::Float64
    qTF::Float64
    mass2::Union{Nothing,Float64}
    PlotParams(order, rs, beta, kF, qTF) = new(order, rs, beta, kF, qTF, nothing)
    PlotParams(order, rs, beta, kF, qTF, mass2) = new(order, rs, beta, kF, qTF, mass2)
end
export PlotParams

# Chemical potential renormalization for Monte-Carlo with counterterms
include("renormalization.jl")
export chemicalpotential_renormalization

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
