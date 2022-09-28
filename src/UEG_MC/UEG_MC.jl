"""Monte-Carlo (MC) evaluation of second-order self-energy moments for the uniform electron gas (UEG)."""
module UEG_MC

using ElectronGas
using ElectronLiquid
using ..FeynmanDiagram
using Lehmann
using LinearAlgebra
using MCIntegration
using ..Parameters

"""UEG MC parameters necessary for plotting in post-processing"""
struct PlotParams
    order::Int
    rs::Float64
    beta::Float64
    kF::Float64
    qTF::Float64
end
export PlotParams

# Propagator evaluation for Monte-Carlo
include("propagators.jl")
using .Propagators
export Propagators

# Low-order moment integration
include("integrate_nonlocal.jl")
export integrate_nonlocal # , integrate_sigma2_nonlocal

# include("build_c1a.jl") # local moment

# High-order moment integration
# include("integrate_c1bL_gamma.jl")
# include("integrate_c1bR_gamma.jl")
# export integrate_c1bL_gamma, integrate_c1bR_gamma

# Full SOSEM integration with/without Gamma_3
# include("integrate_som_gamma0.jl")
# include("integrate_som_gamma.jl")
# export integrate_som_gamma0, integrate_som_gamma

# Direct self-energy integration with/without Gamma_3
# include("integrate_sigma2_gamma0.jl")
# include("integrate_sigma2_gamma.jl")
# export integrate_sigma2_gamma0, integrate_sigma2_gamma

end