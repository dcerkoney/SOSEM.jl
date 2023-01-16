using ElectronGas
using ElectronLiquid.UEG: ParaMC, KOinstant
using FeynmanDiagram
using Lehmann
using LinearAlgebra
using MCIntegration
using Measurements
using SOSEM
using Test

if isempty(ARGS)
    include("test_sosem_integration.jl")
    include("test_lindhard.jl")
    include("test_fock_integration.jl")
    include("test_counterterm.jl")
else
    include(ARGS[1])
end
