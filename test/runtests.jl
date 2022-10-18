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
    include("integrate_fock.jl")
    include("integrate_sosem.jl")
else
    include(ARGS[1])
end