using Test
using SOSEM
using ElectronGas
using Measurements
using ElectronLiquid.UEG: ParaMC

if isempty(ARGS)
    include("integrate.jl")
else
    include(ARGS[1])
end