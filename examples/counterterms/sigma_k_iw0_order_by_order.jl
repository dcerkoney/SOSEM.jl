using CodecZlib
using CompositeGrids
using ElectronLiquid
using FeynmanDiagram
using JLD2
using MCIntegration
using Measurements
using SOSEM

# Change to counterterm directory
if haskey(ENV, "SOSEM_CEPH")
    cd("$(ENV["SOSEM_CEPH"])/examples/counterterms")
elseif haskey(ENV, "SOSEM_HOME")
    cd("$(ENV["SOSEM_HOME"])/examples/counterterms")
end

const lambda_opt = Dict(
    1.0 => [1.75, 1.75, 1.75, 1.75, 1.75],
    2.0 => [2.0, 2.0, 2.0, 2.0, 2.0],
    3.0 => [0.75, 0.75, 1.0, 1.25, 1.75],
    4.0 => [0.625, 0.625, 0.75, 1.0, 1.125],
)

function main()
    # Physical params matching data for SOSEM observables
    dim = 3
    max_orders = [2, 3, 4, 5] # calculate orders 1 & 2 together, and run the rest separately
    beta = 40.0
    rs = 3.0

    # Total number of MCMC evaluations
    neval = 1e10

    # diagtype = :GV
    diagtype = :Parquet
    filename = "data/data_K_$(diagtype).jld2"

    # Remove Fock insertions?
    isFock = false

    # spin-polarization parameter (n_up - n_down) / (n_up + n_down) ∈ [0,1]
    spinPolarPara = 0.0

    # Get self-energy data needed for the chemical potential and Z-factor measurements
    @assert haskey(lambda_opt, rs) "Lambda optima for rs = $(rs) not found!"
    lambdas = lambda_opt[rs]
    for order in max_orders
        mass2 = lambdas[order]
        para = UEG.ParaMC(;
            order=order,
            rs=rs,
            beta=beta,
            mass2=mass2,
            isDynamic=false,
            isFock=isFock,
            dim=dim,
        )
        kF = para.kF

        ######### calculate K dependence #####################
        Nk, korder = 4, 4
        minK = 0.2kF
        kgrid =
            CompositeGrid.LogDensedGrid(:uniform, [0.0, 2.2kF], [kF], Nk, minK, korder).grid
        ngrid = [-1, 0]  # for improved finite-temperature effects

        # Integrate and save self-energy results to file
        Sigma.MC(
            para;
            kgrid=kgrid,
            ngrid=ngrid,
            spinPolarPara=spinPolarPara,
            neval=neval,
            filename=filename,
            diagtype=diagtype,
        )
    end
end

main()
