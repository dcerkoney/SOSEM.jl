using CodecZlib
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

function main()
    # Physical params matching data for SOSEM observables
    dim = 3
    order = [5]
    beta = [40.0]

    # rs = [5.0]
    # mass2 = [...]
    # rs = [4.0]
    # mass2 = [...]

    # rs = [3.0]
    # mass2 = [1.5]
    # rs = [1.0, 2.0, 5.0]
    # rs = [2.0]
    # mass2 = [1.5, 1.75, 2.0]

    #rs = [1.0]
    #mass2 = [3.5]

    ### For SOSEM data ###
    # rs = [2.0]
    # mass2 = [0.4]

    # Total number of MCMC evaluations
    neval = 1e11

    diagtype = :GV
    # diagtype = :Parquet
    filename = "data/data_Z_$(diagtype).jld2"

    # Remove Fock insertions?
    isFock = false

    # spin-polarization parameter (n_up - n_down) / (n_up + n_down) ∈ [0,1]
    spinPolarPara = 0.0

    # Get self-energy data needed for the chemical potential and Z-factor measurements
    for (_rs, _mass2, _beta, _order) in Iterators.product(rs, mass2, beta, order)
        para = UEG.ParaMC(;
            order=_order,
            rs=_rs,
            beta=_beta,
            mass2=_mass2,
            isDynamic=false,
            isFock=isFock,
            dim=dim,
        )

        ######### calculate Z factor ######################
        # For Z(k = 0)
        # kgrid = [0.0]

        # For μ & Z := Z(k = kF)
        kgrid = [para.kF]
        ngrid = [-1, 0]

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
