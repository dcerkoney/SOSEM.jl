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

function main()
    # Physical params matching data for SOSEM observables
    order = [5]
    beta = [40.0]

    # rs = [5.0]
    # mass2 = [0.5, 0.625, 0.875, ...]

    # rs = [4.0]
    # mass2 = [0.5625, 0.625, 0.75, 1.0]

    # rs = [3.0]
    # mass2 = [1.25]

    rs = [2.0]
    mass2 = [1.75, 2.5]

    # rs = [1.0]
    # mass2 = [1.0, 1.75]
    # mass2 = [3.5]

    # Total number of MCMC evaluations
    neval = 1e10

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Remove Fock insertions?
    isFock = false

    # Distinguish results with different counterterm schemes
    ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end

    # Get self-energy data needed for the chemical potential and Z-factor measurements
    for (_rs, _mass2, _beta, _order) in Iterators.product(rs, mass2, beta, order)
        para = UEG.ParaMC(;
            order=_order,
            rs=_rs,
            beta=_beta,
            mass2=_mass2,
            isDynamic=false,
            isFock=isFock,
            dim=3,
        )
        kF = para.kF

        ######### calculate K dependence #####################
        Nk, korder = 4, 4
        minK = 0.2kF
        kgrid =
            CompositeGrid.LogDensedGrid(:uniform, [0.0, 2.2kF], [kF], Nk, minK, korder).grid
        ngrid = [-1, 0]  # for improved finite-temperature effects

        # Build diagrams
        orders = 1:_order
        n_min, n_max = 1, _order
        partition = UEG_MC.counterterm_partitions(
            n_min,
            n_max;
            n_lowest=1,
            renorm_mu=renorm_mu,
            renorm_lambda=renorm_lambda,
        )
        neighbor = UEG.neighbor(partition)
        @time diagram = Sigma.diagram(para, partition)
        valid_partition = diagram[1]

        #! format: off
        reweight_goal = [1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 4.0, 4.0, 8.0, 2.0, 2.0, 2.0,
            4.0, 4.0, 8.0, 4.0, 4.0, 8.0, 8.0, 2.0]
        reweight_pad = repeat([2.0], max(0, length(valid_partition) - length(reweight_goal) + 1))
        reweight_goal = [reweight_goal; reweight_pad]
        @assert length(reweight_goal) ≥ length(valid_partition) + 1
        #! format: on

        sigma, result = Sigma.KW(
            para,
            diagram;
            neighbor=neighbor,
            reweight_goal=reweight_goal[1:(length(valid_partition) + 1)],
            # reweight_goal=reweight_goal[1:(length(partition) + 1)],
            kgrid=kgrid,
            ngrid=ngrid,
            neval=neval,
            parallel=:thread,
        )

        # Save data to JLD2
        if isnothing(sigma) == false
            println("Current working directory: $(pwd())")
            println("Saving data to JLD2...")
            jldopen("data/data_K.jld2", "a+"; compress=true) do f
                if haskey(f, "has_taylor_factors")
                    @assert f["has_taylor_factors"] == true
                else
                    f["has_taylor_factors"] = true
                end
                key = "$(UEG.short(para))"
                if haskey(f, key)
                    @warn("replacing existing data for $key")
                    delete!(f, key)
                end
                f[key] = (ngrid, kgrid, sigma)
                return
            end
            println("done!")
        end
    end
end

main()
