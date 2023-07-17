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
    order = [4]  # C^{(1)}_{N≤5} includes CTs up to 3rd order
    beta = [40.0]

    rs = [1.0]
    mass2 = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5]
    #mass2 = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    #mass2 = [3.0, 3.5, 4.0]

    # rs = [2.0]
    # mass2 = [1.25, 1.5, 1.625, 1.75, 1.875, 2.0]

    # rs = [3.0]
    # mass2 = [0.75, 0.875, 1.0, 1.125, 1.25, 1.5]
    # mass2 = [1.0, 1.25, 1.5, 1.75, 5.0, 6.0]
    # mass2 = [2.0, 2.5, 3.0, 3.5, 4.0]
    # mass2 = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

    # rs = [4.0]
    # mass2 = [2.0, 2.25, 2.5, 2.75, 6.0, 7.0]
    # mass2 = [3.0, 3.5, 4.0, 4.5, 5.0]
    # mass2 = [0.25, 0.5, 0.75, 1.0, 1.25]
    # mass2 = [0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25]

    # rs = [5.0]
    # mass2 = [0.375, 0.5, 0.625, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.5]
    # mass2 = [3.0, 3.25, 3.5, 3.75, 7.0, 8.0]
    # mass2 = [4.0, 4.5, 5.0, 5.5, 6.0]
    # mass2 = [0.1, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5]
    # mass2 = [0.8125, 0.875, 0.9375]

    # mass2 = [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    # mass2 = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]
    # rs = [1.0]
    # mass2 = [1.0]

    # Momentum spacing for finite-difference derivative of Sigma (in units of para.kF)
    δK = 0.01

    # Total number of MCMC evaluations
    neval = 1e11

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
        )

        ######### calculate mass ratio ######################
        # k_points near k = 0
        # kgrid = para.kF * (δK * collect(-6:2:6))
        # k-points near k = kF
        kgrid = para.kF * (1 .+ δK * collect(-6:2:6))
        ngrid = [0]

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
        # reweight_goal = [
        #     1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 
        #     2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        # ]
        #reweight_goal = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0]
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
        )

        # Save data to JLD2
        if isnothing(sigma) == false
            println("Current working directory: $(pwd())")
            println("Saving data to JLD2...")
            jldopen(
                "data/data_mass_ratio.jld2",
                "a+";
                compress=true,
            ) do f
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
