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
    # order = [5]  # C^{(1)}_{N≤6} includes CTs up to 5th order
    order = [4]  # C^{(1)}_{N≤5} includes CTs up to 4th order
    beta = [40.0]
    rs = [1.0, 2.0, 5.0]
    # rs = [1.0]
    # mass2 = [1.0]

    # Using mass2 from optimization of C⁽¹⁾ⁿˡ(k = 0)
    # TODO: Optimize specifically for ReΣ_N(kF, ik0) convergence vs N
    c1nl_mass2_optima = Dict{Float64,Float64}(1.0 => 1.0, 2.0 => 0.4, 5.0 => 0.1375)

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
    # for (_rs, _mass2, _beta, _order) in Iterators.product(rs, mass2, beta, order)
    for (_rs, _beta, _order) in Iterators.product(rs, beta, order)
        @assert haskey(c1nl_mass2_optima, _rs) "Missing optimized mass2 for rs = $_rs"
        para = UEG.ParaMC(;
            order=_order,
            rs=_rs,
            beta=_beta,
            mass2=c1nl_mass2_optima[_rs],
            # mass2=_mass2,
            isDynamic=false,
            isFock=isFock,
        )

        ######### calculate Z factor ######################
        # For Z(k = 0)
        # kgrid = [0.0]
        # For μ & Z := Z(k = kF)
        kgrid = [para.kF]
        ngrid = [0, 1]
        # ngrid = [-1, 0]

        # Build diagrams
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
        reweight_goal = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0]
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
            # jldopen("data_Z$(ct_string)_k0.jld2", "a+"; compress=true) do f
            jldopen("data_Z$(ct_string).jld2", "a+"; compress=true) do f
                #jldopen("data_Z$(ct_string)_kF.jld2", "a+"; compress=true) do f
                # jldopen("data_Z.jld2", "a+") do f
                key = "$(UEG.short(para))"
                if haskey(f, key)
                    @warn("replacing existing data for $key")
                    delete!(f, key)
                end
                return f[key] = (para, ngrid, kgrid, sigma)
            end
            println("done!")
        end
    end
end

main()
