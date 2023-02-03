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

function integrandKW(idx, var, config)
    # function integrandKW(idx, varK, varT, config)
    varK, varT, N, ExtKidx = var
    para, diag, extT, kgrid, ngrid = config.userdata
    diagram = diag[idx]
    weight = diagram.node.current
    l = N[1]
    k = ExtKidx[1]
    varK.data[1, 1] = kgrid[k]
    wn = ngrid[l]

    ExprTree.evalKT!(diagram, varK.data, varT.data, para; eval=UEG_MC.Propagators.eval)
    # ExprTree.evalKT!(diagram, varK.data, varT.data, para)
    w = sum(
        weight[r] * Sigma.phase(varT, extT[idx][ri], wn, para.β) for
        (ri, r) in enumerate(diagram.root)
    )

    loopNum = config.dof[idx][1]
    factor = 1.0 / (2π)^(para.dim * loopNum)
    return w * factor #the current implementation of sigma has an additional minus sign compared to the standard defintion
end

function KW(
    para::ParaMC,
    diagram;
    kgrid=[para.kF],
    ngrid=[0],
    neval=1e6, #number of evaluations
    print=0,
    alpha=3.0, #learning ratio
    config=nothing,
    neighbor,
    reweight_goal,
    kwargs...,
)
    if haskey(kwargs, :solver)
        @assert kwargs[:solver] == :mcmc "Only :mcmc is supported for Sigma.KW"
    end
    para.isDynamic && UEG.MCinitialize!(para)

    dim, β, kF = para.dim, para.β, para.kF
    partition, diagpara, diag, root, extT = diagram

    K = MCIntegration.FermiK(dim, kF, 0.5 * kF, 10.0 * kF; offset=1)
    K.data[:, 1] .= 0.0
    K.data[1, 1] = kF
    # T = MCIntegration.Tau(β, β / 2.0, offset=1)
    T = MCIntegration.Continuous(
        0.0,
        β;
        grid=collect(LinRange(0.0, β, 1000)),
        offset=1,
        alpha=alpha,
    )
    T.data[1] = 0.0
    X = MCIntegration.Discrete(1, length(ngrid); alpha=alpha)
    ExtKidx = MCIntegration.Discrete(1, length(kgrid); alpha=alpha)

    dof = [[p.innerLoopNum, p.totalTauNum - 1, 1, 1] for p in diagpara] # K, T, ExtKidx
    # observable of sigma diagram of different permutations
    obs = [zeros(ComplexF64, length(ngrid), length(kgrid)) for o in 1:length(dof)]

    if isnothing(neighbor)
        neighbor = UEG.neighbor(partition)
    end
    if isnothing(config)
        config = Configuration(;
            var=(K, T, X, ExtKidx),
            dof=dof,
            type=ComplexF64, # type of the integrand
            obs=obs,
            neighbor=neighbor,
            # reweight_goal=reweight_goal,
            userdata=(para, diag, extT, kgrid, ngrid),
            kwargs...,
        )
    end

    result = integrate(
        integrandKW;
        solver=:mcmc,
        config=config,
        neval=neval,
        print=print,
        reweight_goal=reweight_goal,
        measure=Sigma.measureKW,
        kwargs...,
    )

    if isnothing(result) == false
        if print >= 0
            report(result.config)
            println(report(result; pick=o -> first(o)))
            println(result)
        end

        if print >= -2
            println(result)
        end

        # datadict = Dict{eltype(partition),Complex{Measurement{Float64}}}()
        datadict = Dict{eltype(partition),Any}()

        for o in 1:length(dof)
            avg, std = result.mean[o], result.stdev[o]
            r = measurement.(real(avg), real(std))
            i = measurement.(imag(avg), imag(std))
            data = Complex.(r, i)
            datadict[partition[o]] = data
        end
        return datadict, result
    else
        return nothing, nothing
    end
end

function main()
    # Physical params matching data for SOSEM observables
    order = [4]  # C^{(1)}_{N≤5} includes CTs up to 3rd order

    # Grid-search 1: rs, mass2
    # rs = LinRange(0.1, 2.0, 5)
    rs = [1.0]
    #mass2 = LinRange(1.0, 5.0, 5)
    mass2 = [1.0]
    beta = [40.0]

    # Grid-search 2: rs, beta
    #rs = LinRange(0.1, 2.0, 5)
    #mass2 = [<lambda_opt>]
    #beta = [20.0, 40.0, 100.0]

    # Post-search: finer, wider rs mesh at chosen beta and mass2
    #rs = LinRange(0.1, 3.0, 21)
    #mass2 = [<lambda_opt>]
    #beta = [<beta_opt>]

    # Total number of MCMC evaluations
    neval = 1e10

    # Get self-energy data needed for the chemical potential and Z-factor measurements
    for (_rs, _mass2, _beta, _order) in Iterators.product(rs, mass2, beta, order)
        para = UEG.ParaMC(; order=_order, rs=_rs, beta=_beta, mass2=_mass2, isDynamic=false)
        kF = para.kF

        ######### calcualte Z factor ######################
        kgrid = [kF]
        ngrid = [0, 1]
        # ngrid = [-1, 0]

        # Build diagrams
        partition = UEG.partition(_order)
        neighbor = UEG.neighbor(partition)
        @time diagram = Sigma.diagram(para, partition)

        #! format: off
        reweight_goal = [
            1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ]
        #! format: on

        # Integrate
        # sigma, result = KW(
        sigma, result = Sigma.KW(
            para,
            diagram;
            kgrid=kgrid,
            ngrid=ngrid,
            neval=neval,
            print=0,
            alpha=3.0,
            neighbor=neighbor,
            reweight_goal=reweight_goal[1:(length(partition) + 1)],
        )

        # Save data to JLD2
        if isnothing(sigma) == false
            println("Saving data to JLD2...")
            jldopen("data_Z.jld2", "a+"; compress=true) do f
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
