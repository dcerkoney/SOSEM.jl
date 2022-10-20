using ElectronLiquid
using JLD2

# Physical params matching data for SOSEM observables
order = [2]  # C^{(1)}_{N≤4} includes CTs up to 2nd order
rs = [2.0]
mass2 = [0.1]
beta = [200.0]

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
    reweight_goal = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0]

    # Integrate
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
        jldopen("data_Z.jld2", "a+") do f
            key = "$(UEG.short(para))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (para, ngrid, kgrid, sigma)
        end
    end
end
