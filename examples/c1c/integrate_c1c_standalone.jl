using SOSEM: DiagGen, UEG_MC
using ElectronLiquid.UEG: ParaMC

# Debug mode
if isinteractive()
    ENV["JULIA_DEBUG"] = Main
end

para = ParaMC(; rs=5.0, isDynamic=false, beta=40.0, mass2=0.0001)

# Generate the diagrams
settings = DiagGen.Settings(; observable=DiagGen.c1c, n_order=2, verbosity=DiagGen.info)
const diagparam, diag, compiled_diag = DiagGen.build_nonlocal(settings);
DiagGen.checktree(diag, settings)
# DiagGen.checktree(diag, settings; plot=true, maxdepth=10)

const kgrid = [para.kF]
const varK = zeros(3, diag.id.para.totalLoopNum)
const alpha = 3.0
const neval = 1e5

function integrand(vars, config)
    idx = 1
    # R, Theta, Phi, varT, N, ExtKidx = vars
    K, T, ExtKidx = vars
    R, Theta, Phi = K
    # para, diag, extT, kgrid, ngrid, varK = config.userdata
    diagram = compiled_diag
    weight = diagram.node.current
    loopNum = config.dof[idx][1]
    # l = N[1]
    k = ExtKidx[1]
    varK[1, 1] = kgrid[k]
    # wn = ngrid[l]

    phifactor = 1.0
    for i in 1:loopNum
        r = R[i] / (1 - R[i])
        θ = Theta[i]
        ϕ = Phi[i]
        # varK[:, i+1] .= [r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ)]
        varK[1, i + 1] = r * sin(θ) * cos(ϕ)
        varK[2, i + 1] = r * sin(θ) * sin(ϕ)
        varK[3, i + 1] = r * cos(θ)
        phifactor *= r^2 * sin(θ) / (1 - R[i])^2
    end

    T.data[1] = 0
    T.data[2] = -1e-6
    # varT[3:3 + dof[1,2]]

    ExprTree.evalKT!(diagram, varK, T.data, para; eval=UEG_MC.Propagators.eval)
    factor = 1.0 / (2π)^(para.dim * loopNum) * phifactor

    # Dimensionless form
    return weight[diagram.root[1]] * factor / para.qTF^4
end

function measure(vars, obs, weights, config)
    k = vars[3][1]  #K
    for o in 1:(config.N)
        obs[o][k] += weights[o]
    end
end

R = Continuous(0.0, 1.0; alpha=alpha)
Theta = Continuous(0.0, 1π; alpha=alpha)
Phi = Continuous(0.0, 2π; alpha=alpha)
K = CompositeVar(R, Theta, Phi)
T = Continuous(0.0, para.β; offset=2, alpha=alpha)
# X = Discrete(1, length(ngrid), alpha=alpha)
ExtKidx = Discrete(1, length(kgrid); alpha=alpha)

dof = [[diag.id.para.innerLoopNum, diag.id.para.totalTauNum - 2, 1]] # K, T, ExtKidx

# observable of sigma diagram of different permutations
obs = [zeros(length(kgrid))]

res = integrate(
    integrand;
    measure=measure,
    var=(K, T, ExtKidx),
    dof=dof,
    obs=obs,
    neval=1e5,
    print=0,
)
