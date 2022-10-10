using ..DiagGen
using ..UEG_MC

function integrate_nonlocal_with_ct(
    settings::Settings,
    param::UEG.ParaMC,
    partitions::Tuple{Int,Int,Int},
    diagparams::Vector{DiagParaF64},
    exprtrees::Vector{ExprTreeF64};
    kgrid=[0.0],
    alpha=3.0,
    neval=1e5,
    print=-1,
    solver=:vegasmc,
)
    @todo
end

function integrate_nonlocal(
    settings::Settings,
    param::UEG.ParaMC,
    diagparam::DiagParaF64,
    exprtree::ExprTreeF64;
    kgrid=[0.0],
    alpha=3.0,
    neval=1e5,
    print=-1,
    solver=:vegasmc,
)
    # DiagGen config from settings
    cfg = Config(settings)

    # Pass innerLoopNum to integrand
    innerLoopNum = diagparam.innerLoopNum

    # Pass observable and external time signs to integrand
    obs_sign = cfg.obs_sign
    extT_sign = cfg.extT_sign

    @debug "Discontinuity side: $(cfg.discont_side)" maxlog = 1
    @debug "Observable sign: $(cfg.obs_sign)" maxlog = 1
    @debug "External time sign: $(cfg.extT_sign)" maxlog = 1

    # Grid size
    n_kgrid = length(kgrid)

    # Temporary array for combined K-variables [ExtK, K]
    varK = zeros(3, diagparam.totalLoopNum)

    # Build adaptable MC integration variables
    (K, T, ExtKidx) = mc_variables(param, n_kgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T), shape(ExtKidx)
    # We do not integrate the two external times, hence n_τ = totalTauNum - 2
    dof = [[diagparam.innerLoopNum, diagparam.totalTauNum - 2, 1]]

    # UEG SOSEM diagram observables are a function of |k| only (equal-time)
    obs = [zeros(n_kgrid)]

    # Add extT to MC parameters
    updated_mcparam = reconstruct(param; additional=cfg.extT)

    # External times are fixed for left/right measurement of the discontinuity at τ = 0
    T.data[1] = 0
    T.data[2] = extT_sign * 1e-6    # τout = ±δ, depending on the SOSEM observable

    @debug "sgn(τout = ±δ) = $extT_sign" maxlog = 1
    @debug "External time indices = $(cfg.extT)" maxlog = 1
    @debug "External times = $(T.data[[1,2]])" maxlog = 1
    @debug "Remapping $((2, cfg.extT[2])) ↦ $((cfg.extT[2], 2)) in G eval" maxlog = 1

    return integrate(
        integrand;
        solver=solver,
        measure=measure,
        neval=neval,
        print=print,
        # Config kwargs
        userdata=(updated_mcparam, exprtree, innerLoopNum, obs_sign, kgrid, varK),
        var=(K, T, ExtKidx),
        dof=dof,
        obs=obs,
    )
end

function mc_variables(param::UEG.ParaMC, n_kgrid::Int, alpha::Float64)
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Offset T pool by 2 for fixed external times (τin, τout)
    T = Continuous(0.0, param.beta; offset=2, alpha=alpha)
    # Bin in external momentum
    ExtKidx = Discrete(1, n_kgrid; alpha=alpha)
    return (K, T, ExtKidx)
end

function measure(vars, obs, weights, config)
    # ExtK bin index
    ik = vars[3][1]
    return obs[1][ik] += weights[1]
    # @assert config.N == 1
    # for o in 1:(config.N)
    #     obs[o][ik] += weights[o]
    # end
end

function integrand(vars, config)
    # We sample internal momentum/times, and external momentum index
    K, T, ExtKidx = vars
    R, Theta, Phi = K

    # Unpack userdata
    param, exprtree, innerLoopNum, obs_sign, kgrid, varK = config.userdata

    # Get weight from expression tree
    weight = exprtree.node.current

    # External momentum via random index into (discrete) kgrid
    ik = ExtKidx[1]
    # @debug "ik = $ik" maxlog = 1

    # wlog, set ExtK along x-axis
    varK[1, 1] = kgrid[ik]

    # @assert innerLoopNum == config.dof[1][1]
    phifactor = 1.0
    for i in 1:innerLoopNum
        r = R[i] / (1 - R[i])
        θ = Theta[i]
        ϕ = Phi[i]
        varK[1, i + 1] = r * sin(θ) * cos(ϕ)
        varK[2, i + 1] = r * sin(θ) * sin(ϕ)
        varK[3, i + 1] = r * cos(θ)
        phifactor *= r^2 * sin(θ) / (1 - R[i])^2
    end
    # @assert (T.data[1] == 0) && (abs(T.data[2]) == 1e-6)

    # Evaluate the expression tree
    ExprTree.evalKT!(exprtree, varK, T.data, param; eval=UEG_MC.Propagators.eval)
    factor = 1.0 / (2π)^(param.dim * innerLoopNum) * phifactor

    # Non-dimensionalized integrand
    # NOTE: C⁽¹⁾ = Σ(τ = 0⁻) - Σ(τ = 0⁺), so there is an additional
    #       overall sign contribution depending on SOSEM observable
    eTF = param.qTF^2 / (2 * param.me) # Thomas-Fermi energy
    return obs_sign * weight[exprtree.root[1]] * factor / eTF^2
end