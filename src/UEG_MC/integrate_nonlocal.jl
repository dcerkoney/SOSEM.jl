using ..DiagGen
using ..UEG_MC

function integrate_nonlocal(
    cfg::DiagGen.Config,
    mc_params::UEG.ParaMC,
    diag_tree::Diagram,
    expr_tree::ExpressionTree;
    kgrid=[0.0],
    alpha=3.0,
    neval=1e5,
    print=-1,
    is_main_thread=true,
)
    # Pass innerLoopNum to integrand
    innerLoopNum = diag_tree.id.para.innerLoopNum

    # Pass observable and external time signs to integrand
    obs_sign = cfg.obs_sign
    extT_sign = cfg.extT_sign

    if is_main_thread
        @debug "Discontinuity side: $(cfg.discont_side)" maxlog = 1
        @debug "Observable sign: $(cfg.obs_sign)" maxlog = 1
        @debug "External time sign: $(cfg.extT_sign)" maxlog = 1
    end

    # Grid size
    n_kgrid = length(kgrid)

    # Temporary array for combined K-variables [ExtK, K]
    varK = zeros(3, diag_tree.id.para.totalLoopNum)

    # Build adaptable MC integration variables
    (K, T, ExtKidx) = mc_variables(mc_params, n_kgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T), shape(ExtKidx)
    # We do not integrate the two external times, hence n_τ = totalTauNum - 2
    dof = [[diag_tree.id.para.innerLoopNum, diag_tree.id.para.totalTauNum - 2, 1]]

    # UEG SOSEM diagram observables are a function of |k| only (equal-time)
    obs = [zeros(n_kgrid)]

    # Get external time indices for this observable---we will remap them to
    # (1, 2) in the evaluation step to correspond with T-pool offset (= 2)
    extT = diag_tree.id.extT
    # Add extT to MC parameters
    params = reconstruct(mc_params; additional=extT)

    # External times are fixed for left/right measurement of the discontinuity at τ = 0
    T.data[1] = 0
    T.data[2] = extT_sign * 1e-6    # τout = ±δ, depending on the SOSEM observable

    if is_main_thread
        @debug "sgn(τout = ±δ) = $extT_sign" maxlog = 1
        @debug "External time indices = $extT" maxlog = 1
        @debug "External times = $(T.data[[1,2]])" maxlog = 1
        @debug "Remapping $((2, extT[2])) ↦ $((extT[2], 2)) in G eval" maxlog = 1
    end

    return integrate(
        integrand;
        solver=:vegas,
        measure=measure,
        neval=neval,
        print=print,
        # Config kwargs
        userdata=(params, expr_tree, innerLoopNum, obs_sign, kgrid, varK),
        var=(K, T, ExtKidx),
        dof=dof,
        obs=obs,
    )
end

function mc_variables(params::UEG.ParaMC, n_kgrid::Int, alpha::Float64)
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Offset T pool by 2 for fixed external times (τin, τout)
    T = Continuous(0.0, params.β; offset=2, alpha=alpha)
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
    mc_params, expr_tree, innerLoopNum, obs_sign, kgrid, varK = config.userdata

    # Get weight from expression tree
    weight = expr_tree.node.current

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
        # varK[:, i+1] .= [r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ)]
        varK[1, i + 1] = r * sin(θ) * cos(ϕ)
        varK[2, i + 1] = r * sin(θ) * sin(ϕ)
        varK[3, i + 1] = r * cos(θ)
        phifactor *= r^2 * sin(θ) / (1 - R[i])^2
    end
    # @assert (T.data[1] == 0) && (abs(T.data[2]) == 1e-6)

    # Evaluate the expression tree
    ExprTree.evalKT!(expr_tree, varK, T.data, mc_params; eval=UEG_MC.Propagators.eval)
    factor = 1.0 / (2π)^(mc_params.dim * innerLoopNum) * phifactor

    # Non-dimensionalized integrand
    # NOTE: C⁽¹⁾ = Σ(τ = 0⁻) - Σ(τ = 0⁺), so there is an additional
    #       overall sign contribution depending on SOSEM observable
    return obs_sign * weight[expr_tree.root[1]] * factor / (mc_params.kF^2 * mc_params.e0^4)
    # return obs_sign * weight[expr_tree.root[1]] * factor / mc_params.qTF^4
end