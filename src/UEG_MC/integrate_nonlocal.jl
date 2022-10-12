using ..DiagGen
using ..UEG_MC

function integrate_nonlocal_with_ct(
    settings::Settings,
    mcparam::UEG.ParaMC,
    diagparams::Vector{DiagParaF64},
    exprtrees::Vector{ExprTreeF64};
    kgrid=[0.0],
    alpha=3.0,
    neval=1e5,
    print=-1,
    solver=:vegasmc,
)
    # Get necessary DiagGen properties from settings (the observable sign and
    # discontinuity side are the same ∀ partitions, but extT depend on n_loop!)
    dummy_cfg = Config(settings)
    discont_side, obs_sign, extT_sign =
        dummy_cfg.discont_side, dummy_cfg.obs_sign, dummy_cfg.extT_sign


    # We assume that each partition expression tree has a single root
    @assert all(length(et.root) == 1 for et in exprtrees)

    # List of expression tree roots, external times, and inner
    # loop numbers for each tree (to be passed to integrand)
    roots = [et.root[1] for et in exprtrees]
    innerLoopNums = [p.innerLoopNum for p in diagparams]
    # Get tree-dependent external time values directly from expression trees
    extTs = [exprtrees[i].node.object[r].para.extT for (i, r) in enumerate(roots)]

    @debug "Discontinuity side: $discont_side"
    @debug "Observable sign: $obs_sign"
    @debug "sgn(τout = ±δ) = $extT_sign"

    # Grid size
    n_kgrid = length(kgrid)

    # Temporary array for combined K-variables [ExtK, K].
    # We use the maximum necessary loop basis size for K pool.
    maxloops = maximum(p.totalLoopNum for p in diagparams)
    varK = zeros(3, maxloops)

    # Build adaptable MC integration variables
    (K, T, ExtKidx) = mc_variables(mcparam, n_kgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T), shape(ExtKidx)
    # We do not integrate the two external times, hence n_τ = totalTauNum - 2
    dof = [[p.innerLoopNum, p.totalTauNum - 2, 1] for p in diagparams]

    # UEG SOSEM diagram observables are a function of |k| only (equal-time)
    obs = repeat([zeros(n_kgrid)], length(dof))  # observable for each partition

    # Add extT to MC parameters
    # updated_mcparam = reconstruct(mcparam; additional=cfg.extT)

    # External times are fixed for left/right measurement of the discontinuity at τ = 0
    T.data[1] = 0
    T.data[2] = extT_sign * 1e-6    # τout = ±δ, depending on the SOSEM observable

    @debug "External time indices = $extTs"
    @debug "External times = $(T.data[[1,2]])"
    @debug begin
        extT_maps = [(2, extT[2]) => (extT[2], 2) for extT in extTs]
        "Remapping $extT_maps in G evals"
    end

    return integrate(
        integrand;
        solver=solver,
        measure=measure,
        neval=neval,
        print=print,
        # Config kwargs
        # userdata=(updated_mcparam, exprtrees, innerLoopNums, extTs, varK, kgrid, obs_sign),
        userdata=(mcparam, exprtrees, innerLoopNums, extTs, varK, kgrid, obs_sign),
        var=(K, T, ExtKidx),
        dof=dof,
        obs=obs,
    )
end

function integrate_nonlocal(
    settings::Settings,
    mcparam::UEG.ParaMC,
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

    @debug "Discontinuity side: $(cfg.discont_side)"
    @debug "Observable sign: $(cfg.obs_sign)"
    @debug "sgn(τout = ±δ) = $extT_sign"

    # Grid size
    n_kgrid = length(kgrid)

    # Temporary array for combined K-variables [ExtK, K]
    varK = zeros(3, diagparam.totalLoopNum)

    # Build adaptable MC integration variables
    (K, T, ExtKidx) = mc_variables(mcparam, n_kgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T), shape(ExtKidx)
    # We do not integrate the two external times, hence n_τ = totalTauNum - 2
    dof = [[diagparam.innerLoopNum, diagparam.totalTauNum - 2, 1]]

    # UEG SOSEM diagram observables are a function of |k| only (equal-time)
    obs = [zeros(n_kgrid)]

    # Add extT to MC parameters
    # updated_mcparam = reconstruct(mcparam; additional=cfg.extT)

    # External times are fixed for left/right measurement of the discontinuity at τ = 0
    T.data[1] = 0
    T.data[2] = extT_sign * 1e-6    # τout = ±δ, depending on the SOSEM observable

    @debug "External time indices = $(cfg.extT)"
    @debug "External times = $(T.data[[1,2]])"
    @debug "Remapping $((2, extT[2]) => (extT[2], 2)) in G eval"

    return integrate(
        integrand;
        solver=solver,
        measure=measure_single,
        neval=neval,
        print=print,
        # Config kwargs
        userdata=(mcparam, [exprtree], [innerLoopNum], [cfg.extT], varK, kgrid, obs_sign),
        var=(K, T, ExtKidx),
        dof=dof,
        obs=obs,
    )
end

function mc_variables(mcparam::UEG.ParaMC, n_kgrid::Int, alpha::Float64)
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Offset T pool by 2 for fixed external times (τin, τout)
    T = Continuous(0.0, mcparam.β; offset=2, alpha=alpha)
    # Bin in external momentum
    ExtKidx = Discrete(1, n_kgrid; alpha=alpha)
    return (K, T, ExtKidx)
end

"""Measurement for a single diagram tree (without CTs, fixed order in V)."""
@inline function measure_single(vars, obs, weights, config)
    # ExtK bin index
    ik = vars[3][1]
    obs[1][ik] += weights[1]
    return
end

"""Measurement for multiple diagram trees (with CTs, fixed order in ξ)."""
function measure(vars, obs, weights, config)
    # ExtK bin index
    ik = vars[3][1]
    # Measure the weight of each partition
    for o in 1:(config.N)
        obs[o][ik] += weights[o]
    end
    return
end

function integrand(vars, config)
    # We sample internal momentum/times, and external momentum index
    K, T, ExtKidx = vars
    R, Theta, Phi = K

    # Unpack userdata
    mcparam, exprtrees, innerLoopNums, extTs, varK, kgrid, obs_sign = config.userdata

    # Evaluate the integrand for each partition
    integrand = Vector(undef, config.N)
    for i in 1:(config.N)
        # External momentum via random index into kgrid (wlog, we place it along the x-axis)
        ik = ExtKidx[1]
        varK[1, 1] = kgrid[ik]

        # @assert innerLoopNum == config.dof[1][1]
        phifactor = 1.0
        for j in 1:innerLoopNums[i]
            # r = tan(π * R[j] / 2)  # a similar mapping with smaller tail
            r = R[j] / (1 - R[j])
            θ = Theta[j]
            ϕ = Phi[j]
            varK[1, j + 1] = r * sin(θ) * cos(ϕ)
            varK[2, j + 1] = r * sin(θ) * sin(ϕ)
            varK[3, j + 1] = r * cos(θ)
            # phifactor *= r^2 * sin(θ) * (π / 2) * sec(π * R[i] / 2)^2
            phifactor *= r^2 * sin(θ) / (1 - R[j])^2
        end
        @assert (T.data[1] == 0) && (abs(T.data[2]) == 1e-6)

        # Evaluate the expression tree
        ExprTree.evalKT!(
            exprtrees[i],
            varK,
            T.data,
            (mcparam, extTs[i]); # = additional
            eval=UEG_MC.Propagators.eval,
        )
        weight = exprtrees[i].node.current

        # Phase-space and Jacobian factor
        factor = 1.0 / (2π)^(mcparam.dim * innerLoopNums[i]) * phifactor

        # Non-dimensionalized integrand
        # NOTE: C⁽¹⁾ = Σ(τ = 0⁻) - Σ(τ = 0⁺), so there is an additional
        #       overall sign contribution depending on SOSEM observable
        eTF = mcparam.qTF^2 / (2 * mcparam.me) # Thomas-Fermi energy
        # return obs_sign * factor * weight[exprtrees[i].root[1]] / eTF^2
        integrand[i] = sum(obs_sign * factor * weight[r] / eTF^2 for r in exprtrees[i].root)
    end
    return integrand
end