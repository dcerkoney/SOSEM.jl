function integrate_nonlocal(
    # settings::Settings,
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
    # cfg = Config(settings)

    # Pass innerLoopNum to integrand
    innerLoopNum = diagparam.innerLoopNum

    # Grid size
    n_kgrid = length(kgrid)

    # Temporary array for combined K-variables [ExtK, K]
    varK = zeros(3, diagparam.totalLoopNum)

    # Build adaptable MC integration variables
    (K, T, ExtKidx) = mc_variables(mcparam, n_kgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T), shape(ExtKidx)
    # We do not integrate the three external times, hence n_τ = totalTauNum - 3
    dof = [[diagparam.innerLoopNum, diagparam.totalTauNum - 3, 1]]

    # UEG SOSEM diagram observables are a function of |k| only (equal-time)
    obs = [zeros(n_kgrid)]

    # External times are fixed for left/right measurement of the discontinuity at τ = 0
    T.data[1] = -1e-6  # τout = -δ, for observables with discont_side = negative
    T.data[2] = 1e-6   # τout = +δ, for observables with discont_side = positive
    T.data[3] = 0      # τin  = 0,  for all observables

    # Thomas-Fermi energy squared
    eTF2 = mcparam.qTF^4 / (2 * mcparam.me)^2

    # Phase-space factors
    phase_factor = 1.0 / (2π)^(mcparam.dim * innerLoopNum)

    # Total prefactors
    prefactor = -phase_factor / eTF2

    return integrate(
        integrand_single;
        solver=solver,
        measure=measure_single,
        neval=neval,
        print=print,
        # MC config kwargs
        userdata=(mcparam, [exprtree], [innerLoopNum], [prefactor], varK, kgrid),
        var=(K, T, ExtKidx),
        dof=dof,
        obs=obs,
    )
end

function integrate_nonlocal_with_ct(
    # settings::Settings,
    mcparam::UEG.ParaMC,
    diagparams::Vector{DiagParaF64},
    exprtrees::Vector{ExprTreeF64};
    kgrid=[0.0],
    alpha=3.0,
    neval=1e5,
    print=-1,
    solver=:vegasmc,
)
    # We assume that each partition expression tree has a single root
    @assert all(length(et.root) == 1 for et in exprtrees)

    # List of expression tree roots, external times, and inner
    # loop numbers for each tree (to be passed to integrand)
    # roots = [et.root[1] for et in exprtrees]
    innerLoopNums = [p.innerLoopNum for p in diagparams]

    # Grid size
    n_kgrid = length(kgrid)

    # Temporary array for combined K-variables [ExtK, K].
    # We use the maximum necessary loop basis size for K pool.
    maxloops = maximum(p.totalLoopNum for p in diagparams)
    varK = zeros(3, maxloops)

    # Build adaptable MC integration variables
    (K, T, ExtKidx) = mc_variables(mcparam, n_kgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T), shape(ExtKidx)
    # We do not integrate the three external times, hence n_τ = totalTauNum - 3
    dof = [[p.innerLoopNum, p.totalTauNum - 3, 1] for p in diagparams]

    # UEG SOSEM diagram observables are a function of |k| only (equal-time)
    obs = repeat([zeros(n_kgrid)], length(dof))  # observable for each partition

    # External times are fixed for left/right measurement of the discontinuity at τ = 0
    T.data[1] = -1e-6  # τout = -δ, for observables with discont_side = negative
    T.data[2] = 1e-6   # τout = +δ, for observables with discont_side = positive
    T.data[3] = 0      # τin  = 0,  for all observables

    # Thomas-Fermi energy squared
    eTF2 = mcparam.qTF^4 / (2 * mcparam.me)^2

    # Phase-space factors
    phase_factors = [1.0 / (2π)^(mcparam.dim * nl) for nl in innerLoopNums]

    # Total prefactors
    prefactors = -phase_factors / eTF2

    return integrate(
        integrand_single;
        solver=solver,
        measure=measure,
        neval=neval,
        print=print,
        # MC config kwargs
        userdata=(mcparam, exprtrees, innerLoopNums, prefactors, varK, kgrid),
        var=(K, T, ExtKidx),
        dof=dof,
        obs=obs,
    )
end

function integrate_full_nonlocal_with_ct(
    mcparam::UEG.ParaMC,
    diagparams::Vector{DiagParaF64},
    exprtrees::Vector{ExprTreeF64};
    kgrid=[0.0],
    alpha=3.0,
    neval=1e5,
    print=-1,
    solver=:vegasmc,
)
    # NOTE: We assume that each partition expression tree has two roots (1: c1c, 2: rest)
    @assert all(length(et.root) == 2 for et in exprtrees)

    # List of expression tree roots and inner loop numbers for each tree (to be passed to integrand)
    # roots = [et.root for et in exprtrees]
    innerLoopNums = [p.innerLoopNum for p in diagparams]

    # Grid size
    n_kgrid = length(kgrid)

    # Temporary array for combined K-variables [ExtK, K].
    # We use the maximum necessary loop basis size for K pool.
    maxloops = maximum(p.totalLoopNum for p in diagparams)
    varK = zeros(3, maxloops)

    # Build adaptable MC integration variables
    (K, T, ExtKidx) = mc_variables(mcparam, n_kgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(T), shape(ExtKidx)
    # We do not integrate the three external times, hence n_τ = totalTauNum - 3
    dof = [[p.innerLoopNum, p.totalTauNum - 3, 1] for p in diagparams]

    # UEG SOSEM diagram observables are a function of |k| only (equal-time)
    obs = repeat([zeros(n_kgrid)], length(dof))  # observable for each partition

    # Add extT to MC parameters
    # updated_mcparam = reconstruct(mcparam; additional=cfg.extT)

    # External times are fixed for left/right measurement of the discontinuity at τ = 0
    T.data[1] = -1e-6  # τout = -δ, for observables with discont_side = negative
    T.data[2] = 1e-6   # τout = +δ, for observables with discont_side = positive
    T.data[3] = 0      # τin  = 0,  for all observables

    # Thomas-Fermi energy squared
    eTF2 = mcparam.qTF^4 / (2 * mcparam.me)^2

    # Phase-space factors
    phase_factors = [1.0 / (2π)^(mcparam.dim * nl) for nl in innerLoopNums]

    # Total prefactors
    prefactors = -phase_factors / eTF2

    return integrate(
        integrand_full;
        solver=solver,
        measure=measure,
        neval=neval,
        print=print,
        # MC config kwargs
        userdata=(mcparam, exprtrees, innerLoopNums, prefactors, varK, kgrid),
        var=(K, T, ExtKidx),
        dof=dof,
        obs=obs,
    )
end

function mc_variables(mcparam::UEG.ParaMC, n_kgrid::Int, alpha::Float64, nT_fixed::Int=3)
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Offset T pool by 3 for fixed external times (τout-, τout+, τin)
    T = Continuous(0.0, mcparam.β; offset=nT_fixed, alpha=alpha)
    # Bin in external momentum
    ExtKidx = Discrete(1, n_kgrid; alpha=alpha)
    return (K, T, ExtKidx)
end

"""Measurement for a single partition (without CTs, fixed order in V)."""
@inline function measure_single(vars, obs, weights, config)
    ik = vars[3][1]  # ExtK bin index
    obs[1][ik] += weights[1]
    return
end

"""Measurement for multiple (N) partitions (with CTs, fixed order in ξ)."""
function measure(vars, obs, weights, config)
    ik = vars[3][1]  # ExtK bin index
    # Measure the weight of each partition
    for o in 1:(config.N)
        obs[o][ik] += weights[o]
    end
    return
end

function integrand_single(vars, config)
    # We sample internal momentum/times, and external momentum index
    K, T, ExtKidx = vars
    R, Theta, Phi = K

    # Unpack userdata
    mcparam, exprtrees, innerLoopNums, prefactors, varK, kgrid = config.userdata

    # Evaluate the integrand for each partition
    integrand = Vector(undef, config.N)
    for i in 1:(config.N)
        # External momentum via random index into kgrid (wlog, we place it along the x-axis)
        varK[1, 1] = kgrid[ExtKidx[1]]  # ik = ExtKidx[1]

        phifactor = 1.0
        for j in 1:innerLoopNums[i]  # config.dof[i][1]
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
        # @assert (T.data[1] == 0) && (abs(T.data[2]) == 1e-6)

        # Evaluate the expression tree (additional = mcparam)
        ExprTree.evalKT!(exprtrees[i], varK, T.data, mcparam; eval=UEG_MC.Propagators.eval)
        # weight = exprtrees[i].node.current

        # Non-dimensionalized integrand
        # NOTE: C⁽¹⁾ = Σ(τ = 0⁻) - Σ(τ = 0⁺), so there is an additional
        #       overall sign contribution depending on SOSEM observable
        # return obs_sign * factor * weight[exprtrees[i].root[1]] / eTF^2
        root = exprtrees[i].root[1]
        weight = exprtrees[i].node.current
        integrand[i] = phifactor * prefactors[i] * weight[root]
    end
    return integrand
end

function integrand_full(vars, config)
    # We sample internal momentum/times, and external momentum index
    K, T, ExtKidx = vars
    R, Theta, Phi = K

    # Unpack userdata
    mcparam, exprtrees, innerLoopNums, prefactors, varK, kgrid = config.userdata

    # Evaluate the integrand for each partition
    integrand = Vector(undef, config.N)
    for i in 1:(config.N)
        # External momentum via random index into kgrid (wlog, we place it along the x-axis)
        # ik = ExtKidx[1]
        varK[1, 1] = kgrid[ExtKidx[1]]

        phifactor = 1.0
        for j in 1:innerLoopNums[i]  # config.dof[i][1]
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
        # @assert (T.data[1] == 0) && (abs(T.data[2]) == 1e-6)

        # Two roots: one for c1c, and one for the remaining observables (additional = mcparam)
        root1, root2 = exprtrees[i].root[1], exprtrees[i].root[2]

        # Evaluate the expression tree (additional = mcparam)
        ExprTree.evalKT!(exprtrees[i], varK, T.data, mcparam; eval=UEG_MC.Propagators.eval)

        # Non-dimensionalized integrand
        # NOTE: C⁽¹⁾ = Σ(τ = 0⁻) - Σ(τ = 0⁺), so there is an additional
        #       overall sign contribution depending on SOSEM observable
        # return obs_sign * factor * weight[exprtrees[i].root[1]] / eTF^2
        weight = exprtrees[i].node.current
        integrand[i] = phifactor * prefactors[i] * (weight[root1] + weight[root2])
    end
    return integrand
end
