"""Dimensionless Lindhard function F(x)."""
function lindhard(x, epsilon=1e-5)
    # Exact limits at 0 and 1
    if x ≈ 0
        return 1
    elseif x ≈ 1
        return 1 / 2
    end
    if x > 1.0 / epsilon
        # Taylor expansion for large x
        r = 1 / x
        return r^2 / 6 + r^4 / 30 + r^6 / 70
    else
        return 1 / 2 + ((1 - x^2) / (4x)) * log(abs((1 + x) / (1 - x)))
    end
end

"""Bare Coulomb interaction."""
function CoulombBareinstant(q, p::ParaMC)
    return KOinstant(q, p.e0, p.dim, 0.0, 0.0, p.kF)
end

"""Evaluate a bare Coulomb interaction line."""
function DiagTree.eval(id::BareInteractionId, K, _, varT, p::ParaMC)
    return CoulombBareinstant(norm(K), p)
end

"""Evaluate an instantaneous bare Green's function."""
function DiagTree.eval(id::BareGreenId, K, _, varT, p::ParaMC)
    β, me, μ, massratio = p.β, p.me, p.μ, p.massratio
    ϵ = norm(K)^2 / (2me * massratio) - μ
    # Overall sign difference relative to the Negle & Orland convention
    return -Spectral.kernelFermiT(-1e-8, ϵ, β)
end

"""Constructs diagram parameters for the bare Fock self-energy."""
function fock_param()
    # Instantaneous bare interaction (interactionTauNum = 1) 
    # => innerLoopNum = totalTauNum = 1
    return DiagParaF64(;
        type=SigmaDiag,
        hasTau=false,
        firstTauIdx=1,
        innerLoopNum=1,
        totalTauNum=1,
        filter=[NoHartree],
        interaction=[FeynmanDiagram.Interaction(ChargeCharge, Instant)],
    )
end

"""Build variable pools for the Fock MC integration."""
function fock_mc_variables(n_kgrid::Int, alpha::Float64)
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Bin in external momentum
    ExtKidx = Discrete(1, n_kgrid; alpha=alpha)
    return (K, ExtKidx)
end

"""Measurement for a single diagram tree (without CTs, fixed order in V)."""
function measure(vars, obs, weights, config)
    # ExtK bin index
    ik = vars[2][1]
    obs[1][ik] += weights[1]
    return
end

"""Integrand for the Fock self-energy non-dimensionalized by E²_{TF} ~ q⁴_{TF}."""
function integrand(vars, config)
    # Sample internal momentum (the Fock diagram is instantaneous)
    K, ExtKidx = vars
    R, Theta, Phi = K

    # Unpack userdata
    mcparam, exprtree, varK, kgrid = config.userdata

    # External momentum via random index into kgrid (wlog, we place it along the x-axis)
    ik = ExtKidx[1]
    varK[1, 1] = kgrid[ik]

    phifactor = 1.0
    innerLoopNum = config.dof[1][1]
    for i in 1:innerLoopNum
        r = R[i] / (1 - R[i])
        θ = Theta[i]
        ϕ = Phi[i]
        varK[1, i + 1] = r * sin(θ) * cos(ϕ)
        varK[2, i + 1] = r * sin(θ) * sin(ϕ)
        varK[3, i + 1] = r * cos(θ)
        phifactor *= r^2 * sin(θ) / (1 - R[i])^2
    end

    # Evaluate the expression tree (additional = mcparam)
    weight = exprtree.node.current
    ExprTree.evalKT!(exprtree, varK, [], mcparam)
    r = exprtree.root[1]  # only one root

    # Phase-space and Jacobian factor
    # NOTE: extra minus sign on self-energy definition!
    factor = 1.0 / (2π)^(mcparam.dim * innerLoopNum) * phifactor

    # Return the non-dimensionalized Fock integrand
    eTF = mcparam.qTF^2 / (2 * mcparam.me)  # Thomas-Fermi energy
    return factor * weight[r] / eTF
end

"""Test MC integration on the Fock self-energy Σₓ(k)."""
function fock_integral(;
    kgrid=[0.0],
    beta=200.0,
    alpha=2.0,
    neval=5e5,
    mcprint=-1,
    zscore_window=5,
    solver=:vegas,
    verbosity=DiagGen.quiet,
)
    @assert 0 in kgrid

    # UEG parameters for MC integration
    param = ParaMC(; order=1, rs=2.0, beta=beta, isDynamic=false)

    # Generate the diagram and expression trees
    diagparam = fock_param()
    fock_diagram = Parquet.sigma(diagparam; name=:Σₓ).diagram[1]
    exprtree = ExprTree.build([fock_diagram])
    @test length(exprtree.root) == 1

    # Grid size
    n_kgrid = length(kgrid)

    # Temporary array for combined K-variables [ExtK, K].
    # We use the maximum necessary loop basis size for K pool.
    varK = zeros(3, diagparam.totalLoopNum)

    # Build adaptable MC integration variables
    var = fock_mc_variables(n_kgrid, alpha)

    # MC configuration degrees of freedom (DOF): shape(K), shape(ExtKidx)
    dof = [[diagparam.innerLoopNum, 1]]

    # Observe over external k-point grid
    obs = [zeros(n_kgrid)]

    # Integrate
    res = integrate(
        integrand;
        solver=solver,
        measure=measure,
        neval=neval,
        print=mcprint,
        # MC config kwargs
        userdata=(param, exprtree, varK, kgrid),
        var=var,
        dof=dof,
        obs=obs,
    )

    # The nondimensionalized Fock self-energy is the negative Lindhard function
    exact = -lindhard.(kgrid / param.kF)

    # Check the MC results against the exact (non-dimensionalized)
    # Fock (exhange) self-energy: Σₓ(k) / E²_{TF} = -F(k / kF)
    meas = measurement.(res.mean, res.stdev)
    scores = stdscore.(meas, exact)
    score_k0 = scores[1]
    worst_score = argmax(abs, scores)

    # Result should be accurate to within the specified standard score (by default, 5σ)
    if mcprint > -2
        print("""
              Σₓ(k) ($solver):
               • Exact value    (k = 0): $(exact[1])
               • Measured value (k = 0): $(meas[1])
               • Standard score (k = 0): $score_k0
               • Worst standard score: $worst_score
              """)
    end
    return abs(worst_score) ≤ zscore_window
end

@testset "Fock self-energy integration" begin
    test_solvers = [:vegas, :vegasmc]
    kgrid = collect(LinRange(0, 3, 49))
    for solver in test_solvers
        mcprint = (solver == :vegasmc) ? -1 : -2
        @test fock_integral(; kgrid=kgrid, mcprint=mcprint, solver=solver)
    end
end
