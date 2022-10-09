"""Integration test for bare (O(V²)), uniform (k = 0) SOSEM observables."""
function bare_integral_k0(;
    observable::DiagGen.Observables,
    beta=200.0,
    alpha=2.0,
    neval=5e5,
    print=-1,
    zscore_window=5,
    solver=:vegasmc,
)
    # Settings for diagram generation
    settings = DiagGen.Settings(;
        observable=observable,
        n_order=2,
        verbosity=DiagGen.quiet,
        expand_bare_interactions=false,
    )

    # UEG parameters for MC integration
    param = ParaMC(; order=settings.n_order, rs=2.0, beta=beta, isDynamic=false)

    # Generate the diagrams
    diagparam, diagtree, exprtree = DiagGen.build_nonlocal(settings)
    DiagGen.checktree(diagtree, settings)
    @test length(exprtree.root) == 1

    # Loop over external momenta and integrate
    res = UEG_MC.integrate_nonlocal(
        settings,
        param,
        diagparam,
        exprtree;
        kgrid=[0.0],
        alpha=alpha,
        neval=neval,
        print=print,
        solver=solver,
    )

    # Get exact uniform value for this SOSEM observable
    exact = DiagGen.get_exact_k0(observable)

    # Test standard score (z-score) of the measurement
    meas = measurement(res.mean[1], res.stdev[1])
    score = stdscore(meas, exact)
    obsstring = DiagGen.bare_string(observable)

    # Result should be accurate to within the specified standard score (by default, 5σ)
    println("""
            $obsstring ($solver):
             • Exact: $exact
             • Measured: $meas
             • Standard score: $score
            """)
    return abs(score) <= zscore_window
end

@testset verbose = true "Integration" begin
    test_solvers = [:vegas]
    # test_solvers = [:vegas, :vegasmc]
    @testset "C₂⁽¹ᵇ⁾ᴸ" begin
        for solver in test_solvers
            @test_broken bare_integral_k0(observable=DiagGen.c1bL0, solver=solver)
        end
    end
    @testset "C₂⁽¹ᶜ⁾" begin
        for solver in test_solvers
            @test bare_integral_k0(observable=DiagGen.c1c, solver=solver)
        end
    end
    @testset "C₂⁽¹ᵈ⁾" begin
        for solver in test_solvers
            @test_broken bare_integral_k0(observable=DiagGen.c1d, solver=solver)
        end
    end
end