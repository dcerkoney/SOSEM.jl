"""Integration test for bare (O(V²)), uniform (k = 0) SOSEM observables."""
function bare_integral_k0(;
    observable::DiagGen.Observable,
    beta=200.0,
    alpha=2.0,
    neval=5e5,
    mcprint=-1,
    zscore_window=6,
    solver=:vegasmc,
    verbosity=DiagGen.quiet,
)
    # Settings for diagram generation
    settings = DiagGen.Settings(;
        observable=observable,
        verbosity=verbosity,
        expand_bare_interactions=false,
    )

    # UEG parameters for MC integration
    param = ParaMC(; order=settings.max_order, rs=1.0, beta=beta, isDynamic=false)

    # Generate the diagrams for the implicitly fixed-order calculation

    # Test against explicit fixed-order calculation
    diagparam, diagtree, exprtree = DiagGen.build_nonlocal_fixed_order(settings)
    diagparam_v2, _, _ = DiagGen.build_nonlocal(settings)

    # Check that explicit/implicit fixed-order calculations 
    # both give the same generated diagram parameters
    @test diagparam == diagparam_v2[1]

    DiagGen.checktree(diagtree, settings)
    @test length(exprtree.root) == 1

    # Integrate the non-local moment
    res = UEG_MC.integrate_nonlocal(
        # settings,
        param,
        diagparam,
        exprtree;
        kgrid=[0.0],
        alpha=alpha,
        neval=neval,
        print=mcprint,
        solver=solver,
    )

    # Get exact uniform value for this SOSEM observable
    exact = DiagGen.get_exact_k0(observable)

    # Test standard score (z-score) of the measurement
    meas = measurement(res.mean[1], res.stdev[1])
    score = stdscore(meas, exact)
    obsstring = DiagGen.get_bare_string(observable)

    # Result should be accurate to within the specified standard score (by default, 5σ)
    if mcprint > -2
        print("""
              $obsstring ($solver):
               • Exact: $exact
               • Measured: $meas
               • Standard score: $score
              """)
    end
    return abs(score) ≤ zscore_window
end

"""
Integration test for bare (O(V²)), uniform (k = 0) SOSEM observable using
the multi-partition integrator intended for evaluation with counterterms.
"""
function bare_integral_k0_multipartition(;
    observable::DiagGen.Observable,
    beta=200.0,
    alpha=2.0,
    neval=5e5,
    mcprint=-1,
    zscore_window=6,
    solver=:vegasmc,
    verbosity=DiagGen.quiet,
)
    # Settings for diagram generation
    settings = DiagGen.Settings(;
        observable=observable,
        max_order=2,
        verbosity=verbosity,
        expand_bare_interactions=false,
    )

    # UEG parameters for MC integration
    param = ParaMC(; order=settings.max_order, rs=1.0, beta=beta, isDynamic=false)

    # Build diagram and expression trees for all loop and counterterm partitions
    partitions, diagparams, diagtrees, exprtrees =
        DiagGen.build_nonlocal_with_ct(settings; renorm_mu=false)

    @test partitions == [(2, 0, 0)]
    @test all(length(et.root) == 1 for et in exprtrees)

    # Integrate the non-local moment
    res = UEG_MC.integrate_nonlocal_with_ct(
        # settings,
        param,
        diagparams,
        exprtrees;
        kgrid=[0.0],
        alpha=alpha,
        neval=neval,
        print=mcprint,
        solver=solver,
    )

    # Get exact uniform value for this SOSEM observable
    exact = DiagGen.get_exact_k0(observable)

    # Test standard score (z-score) of the measurement
    meas = measurement(res.mean[1], res.stdev[1])
    score = stdscore(meas, exact)
    obsstring = DiagGen.get_bare_string(observable)

    # Result should be accurate to within the specified standard score (by default, 5σ)
    if mcprint > -2
        print("""
              $obsstring ($solver):
               • Exact: $exact
               • Measured: $meas
               • Standard score: $score
              """)
    end
    return abs(score) ≤ zscore_window
end

@testset verbose = true "Single partition integration" begin
    test_solvers = [:vegasmc]
    @testset "C₂⁽¹ᵇ⁾ᴸ" begin
        for solver in test_solvers
            @test_broken bare_integral_k0(; observable=DiagGen.c1bL0, solver=solver)
        end
    end
    @testset "C₂⁽¹ᶜ⁾" begin
        for solver in test_solvers
            @test bare_integral_k0(; observable=DiagGen.c1c, solver=solver)
        end
    end
    @testset "C₂⁽¹ᵈ⁾" begin
        for solver in test_solvers
            @test_broken bare_integral_k0(; observable=DiagGen.c1d, solver=solver)
        end
    end
end

@testset verbose = true "Multi-partition integration" begin
    test_solvers = [:vegasmc]
    @testset "C₂⁽¹ᵇ⁾ᴸ" begin
        for solver in test_solvers
            @test_broken bare_integral_k0_multipartition(;
                observable=DiagGen.c1bL0,
                solver=solver,
                # mcprint=-2,
            )
        end
    end
    @testset "C₂⁽¹ᶜ⁾" begin
        for solver in test_solvers
            @test bare_integral_k0_multipartition(;
                observable=DiagGen.c1c,
                solver=solver,
                # mcprint=-2,
            )
        end
    end
    @testset "C₂⁽¹ᵈ⁾" begin
        for solver in test_solvers
            @test_broken bare_integral_k0_multipartition(;
                observable=DiagGen.c1d,
                solver=solver,
                # mcprint=-2,
            )
        end
    end
end
