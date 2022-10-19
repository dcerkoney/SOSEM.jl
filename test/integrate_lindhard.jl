"""Dimensionless Lindhard function F(x)."""
function lindhard(x, taylor_expand, epsilon=1e-7)
    # Exact limits at 0 and 1
    if x == 0
        return 1
    elseif x == 1
        return 1 / 2
    end
    if taylor_expand && x > 1.0 / epsilon
        # Taylor expansion for large x
        r = 1 / x
        return r^2 / 6 + r^4 / 30 + r^6 / 70
    else
        return 1 / 2 + ((1 - x^2) / (4x)) * log(abs((1 + x) / (1 - x)))
    end
end

"""
Integrand for F(x[R]).
Maps the improper integral of F(x) on [0, ∞) to the unit interval [0, 1).
"""
function integrand(R, c, taylor_expand)
    x = R[1] / (1 - R[1])
    jacobian_x_R = 1.0 / (1 - R[1])^2
    return jacobian_x_R * lindhard(x, taylor_expand)
end

"""Integrate F(x[R]), parameterized in terms of R ∈ [0, 1)."""
function integrate_lindhard(;
    taylor_expand,
    alpha=2.0,
    neval=1e7,
    mcprint=-1,
    zscore_window=5,
    solver=:vegas,
)
    # Build pool for R ∈ [0, 1) and integrate F(x[R])
    R = Continuous(0.0, 1.0; alpha=alpha)
    local res
    try
        # Produces an error when taylor_expand == false
        res = integrate(
            (R, c) -> integrand(R, c, taylor_expand);
            neval=neval,
            print=mcprint,
            solver=solver,
        )
    catch errmsg
        rethrow(errmsg)
    end

    # Test the z-score of the MC results
    exact = π^2 / 8
    meas = measurement(res.mean, res.stdev)
    score = stdscore(meas, exact)

    # Result should be accurate to within the specified standard score (by default, 5σ)
    if mcprint > -2
        print("""
              ∫dxF(x) from 0 to ∞ ($solver):
               • Exact value (= π²/8): $exact
               • Measured value      : $meas
               • Standard score      : $score
              """)
    end
    return abs(score) ≤ zscore_window
end

@testset "Lindhard function improper integration" begin
    test_solvers = [:vegas, :vegasmc]
    for solver in test_solvers
        # vegas/vegasmc solvers break down due to tail round-off error
        @test_throws str -> any(
            occursin(s, str) for s in [
                r"normalization of block \d+ is NaN, which is not positively defined!",
                "AssertionError: histogram should be all finite",
            ]
        ) integrate_lindhard(taylor_expand=false, mcprint=-2, solver=solver)

        # Taylor expansion of F(x) for large x resolves this issue
        mcprint = (solver == :vegas) ? -1 : -2
        @test integrate_lindhard(taylor_expand=true, mcprint=mcprint, solver=solver)
    end
end
