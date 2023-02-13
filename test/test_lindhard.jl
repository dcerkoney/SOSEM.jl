"""
Dimensionless Lindhard function F(x) for the UEG, with optional Taylor expansion for testing.
Here x is a dimensionless wavenumber.
"""
function lindhard(x::T, taylor_expand; epsilon=1e-5) where {T<:Real}
    if UEG_MC.almostzero(x)
        return 1
    elseif x ≈ 1
        return 1 / 2
    elseif x < 0
        throw(DomainError(x))  # x should be a non-negative number
    end
    if taylor_expand && x > 1.0 / epsilon
        # Taylor expansion for large x
        r = 1 / x
        return r^2 / 3 + r^4 / 15 + r^6 / 35
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

@testset "Lindhard function" begin
    @test UEG_MC.lindhard(0) == 1
    @test UEG_MC.lindhard(1e-8) == 1
    @test UEG_MC.lindhard(1) == 1 / 2
    @test UEG_MC.lindhard(1 + 1e-8) == 1 / 2
end

@testset "Yukawa-screened Lindhard function" begin
    @test UEG_MC.screened_lindhard(0; lambda=eps(Float64) / 10) ≈ UEG_MC.lindhard(0)
    @test UEG_MC.screened_lindhard(1; lambda=eps(Float64) / 10) ≈ UEG_MC.lindhard(1)
    @test UEG_MC.screened_lindhard(π; lambda=eps(Float64) / 10) ≈ UEG_MC.lindhard(π)
end

@testset "Lindhard function improper integration" begin
    # vegas/vegasmc solvers break down due to tail round-off error.
    # Taylor expansion of F(x) for large x resolves this issue
    test_solvers = [:vegas, :vegasmc]
    # vegas/vegasmc solvers break down due to tail round-off error
    for solver in test_solvers
        mcprint = (solver == :vegas) ? -1 : -2
        @test integrate_lindhard(; taylor_expand=true, mcprint=mcprint, solver=solver)
    end
end
