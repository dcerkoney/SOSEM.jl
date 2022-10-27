"""
Defines a machine epsilon for a real subtype by falling 
back to infinite precision for non-float reals.
"""
fallback_epsilon(::Type{R}) where {R<:Real} = R <: AbstractFloat ? eps(R) : 0

"""
Returns whether a real number x is approximately zero modulo a machine 
epsilon which falls back to infinite precision for non-float reals.
"""
function almostzero(x::R) where {R<:Real}
    return isapprox(x, 0; atol=sqrt(fallback_epsilon(R)), rtol=0)
end

"""Dimensionless Lindhard function F(x) for the UEG. Here x is a dimensionless wavenumber."""
function lindhard(x::T; epsilon=1e-5) where {T<:Real}
    # Exact limits near 0 and 1
    if almostzero(x)
        return 1
    elseif x ≈ 1
        return 1 / 2
    elseif x < 0
        throw(DomainError(x))  # x should be a non-negative number
    end
    # Use Taylor expansion for large x, and exact function otherwise
    if x > 1.0 / epsilon
        r = 1 / x
        return r^2 / 3 + r^4 / 15 + r^6 / 35
    else
        return 1 / 2 + ((1 - x^2) / (4x)) * log(abs((1 + x) / (1 - x)))
    end
end

"""
Dimensionless Lindhard function F(x, lambda) for the screened UEG with Yukawa interaction V[λ].
Here x is a dimensionless wavenumber and lambda is the dimensionless Yukawa mass squared.
"""
function screened_lindhard(x::S; lambda::T, epsilon=1e-5) where {S,T<:Real}
    if lambda < 0
        # lambda should be a non-negative number
        throw(DomainError(lambda))
    elseif almostzero(lambda)
        return lindhard(x; epsilon=epsilon)
    else
        # Dimensionless Yukawa mass (lambda = m²)
        m = sqrt(lambda)
        # Exact limits near 0 and 1
        if almostzero(x)
            return 1 - m * atan(1 / m)
        elseif x ≈ 1
            return 1 / 2 - (m / 2) * atan(2 / m) + (lambda / 2) * atan(2 / (2 + lambda))
        elseif x < 0
            throw(DomainError(x))  # x should be a non-negative number
        end
        # Use Taylor expansion for large x, and exact function otherwise
        if x > 1.0 / epsilon
            r = 1 / x
            return r^2 / 3 +
                   r^4 * (1 / 15 - lambda / 3) +
                   r^6 * (1 / 35 - 2lambda / 5 + lambda^2 / 3)
        else
            return 1 / 2 +
                   (m / 2) * (atan((x - 1) / m) - atan((x + 1) / m)) +
                   ((1 + lambda - x^2) / (4x)) * atanh(2x / (1 + lambda + x^2))
        end
    end
end
