"""
Same as CounterTerm.chemicalpotential_renormalization, but with lowest loop 
orders increased by 1 everywhere (SOSEM observables start at 2nd loop order).
"""
function chemicalpotential_renormalization(data, δμ; lowest_order=2, min_order=2, max_order)
    @assert max_order <= 5 "Order $order hasn't been implemented!"
    println(δμ)
    @assert length(δμ) >= max_order - lowest_order
    data = CounterTerm.mergeInteraction(data)
    d = data
    # To maximum supported counterterm order, z = [C2, C3, C4, C5]
    z = Vector{eltype(values(d))}(undef, max_order)
    if min_order ≤ 2 ≤ max_order
        #    Σ1 = Σ10
        # => C2 = C20
        z[1] = d[(2, 0)]
    end
    if min_order ≤ 3 ≤ max_order
        #    Σ2 = Σ20 + Σ11*δμ1
        # => C3 = C30 + C21*δμ1
        z[2] = d[(3, 0)] + δμ[1] * d[(2, 1)]
    end
    if min_order ≤ 4 ≤ max_order
        #    Σ3 = Σ30 + Σ21*δμ1 + Σ12*δμ1^2 + Σ11*δμ2
        # => C4 = C40 + C31*δμ1 + C22*δμ1^2 + C21*δμ2 
        z[3] = d[(4, 0)] + δμ[1] * d[(3, 1)] + δμ[1]^2 * d[(2, 2)] + δμ[2] * d[(2, 1)]
    end
    if min_order ≤ 5 ≤ max_order
        #    Σ4 = Σ40 + Σ31*δμ1 + Σ22*δμ1^2 + Σ21*δμ2 + Σ13*δμ1^3 + Σ12*(2*δμ1*δμ2) + Σ11*δμ3
        # => C5 = C50 + C41*δμ1 + C32*δμ1^2 + C31*δμ2 + C23*δμ1^3 + C22*(2*δμ1*δμ2) + C21*δμ3
        #! format: off
        z[4] = d[(5, 0)] + δμ[1] * d[(4, 1)] + δμ[1]^2 * d[(3, 2)] + δμ[2] * d[(3, 1)] +
               (δμ[1])^3 * d[(2, 3)] + 2 * δμ[1] * δμ[2] * d[(2, 2)] + δμ[3] * d[(2, 1)]
        #! format: on
    end
    return z
end
function chemicalpotential_renormalization(order, data, δμ)
    return chemicalpotential_renormalization(data, δμ; max_order=order)
end

"""
Computes the exact value for the lowest-order chemical 
potential renormalization δμ₁ = ReΣ₁[λ](kF, 0). Note
that using N&O convention, there is an extra overall
minus sign.
"""
function delta_mu1(param::UEG.ParaMC)
    # Dimensionless wavenumber at the Fermi surface (x = k / kF)
    x = 1
    # Dimensionless Yukawa mass squared (lambda = λ / kF²)
    lambda = param.mass2 / param.kF^2
    # Dimensionless screened Lindhard function
    F_x_lambda = UEG_MC.screened_lindhard(x; lambda=lambda)
    # δμ₁ cancels the real part of the Fock self-energy
    # at the Fermi surface for a Yukawa-screened UEG.
    return -(param.e0^2 * param.kF / (2 * pi^2 * param.ϵ0)) * F_x_lambda
end
