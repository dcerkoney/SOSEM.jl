"""
Same as CounterTerm.chemicalpotential_renormalization, but with lowest loop 
orders increased by 1 everywhere (SOSEM observables start at (2,0,0))
"""
function chemicalpotential_renormalization(order, data, δμ)
    # _partition = sort([k for k in keys(rdata)])
    # println(_partition)
    @assert order <= 5 "Order $order hasn't been implemented!"
    println(δμ)
    @assert length(δμ) >= order - 2
    data = CounterTerm.mergeInteraction(data)
    d = data
    # println("size: ", size(d[(1, 0)]))
    z = Vector{eltype(values(d))}(undef, order)
    if order >= 2
        z[1] = d[(2, 0)]
    end
    if order >= 3
        z[2] = d[(3, 0)] + δμ[1] * d[(2, 1)]
    end
    if order >= 4
        # Σ3 = Σ30+Σ11*δμ2+Σ12*δμ1^2+Σ21*δμ1
        z[3] = d[(4, 0)] + δμ[1] * d[(3, 1)] + δμ[1]^2 * d[(2, 2)] + δμ[2] * d[(2, 1)]
    end
    if order >= 5
        # Σ4 = Σ40+Σ11*δμ3+Σ12*(2*δμ1*δμ2)+Σ13*δμ1^3+Σ21*δμ2+Σ22*δμ1^2+Σ31*δμ1
        z[4] = d[(5, 0)] + δμ[1] * d[(4, 1)] + δμ[1]^2 * d[(3, 2)] + δμ[2] * d[(3, 1)] + (δμ[1])^3 * d[(2, 3)] + 2 * δμ[1] * δμ[2] * d[(2, 2)] + δμ[3] * d[(2, 1)]
        # z[4] = d[(4, 0)] + δμ[2] * d[(2, 1)] + δμ[3] * d[(1, 1)]
    end
    return z
end
