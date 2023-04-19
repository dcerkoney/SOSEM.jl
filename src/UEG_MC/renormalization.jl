function chemicalpotential_renormalization_poln(data, δμ; min_order=1, max_order)
    @assert max_order ≤ 4 "Order $max_order hasn't been implemented!"
    println(δμ)
    @assert length(δμ) + 1 ≥ max_order
    d = CounterTerm.mergeInteraction(data)
    # To maximum supported counterterm order, z = [O1, O2, O3, O4]
    T = valtype(d)
    z = RenormMeasType{T}()
    # Requires order 1
    if min_order ≤ 1 ≤ max_order
        z[1] = d[(1, 0)]
    end
    # Requires order 2
    if min_order ≤ 2 ≤ max_order
        # Π2 = Π20 + Π11*δμ1
        z[2] = d[(2, 0)] + δμ[1] * d[(1, 1)]
    end
    # Requires orders 2 and 3
    if min_order ≤ 3 ≤ max_order
        # Π3 = Π30 + Π21*δμ1 + Π12*δμ1^2 + Π11*δμ2
        z[3] = d[(3, 0)] + δμ[1] * d[(2, 1)] + δμ[1]^2 * d[(1, 2)] + δμ[2] * d[(1, 1)]
    end
    # Requires orders 2, 3, and 4
    if min_order ≤ 4 ≤ max_order
        # Π4 = Π40 + Π31*δμ1 + Π22*δμ1^2 + Π21*δμ2 + Π13*δμ1^3 + Π12*(2*δμ1*δμ2) + Π11*δμ3
        #! format: off
        z[4] = d[(4, 0)] + δμ[1] * d[(3, 1)] + δμ[1]^2 * d[(2, 2)] + δμ[2] * d[(2, 1)] +
               (δμ[1])^3 * d[(1, 3)] + 2 * δμ[1] * δμ[2] * d[(1, 2)] + δμ[3] * d[(1, 1)]
        #! format: on
    end
    return z
end

function chemicalpotential_renormalization_sigma(data, δμ; min_order=1, max_order)
    @assert max_order ≤ 4 "Order $max_order hasn't been implemented!"
    println(δμ)
    @assert length(δμ) + 1 ≥ max_order
    d = CounterTerm.mergeInteraction(data)
    # To maximum supported counterterm order, z = [O1, O2, O3, O4]
    T = valtype(d)
    z = RenormMeasType{T}()
    # Requires order 1
    if min_order ≤ 1 ≤ max_order
        z[1] = d[(1, 0)]
    end
    # Requires order 2
    if min_order ≤ 2 ≤ max_order
        # Σ2 = Σ20 + Σ11*δμ1
        z[2] = d[(2, 0)] + δμ[1] * d[(1, 1)]
    end
    # Requires orders 2 and 3
    if min_order ≤ 3 ≤ max_order
        # Σ3 = Σ30 + Σ21*δμ1 + Σ12*δμ1^2 + Σ11*δμ2
        z[3] = d[(3, 0)] + δμ[1] * d[(2, 1)] + δμ[1]^2 * d[(1, 2)] + δμ[2] * d[(1, 1)]
        # z[3] = d[(3, 0)] + δμ[1] * d[(2, 1)] + δμ[1]^2 * d[(1, 2)] + δμ[2] * d[(1, 1)]
    end
    # Requires orders 2, 3, and 4
    if min_order ≤ 4 ≤ max_order
        # Σ4 = Σ40 + Σ31*δμ1 + Σ22*δμ1^2 + Σ21*δμ2 + Σ13*δμ1^3 + Σ12*(2*δμ1*δμ2) + Σ11*δμ3
        #! format: off
        z[4] = d[(4, 0)] + δμ[1] * d[(3, 1)] + δμ[1]^2 * d[(2, 2)] + δμ[2] * d[(2, 1)] +
               (δμ[1])^3 * d[(1, 3)] + 2 * δμ[1] * δμ[2] * d[(1, 2)] + δμ[3] * d[(1, 1)]
        # z[4] = d[(4, 0)] + δμ[1] * d[(3, 1)] + δμ[1]^2 * d[(2, 2)] + δμ[2] * d[(2, 1)] +
        #        (δμ[1])^3 * d[(1, 3)] + 2 * δμ[1] * δμ[2] * d[(1, 2)] + δμ[3] * d[(1, 1)]
        #! format: on
    end
    # Requires orders 2, 3, 4, and 5
    if min_order ≤ 5 ≤ max_order
        # G4 = G40 + G31*δμ1 + G22*δμ1^2 + G13*δμ1^3 + G21*δμ2 + G12*(2*δμ1*δμ2) 
        #          + G04*δμ1^4 + G03*(2*δμ2*δμ1^2) + G02*(2*δμ3*δμ1) + G01*δμ4
        #! format: off
        z[5] = d[(5, 0)] + δμ[1] * d[(4, 1)] + δμ[1]^2 * d[(3, 2)] + δμ[1]^3 * d[(2, 3)] +
               δμ[2] * d[(3, 1)] + 2 * δμ[1] * δμ[2] * d[(2, 2)] + δμ[1]^4 * d[(1, 4)] +
               2 * δμ[2] * δμ[1]^2 * d[(1, 3)] + 2 * δμ[3] * δμ[1] * d[(2, 2)] + δμ[4] * d[(1, 1)]
        #! format: on
    end
    return z
end

function chemicalpotential_renormalization_green(data, δμ; min_order=0, max_order)
    @assert max_order ≤ 4 "Order $max_order hasn't been implemented!"
    println(δμ)
    @assert length(δμ) ≥ max_order
    d = CounterTerm.mergeInteraction(data)
    # To maximum supported counterterm order, z = [𝓞_0, 𝓞_1, 𝓞_2, 𝓞_3, 𝓞_4]
    T = valtype(d)
    z = RenormMeasType{T}()
    # Requires order 0
    if min_order ≤ 0 ≤ max_order
        z[0] = d[(0, 0)]
    end
    # Requires order 1
    if min_order ≤ 1 ≤ max_order
        # G1 = G10 + G01*δμ1
        z[1] = d[(1, 0)] + δμ[1] * d[(0, 1)]
    end
    # Requires orders 1 and 2
    if min_order ≤ 2 ≤ max_order
        # G2 = G20 + G11*δμ1 + G02*δμ1^2 + G01*δμ2
        z[2] = d[(2, 0)] + δμ[1] * d[(1, 1)] + δμ[1]^2 * d[(0, 2)] + δμ[2] * d[(0, 1)]
    end
    # Requires orders 1, 2, and 3
    if min_order ≤ 3 ≤ max_order
        # G3 = G30 + G21*δμ1 + G12*δμ1^2 + G11*δμ2 + G03*δμ1^3 + G02*(2*δμ1*δμ2) + G01*δμ3
        #! format: off
        z[3] = d[(3, 0)] + δμ[1] * d[(2, 1)] + δμ[1]^2 * d[(1, 2)] + δμ[2] * d[(1, 1)] +
               δμ[1]^3 * d[(0, 3)] + 2 * δμ[1] * δμ[2] * d[(0, 2)] + δμ[3] * d[(0, 1)]
        #! format: on
    end
    # Requires orders 1, 2, 3, and 4
    if min_order ≤ 4 ≤ max_order
        # G4 = G40 + G31*δμ1 + G22*δμ1^2 + G13*δμ1^3 + G21*δμ2 + G12*(2*δμ1*δμ2) 
        #          + G04*δμ1^4 + G03*(2*δμ2*δμ1^2) + G02*(2*δμ3*δμ1) + G01*δμ4
        #! format: off
        z[4] = d[(4, 0)] + δμ[1] * d[(3, 1)] + δμ[1]^2 * d[(2, 2)] + δμ[1]^3 * d[(1, 3)] +
               + δμ[2] * d[(2, 1)] + 2 * δμ[1] * δμ[2] * d[(1, 2)] + δμ[1]^4 * d[(0, 4)] +
               2 * δμ[2] * δμ[1]^2 * d[(0, 3)] + 2 * δμ[3] * δμ[1] * d[(0, 2)] + δμ[4] * d[(0, 1)]
        #! format: on
    end
    return z
end

"""
Same as chemicalpotential_renormalization, but with lowest loop 
orders increased by 1 everywhere (SOSEM observables start at 2nd loop order).
"""
function chemicalpotential_renormalization_sosem(
    data,
    δμ;
    lowest_order=2,
    min_order=2,
    max_order,
)
    @assert max_order ≤ 5 "Order $order hasn't been implemented!"
    println(δμ)
    @assert length(δμ) ≥ max_order - lowest_order
    d = CounterTerm.mergeInteraction(data)
    # If the lowest_order for this observable is greater than 2,
    # then all counterterms of partition type (2, n) are zero.
    if lowest_order > 2
        for n in 1:lowest_order
            d[(2, n)] = zero(d[(lowest_order, 0)])
        end
    end
    # To maximum supported counterterm order, z = [C2, C3, C4, C5]
    T = valtype(d)
    z = RenormMeasType{T}()
    # Requires order 2
    if min_order ≤ 2 ≤ max_order
        #    Σ1 = Σ10
        # => C2 = C20
        z[2] = d[(2, 0)]
    end
    # Requires order 3
    if min_order ≤ 3 ≤ max_order
        #    Σ2 = Σ20 + Σ11*δμ1
        # => C3 = C30 + C21*δμ1
        z[3] = d[(3, 0)] + δμ[1] * d[(2, 1)]
    end
    # Requires orders 3 and 4
    if min_order ≤ 4 ≤ max_order
        #    Σ3 = Σ30 + Σ21*δμ1 + Σ12*δμ1^2 + Σ11*δμ2
        # => C4 = C40 + C31*δμ1 + C22*δμ1^2 + C21*δμ2 
        z[4] = d[(4, 0)] + δμ[1] * d[(3, 1)] + δμ[1]^2 * d[(2, 2)] + δμ[2] * d[(2, 1)]
    end
    # Requires orders 3, 4, and 5
    if min_order ≤ 5 ≤ max_order
        #    Σ4 = Σ40 + Σ31*δμ1 + Σ22*δμ1^2 + Σ21*δμ2 + Σ13*δμ1^3 + Σ12*(2*δμ1*δμ2) + Σ11*δμ3
        # => C5 = C50 + C41*δμ1 + C32*δμ1^2 + C31*δμ2 + C23*δμ1^3 + C22*(2*δμ1*δμ2) + C21*δμ3
        #! format: off
        z[5] = d[(5, 0)] + δμ[1] * d[(4, 1)] + δμ[1]^2 * d[(3, 2)] + δμ[2] * d[(3, 1)] +
               (δμ[1])^3 * d[(2, 3)] + 2 * δμ[1] * δμ[2] * d[(2, 2)] + δμ[3] * d[(2, 1)]
        #! format: on
    end
    return z
end
function chemicalpotential_renormalization_sosem(order, data, δμ)
    return chemicalpotential_renormalization_sosem(data, δμ; max_order=order)
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
    F_x_lambda = screened_lindhard(x; lambda=lambda)
    # δμ₁ cancels the real part of the Fock self-energy
    # at the Fermi surface for a Yukawa-screened UEG.
    return -(param.e0^2 * param.kF / (2 * pi^2 * param.ϵ0)) * F_x_lambda
end

"""Load counterterm data from CSV file."""
function load_z_mu(
    param::UEG.ParaMC;
    parafilename="examples/counterterms/para.csv",
    ct_filename="examples/counterterms/data_Z.jld2",
)
    # Load μ from csv
    local ct_data
    filefound = false
    f = jldopen(ct_filename, "r")
    for key in keys(f)
        if UEG.paraid(f[key][1]) == UEG.paraid(param)
            ct_data = f[key]
            filefound = true
        end
    end
    if !filefound
        throw(KeyError(UEG.paraid(param)))
    end

    df = fromFile(parafilename)
    para, _, _, data = ct_data
    printstyled(UEG.short(para); color=:yellow)
    println()

    function zfactor(data, β)
        return @. (imag(data[2, 1]) - imag(data[1, 1])) / (2π / β)
    end

    function mu(data)
        return real(data[1, 1])
    end

    for p in sort([k for k in keys(data)])
        println("$p: μ = $(mu(data[p]))   z = $(zfactor(data[p], para.β))")
    end

    μ = Dict()
    for (p, val) in data
        μ[p] = mu(val)
    end
    z = Dict()
    for (p, val) in data
        z[p] = zfactor(val, para.β)
    end

    return z, μ
end

"""Modified from EFT_UEG to store parafile in local (SOSEM) directory."""
function fromFile(parafile=parafileName)
    println("Reading para from $parafile")
    try
        data, header = readdlm(parafile, ','; header=true)
        df = DataFrame(data, vec(header))
        CounterTerm.sortdata!(df)
        return df
    catch e
        println(e)
        println("Failed to load from $parafile. We will initialize the file instead")
        return nothing
    end
end

"""Modified from EFT_UEG to store parafile in local (SOSEM) directory."""
function toFile(df, parafile=parafileName)
    if isnothing(df)
        @warn "Empty dataframe $df, nothing to save"
        return
    end
    CounterTerm.sortdata!(df)
    println("Save the parameters to the file $parafile")
    return writedlm(parafile, Iterators.flatten(([names(df)], eachrow(df))), ',')
end
