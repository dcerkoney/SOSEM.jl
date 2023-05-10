function chemicalpotential_renormalization(data, δμ; n_min, min_order=n_min, max_order)
    @assert max_order ≤ n_min + 4 "Order $order hasn't been implemented!"
    # println(δμ)
    @assert length(δμ) ≥ max_order - n_min
    d = CounterTerm.mergeInteraction(data)  # interaction-merged data
    T = valtype(d)
    d_renorm = RenormMeasType{T}()
    # Renormalize data to maximum supported counterterm order: 
    # [𝓞_{nmin}, 𝓞_{nmin+1}, 𝓞_{nmin+2}, 𝓞_{nmin+3}, 𝓞_{nmin+4}]
    if min_order ≤ n_min ≤ max_order
        d_renorm[n_min] = d[(n_min, 0)]
    end
    if min_order ≤ n_min + 1 ≤ max_order
        d_renorm[n_min + 1] = d[(n_min + 1, 0)] + d[(n_min, 1)] * δμ[1]
    end
    if min_order ≤ n_min + 2 ≤ max_order
        #! format: off
        d_renorm[n_min + 2] =
            d[(n_min + 2, 0)]           +
            d[(n_min + 1, 1)] * δμ[1]   +
            d[(n_min    , 2)] * δμ[1]^2 +
            d[(n_min    , 1)] * δμ[2]
        #! format: on
    end
    if min_order ≤ n_min + 3 ≤ max_order
        #! format: off
        d_renorm[n_min + 3] =
            d[(n_min + 3, 0)]                     +
            d[(n_min + 2, 1)] * δμ[1]             +
            d[(n_min + 1, 2)] * δμ[1]^2           +
            d[(n_min    , 3)] * δμ[1]^3           +
            d[(n_min + 1, 1)] * δμ[2]             +
            d[(n_min    , 2)] * 2 * δμ[1] * δμ[2] +
            d[(n_min    , 1)] * δμ[3]
        #! format: on
    end
    if min_order ≤ n_min + 4 ≤ max_order
        #! format: off
        d_renorm[n_min + 4] =
            d[(n_min + 4, 0)]                                 + 
            d[(n_min + 3, 1)] * δμ[1]                         + 
            d[(n_min + 2, 2)] * δμ[1]^2                       +
            d[(n_min + 1, 3)] * δμ[1]^3                       +
            d[(n_min    , 4)] * δμ[1]^4                       +
            d[(n_min + 2, 1)] * δμ[2]                         +
            d[(n_min + 1, 2)] * 2 * δμ[1] * δμ[2]             +
            d[(n_min    , 3)] * 3 * δμ[1]^2 * δμ[2]           +
            d[(n_min    , 2)] * (δμ[2]^2 + 2 * δμ[1] * δμ[3]) +
            d[(n_min + 1, 1)] * δμ[3]                         +
            d[(n_min    , 1)] * δμ[4]
        #! format: on
    end
    return d_renorm
end

"""
Same as chemicalpotential_renormalization, but allow for lowest_order >= n_min
(one SOSEM observable starts at 3rd loop order (n_min + 1)).
"""
# TODO: data could be merged or unmerged => merge it here instead of chemicalpotential_renormalization?
function chemicalpotential_renormalization_sosem(
    data,
    δμ;
    lowest_order=2,
    min_order=2,
    max_order,
)
    # If the lowest_order for this SOSEM observable is greater than 2,
    # then all counterterms of partition type (2, n) are zero. Here we
    # add back these partitions so the corresponding terms in the
    # renormalization function are both defined and zeroed out.
    data_with_missing_partns = deepcopy(data)
    if lowest_order > 2
        for n in 1:lowest_order
            # data_with_missing_partns[(2, n, 0)] = zero(valtype(data))
            if length(collect(keys(data))[1]) == 2
                data_with_missing_partns[(2, n)] =
                    zero(data[(max_order, 0)])
            else
                data_with_missing_partns[(2, n, 0)] =
                    zero(data[(max_order - lowest_order, 0, 0)])
            end
        end
    end
    return chemicalpotential_renormalization(
        data_with_missing_partns,
        δμ;
        n_min=lowest_order,
        min_order=min_order,
        max_order=max_order,
    )
end
function chemicalpotential_renormalization_sosem(order, data, δμ)
    return chemicalpotential_renormalization_sosem(data, δμ; max_order=order)
end

function chemicalpotential_renormalization_sigma(data, δμ; min_order=1, max_order)
    return chemicalpotential_renormalization(
        data,
        δμ;
        n_min=1,
        min_order=min_order,
        max_order=max_order,
    )
end

function chemicalpotential_renormalization_green(data, δμ; min_order=0, max_order)
    return chemicalpotential_renormalization(
        data,
        δμ;
        n_min=0,
        min_order=min_order,
        max_order=max_order,
    )
end

const chemicalpotential_renormalization_density = chemicalpotential_renormalization_green
const chemicalpotential_renormalization_poln = chemicalpotential_renormalization_sigma
const chemicalpotential_renormalization_susceptibility =
    chemicalpotential_renormalization_sigma

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
