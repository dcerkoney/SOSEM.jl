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
                data_with_missing_partns[(2, n)] = zero(data[(max_order, 0)])
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

"""Computes the exact first-order (screened HF) effective mass ratio"""
function fock_mass_ratio(param::UEG.ParaMC)
    return
end

"""Load z and μ data from JLD2 file."""
function load_z_mu(
    param::UEG.ParaMC;
    ct_filename="examples/counterterms/data/data_Z_with_ct_mu_lambda.jld2",
)
    # Derive z and μ from JLD2 data
    print("Deriving counterterm data from JLD2...")
    local has_taylor_factors
    local ct_data
    filefound = false
    f = jldopen(ct_filename, "r")
    if haskey(f, "has_taylor_factors") == false
        error(
            "Data missing key 'has_taylor_factors', process with script 'add_taylor_factors_to_counterterm_data.jl'!",
        )
    end
    has_taylor_factors::Bool = f["has_taylor_factors"]
    for key in keys(f)
        key == "has_taylor_factors" && continue
        if UEG.paraid(f[key][1]) == UEG.paraid(param)
            ct_data = f[key]
            filefound = true
        end
    end
    if !filefound
        throw(KeyError(UEG.paraid(param)))
    end
    # df = fromFile(parafilename)
    para, _, _, data = ct_data
    printstyled(UEG.short(para); color=:yellow)
    println()

    function get_sw(sigma_data, β)
        sw = @. (imag(sigma_data[2, 1]) - imag(sigma_data[1, 1])) / (2π / β)
        # Add Taylor factors if not present
        if has_taylor_factors == false
            for (p, v) in sw
                sw[p] = v / (factorial(p[2]) * factorial(p[3]))
            end
        end
        return sw
    end
    function get_mu(sigma_data)
        mu = real(sigma_data[1, 1])
        # Add Taylor factors if not present
        if has_taylor_factors == false
            for (p, v) in mu
                mu[p] = v / (factorial(p[2]) * factorial(p[3]))
            end
        end
        return mu
    end

    for p in sort([k for k in keys(data)])
        println("$p: mu = $(get_mu(data[p]))   sw = $(get_sw(data[p], para.β))")
    end
    mu = Dict()
    for (p, val) in data
        mu[p] = get_mu(val)
    end
    sw = Dict()
    for (p, val) in data
        sw[p] = get_sw(val, para.β)
    end
    return sw, mu
end

"""Load counterterm data from CSV/JLD2 file."""
function load_z_mu_counterterms(
    param::UEG.ParaMC;
    max_order=param.order,
    parafilename="examples/counterterms/data/para_0m1.csv",
    ct_filename="examples/counterterms/data/data_Z_with_ct_mu_lambda_kF.jld2",
    isFock=false,
    verbose=1,
)
    # Load z and μ data from file
    print("Loading counterterm data from CSV...")
    try
        # First, try loading data from CSV file
        mu, sw = UEG_MC.getSigma(para; parafile=parafilename)
    catch
        # Fallback mode: Re-derive counterterms from JLD2 file
        @warn "falling back to JLD2 data..."
        local has_taylor_factors
        local ct_data
        filefound = false
        f = jldopen(ct_filename, "r")
        if haskey(f, "has_taylor_factors") == false
            error(
                "Data missing key 'has_taylor_factors', process with script 'add_taylor_factors_to_counterterm_data.jl'!",
            )
        end
        has_taylor_factors::Bool = f["has_taylor_factors"]
        for key in keys(f)
            key == "has_taylor_factors" && continue
            if UEG.paraid(f[key][1]) == UEG.paraid(param)
                ct_data = f[key]
                filefound = true
            end
        end
        if !filefound
            throw(KeyError(UEG.paraid(param)))
        end
        # df = fromFile(parafilename)
        para, _, _, data = ct_data
        printstyled(UEG.short(para); color=:yellow)
        println()

        function get_sw(sigma_data, β)
            sw = @. (imag(sigma_data[2, 1]) - imag(sigma_data[1, 1])) / (2π / β)
            # Add Taylor factors if not present
            if has_taylor_factors == false
                for (p, v) in sw
                    sw[p] = v / (factorial(p[2]) * factorial(p[3]))
                end
            end
            return sw
        end
        function get_mu(sigma_data)
            mu = real(sigma_data[1, 1])
            # Add Taylor factors if not present
            if has_taylor_factors == false
                for (p, v) in mu
                    mu[p] = v / (factorial(p[2]) * factorial(p[3]))
                end
            end
            return mu
        end

        for p in sort([k for k in keys(data)])
            println("$p: mu = $(get_mu(data[p]))   sw = $(get_sw(data[p], para.β))")
        end
        mu = Dict()
        for (p, val) in data
            mu[p] = get_mu(val)
        end
        sw = Dict()
        for (p, val) in data
            sw[p] = get_sw(val, para.β)
        end
    end
    # Derive counterterms
    print("...deriving counterterms...")
    δzi, δμ, δz = CounterTerm.sigmaCT(max_order, mu, sw; isfock=isFock, verbose=verbose)
    println("done!")
    println("δzi: ", δzi, "\tδμ: ", δμ, "\tδz: ", δz)
    return δzi, δμ, δz
end

"""Loads chemical potential counterterm data δμ from CSV/JLD2 file."""
load_mu_counterterm(args...; kwargs...) = load_z_mu_counterterms(args...; kwargs...)[2]

"""
    function getSigma(para::ParaMC; order=para.order, parafile=parafileName)

    Read self-energy parameters from the file.
    Modified from EFT_UEG to load parafile from local (SOSEM) directory.

# Arguments
df     : DataFrame
paraid : Dictionary of parameter names and values
order  : the truncation order
"""
function getSigma(para::ParaMC; order=para.order, parafile=parafileName)
    df = fromFile(parafile)
    @assert isnothing(df) == false "file $parafile failed to load"
    return CounterTerm.getSigma(df, UEG.paraid(para), order)
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
