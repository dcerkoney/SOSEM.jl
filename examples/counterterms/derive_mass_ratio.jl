using CodecZlib
using ElectronLiquid
using Measurements
using JLD2
using SOSEM

# Change to counterterm directory
if haskey(ENV, "SOSEM_CEPH")
    cd("$(ENV["SOSEM_CEPH"])/examples/counterterms")
elseif haskey(ENV, "SOSEM_HOME")
    cd("$(ENV["SOSEM_HOME"])/examples/counterterms")
end

# order = [5]
order = [4]
beta = [40.0]

### rs = 1 ###
# rs = [1.0]
# mass2 = [2.0]
# mass2 = [3.5]
# mass2 = [2.5, 3.0, 3.5, 4.0]
# mass2 = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

### rs = 2 ###
# rs = [2.0]
# order = [5]
# mass2 = [2.5]
# mass2 = [1.625, 1.75, 1.875, 2.0]
# mass2 = [0.1, 0.25, 0.5, 1.0, 1.25, 1.5, 1.625, 1.75, 1.875, 2.0, 2.5, 3.0]

# rs = [2.0]
# mass2 = [1.75]
# mass2 = [1.5, 1.75, 2.0]
# mass2 = [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

### rs = 3 ###
rs = [3.0]
mass2 = [1.75]
# mass2 = [1.75, 2.0]
# mass2 = [1.0, 1.125, 1.25, 1.5]
# mass2 = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 2.0]

# rs = [3.0]
# mass2 = [0.75, 0.875, 1.0, 1.125, 1.25, 1.5]
# mass2 = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]

### rs = 4 ###
# rs = [4.0]
# mass2 = [0.875, 1.125, 1.25, 1.5]
# mass2 = [1.0]
# mass2 = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 2.0]
# mass2 = [0.25, 0.5, 0.75, 1.0, 1.25]
# mass2 = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]

### rs = 5 ###
# rs = [5.0]
# N = 5
# mass2 = [0.8125, 0.875, 0.9375]
# N = 4
# mass2 = [0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5]
# mass2 = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]

# Momentum spacing for finite-difference derivative of Sigma (in units of para.kF)
# δK = 0.01   # kspacings = -6:2:6 
# δK = 0.005  # kspacings = -15:15

# # Oldest grids for rs=1 check
# δK = 0.005
# idks = collect(1:15)
# kspacings = collect(0:30)
# dks = δK * collect(kspacings)

# Which finite difference method for numerical k derivative?
# method = :forward
method = :central

# Enable/disable interaction and chemical potential counterterms
renorm_mu = true
renorm_lambda = true

# Remove Fock insertions?
isFock = false

# Distinguish results with different counterterm schemes
ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
if renorm_mu
    ct_string *= "_mu"
end
if renorm_lambda
    ct_string *= "_lambda"
end

# Location of mass shift data
const filename = "data/data_mass_ratio.jld2"

# Location of z/μ data
const parafilename = "data/para.csv"

# function process_mass_ratio(datatuple, δzi, δμ, isSave)
function process_mass_ratio(
    datatuple,
    para,
    isSave,
    has_taylor_factors;
    idk=1,
    method=:central,
)
    println("\nProcessing mass ratio (idk=$idk)...")
    @assert method in [:forward, :central]
    if method == :forward
        @warn "Forward difference method is deprecated, use `method = :central`!"
    end
    # df = UEG_MC.fromFile(parafilename)
    ngrid, kgrid, data = datatuple
    printstyled(UEG.short(para); color=:yellow)
    println()

    nk = length(kgrid)
    @assert nk == 7
    δK = 0.01

    # Old k-spacing is 0.005, new k-spacing is 0.01
    # δK = nk == length(-15:15) ? 0.005 : 0.01

    # Derive kgrid indices & spacing δK from data
    dks = round.(kgrid / para.kF .- 1; sigdigits=13)
    kspacings = trunc.(Int, dks / δK)
    @assert kspacings == collect(-6:2:6)

    # if nk == length(-6:2:6)
    #     @assert kspacings == collect(-6:2:6)
    # elseif nk == length(-15:15)
    #     @assert kspacings == collect(-15:15)
    # else
    #     error("kgrid spacing $(kspacings) not supported!")
    # end

    idks = collect(eachindex(kspacings[kspacings .>= 0])) .- 1
    @assert idk ∈ idks
    @assert dks == δK * collect(kspacings)

    # Get Fermi index
    ikF = searchsortedfirst(kgrid, para.kF)
    # @assert ikF == 1

    # println(kgrid)
    println("\nikF: $ikF")
    println("δK: $δK")
    println("idks: $idks")
    println("kspacings: $kspacings")
    println("dks: $dks")

    # Max order in RPT calculation
    max_order = para.order
    println("\nMax order: ", max_order)

    # Load counterterm data from CSV file
    mu, sw = UEG_MC.getSigma(para; parafile=parafilename)
    δzi, δμ, δz = CounterTerm.sigmaCT(max_order, mu, sw; verbose=1)
    println("δzi:\n", δzi, "\nδμ:\n", δμ, "\nδz:\n", δz)
    # error("done testing")

    # Convert data to a Dict of measurements with interaction counterterms merged
    _data = Dict{keytype(data),valtype(data)}()
    for (k, v) in data
        if has_taylor_factors
            _data[k] = v
        else
            _data[k] = v / (factorial(k[2]) * factorial(k[3]))
        end
        # _data[k] = v / (factorial(k[2]) * factorial(k[3]))
    end
    merged_data = CounterTerm.mergeInteraction(_data)

    # Renormalize self-energy data (ngrid, kgrid) to get Sigma to order N in RPT
    Σ_renorm = CounterTerm.chemicalpotential_renormalization(max_order, merged_data, δμ)
    # Σ_renorm =
    #     UEG_MC.chemicalpotential_renormalization_sigma(merged_data, δμ; max_order=max_order)

    # Compute shifts δm and δs for each order n in RPT
    δm = Measurement{Float64}[]
    δs = Measurement{Float64}[]
    for n in 1:max_order
        Σ_n = Σ_renorm[n]
        # ∂ₖReΣ(k, ikₙ = 0) at the Fermi surface k = kF
        @assert idk ∈ idks
        @assert kgrid[ikF] ≈ para.kF
        @assert kgrid[ikF + idk] - kgrid[ikF] ≈ para.kF * dks[ikF + idk]

        # Forward difference method
        if method == :forward
            ds_dk =
                (real(Σ_n[1, ikF + idk]) - real(Σ_n[1, ikF])) /
                (kgrid[ikF + idk] - kgrid[ikF])
        else # central difference method
            @assert kgrid[ikF] - kgrid[ikF - idk] ≈ para.kF * dks[ikF + idk]
            @assert kgrid[ikF + idk] - kgrid[ikF - idk] ≈ 2 * para.kF * dks[ikF + idk]
            ds_dk =
                (real(Σ_n[1, ikF + idk]) - real(Σ_n[1, ikF - idk])) /
                (kgrid[ikF + idk] - kgrid[ikF - idk])
        end
        # Store δm and δs counterterms
        # NOTE: Extra minus sign on definition of Σ
        dm = (para.me / para.kF) * ds_dk
        ds = δzi[n]
        push!(δm, -dm)
        push!(δs, -ds)
        # Printout for benchmark
        println(
            "(Order $n) δK = $(dks[ikF + idk]), Re(Σ_$n(0, kF + δK)) = $(real(Σ_n[1, ikF + idk]))",
        )
    end

    # Display δs_n and δm_n
    println()
    for i in eachindex(δs)
        println(" • δs_$i = $(δs[i])")
    end
    println()
    for i in eachindex(δm)
        println(" • δm_$i = $(δm[i])")
    end

    # The inverse Z-factor is (1 - δs[ξ])
    zinv = Measurement{Float64}[1; accumulate(+, δzi; init=1)]

    # Power series terms for (1 + δM[ξ]) = (1 + δm[ξ])⁻¹ and
    # (m⋆/m)[ξ] = (1 - δs[ξ]) ∘ (1 + δm[ξ])⁻¹ = (1 - δs[ξ]) ∘ (1 + δM[ξ])
    δM = Measurement{Float64}[]
    δr = Measurement{Float64}[1.0]
    if max_order ≥ 1
        push!(δM, -δm[1])
        push!(δr, δM[1] - δs[1])
    end
    if max_order ≥ 2
        push!(δM, δm[1]^2 - δm[2])
        push!(δr, δM[2] - δM[1] * δs[1] - δs[2])
    end
    if max_order ≥ 3
        push!(δM, -δm[1]^3 + 2 * δm[1] * δm[2] - δm[3])
        push!(δr, δM[3] - δM[2] * δs[1] - δM[1] * δs[2] - δs[3])
    end
    if max_order ≥ 4
        push!(δM, δm[1]^4 - 3 * δm[1]^2 * δm[2] + δm[2]^2 + 2 * δm[1] * δm[3] - δm[4])
        push!(δr, δM[4] - δM[3] * δs[1] - δM[2] * δs[2] - δM[1] * δs[3] - δs[4])
    end
    if max_order ≥ 5
        push!(
            δM,
            -δm[1]^5 + 4 * δm[1]^3 * δm[2] - 3 * δm[1] * δm[2]^2 - 3 * δm[1]^2 * δm[3] +
            2 * δm[2] * δm[3] +
            2 * δm[1] * δm[4] - δm[5],
        )
        push!(
            δr,
            δM[5] - δM[4] * δs[1] - δM[3] * δs[2] - δM[2] * δs[3] - δM[1] * δs[4] - δs[5],
        )
    end
    # δM = Measurement{Float64}[
    #     -δm[1],
    #     δm[1]^2 - δm[2],
    #     -δm[1]^3 + 2 * δm[1] * δm[2] - δm[3],
    #     δm[1]^4 - 3 * δm[1]^2 * δm[2] + 2 * δm[1] * δm[3] - δm[4],
    #     -δm[1]^5 + 4 * δm[1]^3 * δm[2] - 3 * δm[1] * δm[2]^2 - 3 * δm[1]^2 * δm[3] + 2 * δm[2] * δm[3] + 2 * δm[1] * δm[4] - δm[5],
    # ]
    # # Power series terms for (m⋆/m)[ξ] = (1 - δs[ξ]) ∘ (1 + δm[ξ])⁻¹ = (1 - δs[ξ]) ∘ (1 + δM[ξ])
    # δr = Measurement{Float64}[
    #     1.0,
    #     δM[1] - δs[1],
    #     δM[2] - δM[1] * δs[1] - δs[2],
    #     δM[3] - δM[2] * δs[1] - δM[1] * δs[2] - δs[3],
    #     δM[4] - δM[3] * δs[1] - δM[2] * δs[2] - δM[1] * δs[3] - δs[4],
    #     δM[5] - δM[4] * δs[1] - δM[3] * δs[2] - δM[2] * δs[3] - δM[1] * δs[4] - δs[5],
    # ]
    # println(δM)

    # Aggregate terms in power series for (m⋆/m)[ξ]
    mass_ratios = accumulate(+, δr)
    # Print results
    for i in eachindex(mass_ratios)
        N = i - 1
        # println(δr[i].val, "\t", δr[i].err)
        # println(mass_ratios[i].val, "\t", mass_ratios[i].err)
        println("\n(N = $N)\nδ(m⋆/m) = $(δr[i])\nm⋆/m = $(mass_ratios[i])")
    end
    println()
    println(δr)
    println(mass_ratios)

    if isSave
        println("Current working directory: $(pwd())")
        println("Saving data to JLD2...")
        jldopen("data/mass_ratio.jld2", "a+"; compress=true) do f
            key = "$(UEG.short(para))_idk=$(idk)"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (ngrid, kgrid, mass_ratios)
        end
        if idk == 1
            jldopen("data/inverse_zfactor.jld2", "a+"; compress=true) do f
                key = "$(UEG.short(para))"
                if haskey(f, key)
                    @warn("replacing existing data for $key")
                    delete!(f, key)
                end
                return f[key] = (ngrid, kgrid, zinv)
            end
        end
    end
    return mass_ratios
end

if abspath(PROGRAM_FILE) == @__FILE__
    # @assert length(ARGS) >= 1 "One argument for the data file name is required!"
    # filename = ARGS[1]
    isSave = false
    if length(ARGS) >= 1 &&
       (ARGS[1] == "s" || ARGS[1] == "-s" || ARGS[1] == "--save" || ARGS[1] == " save")
        # the second parameter may be set to save the derived parameters
        isSave = true
    end
    f = jldopen(filename, "r")
    for (_rs, _mass2, _beta, _order) in Iterators.product(rs, mass2, beta, order)
        para = UEG.ParaMC(;
            rs=_rs,
            beta=_beta,
            order=_order,
            mass2=_mass2,
            isDynamic=false,
            isFock=isFock,
        )
        mass_ratios = []
        if haskey(f, "has_taylor_factors") == false
            error(
                "Data missing key 'has_taylor_factors', process with script 'add_taylor_factors_to_counterterm_data.jl'!",
            )
        end
        has_taylor_factors::Bool = f["has_taylor_factors"]
        for key in keys(f)
            key == "has_taylor_factors" && continue
            loadpara = UEG.ParaMC(string(split(key, "_idk=")[1]))
            if UEG.paraid(loadpara) == UEG.paraid(para)
                htf_str = has_taylor_factors ? "with" : "without"
                println("Found data $(htf_str) Taylor factors...")
                # Derive the list of positive index offsets from ikF via kgrid
                # (f[key] = ngrid, kgrid, sigma)
                kgrid = f[key][2]
                ikF = searchsortedfirst(kgrid, para.kF)
                idks = collect(eachindex(kgrid[(ikF + 1):end]))
                @assert all(ikF + idk ∈ eachindex(kgrid) for idk in idks)
                @assert all(ikF - idk ∈ eachindex(kgrid) for idk in idks)
                for idk in idks
                    mass_ratio = process_mass_ratio(
                        f[key],
                        loadpara,
                        isSave,
                        has_taylor_factors;
                        idk=idk,
                        method=method,
                    )
                    push!(mass_ratios, mass_ratio)
                end
                println("\nDone!")
            end
        end
    end
end
