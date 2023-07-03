using CodecZlib
using ElectronLiquid
using Measurements
# using Printf
using JLD2
using SOSEM

# Change to counterterm directory
if haskey(ENV, "SOSEM_CEPH")
    cd("$(ENV["SOSEM_CEPH"])/examples/counterterms")
elseif haskey(ENV, "SOSEM_HOME")
    cd("$(ENV["SOSEM_HOME"])/examples/counterterms")
end

############################################
# For lambda optimization
############################################
order = [5]
# order = [4]
beta = [40.0]

### rs = 1 ###
# rs = [1.0]
# mass2 = [1.0]

### rs = 2 ###
# rs = [2.0]
# mass2 = [1.25, 1.5, 1.625, 1.75, 1.875, 2.0]

# rs = [2.0]
# mass2 = [1.75]
# mass2 = [1.5, 1.75, 2.0]
# mass2 = [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

### rs = 3 ###
# rs = [3.0]
# mass2 = [0.75, 0.875, 1.0, 1.125, 1.25, 1.5]
# mass2 = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]

### rs = 4 ###
# rs = [4.0]
# mass2 = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125]
# mass2 = [0.25, 0.5, 0.75, 1.0, 1.25]
# mass2 = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]

### rs = 5 ###
# N = 5
rs = [5.0]
mass2 = [0.8125, 0.875, 0.9375]
# N = 4
# rs = [5.0]
# mass2 = [0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5]
# mass2 = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]

# Momentum spacing for finite-difference derivative of Sigma (in units of para.kF)
δK = 0.01
# δK = 0.005  # spacings n*δK = 0.15–0.3 not relevant for rs = 1.0 => reduce δK by half

# We estimate the derivative wrt k using grid points kgrid[ikF] and kgrid[ikF + idk]
kspacings = -6:2:6
idks = eachindex(kspacings)


# kgrid indices & spacings
dks = δK * collect(kspacings)

# Which finite difference method for numerical k derivative?
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

# const filename = "data/data_mass_ratio$(ct_string).jld2"
# const filename = "data/data_mass_ratio$(ct_string).jld2"
# const filename = "data/data_mass_ratio$(ct_string)_kF_gridtest.jld2"
const filename = "data/data_mass_ratio$(ct_string)_kF_gridtest_archive1.jld2"
const parafilename = "data/para.csv"

# function process_mass_ratio(datatuple, δzi, δμ, isSave)
function process_mass_ratio(
    datatuple,
    isSave,
    has_taylor_factors_mass;
    idk=1,
    method=:forward,
)
    print("processing mass ratio...")
    @assert idk ∈ idks
    @assert method in [:forward, :central]
    # df = UEG_MC.fromFile(parafilename)
    para, ngrid, kgrid, data = datatuple
    printstyled(UEG.short(para); color=:yellow)

    # Get Fermi index
    ikF = searchsortedfirst(kgrid, para.kF)
    println("ikF = $ikF")

    # Max order in RPT calculation
    max_order = para.order
    println("Max order: ", max_order)

    # Reexpand merged data in powers of μ
    ct_filename = "data/data_Z$(ct_string)_kF_opt_archive1.jld2"
    # ct_filename = "data/data_Z$(ct_string)_kF_opt.jld2"
    # ct_filename = "data/data_Z$(ct_string)_kF.jld2"
    # ct_filename = "data/data_Z$(ct_string).jld2"
    z, μ, has_taylor_factors_zmu = UEG_MC.load_z_mu(para; ct_filename=ct_filename, parafilename=parafilename)
    # Add Taylor factors to CT data
    for (p, v) in z
        if has_taylor_factors_zmu
            z[p] = v
        else
            z[p] = v / (factorial(p[2]) * factorial(p[3]))
        end
    end
    for (p, v) in μ
        if has_taylor_factors_zmu
            μ[p] = v
        else
            μ[p] = v / (factorial(p[2]) * factorial(p[3]))
        end
    end
    δzi, δμ, _ = CounterTerm.sigmaCT(max_order, μ, z; verbose=1)
    println("Computed δμ: ", δμ)

    # Total Z to order max_order
    z = 1 - sum(δzi)
    println("δzi:\n", δzi, "\n", "z = ", z)

    # Convert data to a Dict of measurements with interaction counterterms merged
    _data = Dict{keytype(data),valtype(data)}()
    for (k, v) in data
        if has_taylor_factors_mass
            _data[k] = v
        else
            _data[k] = v / (factorial(k[2]) * factorial(k[3]))
        end
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
        # Σ_N = Σ_total[n]  # bug!
        Σ_n = Σ_renorm[n]
        # ∂ₖReΣ(k, ikₙ = 0) at the Fermi surface k = kF
        # NOTE: n1 = 0, k1 = kF, k2 = kF * (1 + δK), k2 - k1 = kF * δK
        # idk = 1
        @assert idk ∈ idks
        @assert kgrid[ikF] ≈ para.kF
        # @assert kgrid[ikF + 1] ≈ para.kF + para.kF * δK
        @assert kgrid[ikF + kspacings[idk]] - kgrid[ikF] ≈ para.kF * dks[idk]

        # Forward difference method
        if method == :forward
            ds_dk =
                (real(Σ_n[1, ikF + idk]) - real(Σ_n[1, ikF])) /
                (kgrid[ikF + idk] - kgrid[ikF])
        else # central difference method
            @assert kgrid[ikF - 1] ≈ para.kF - para.kF * δK
            @assert kgrid[ikF] - kgrid[ikF - idk] ≈ para.kF * dks[idk]
            # TODO: test central difference method for ds_dk
            ds_dk =
                (real(Σ_n[1, ikF + idk]) - real(Σ_n[1, ikF - idk])) /
                (kgrid[ikF + idk] - kgrid[ikF - idk])
        end

        dm = (para.me / para.kF) * ds_dk
        # println("ds_dk = ", ds_dk)
        # Use existing Z-factor data
        ds = δzi[n]
        # Recompute Z-factor from new Σ run
        # ds = (para.β / 2π) * (imag(Σ_n[2, 1]) - imag(Σ_n[1, 1])) # = zfactor(Σ_n, para.β) = δ(1/z)
        # NOTE: Extra minus sign on definition of Σ
        push!(δm, -dm)
        push!(δs, -ds)
        # Printout for benchmark
        println("Order $n:")
        println("δK = $(dks[idk]), Re(Σ_$n(0, kF + δK)) = $(real(Σ_n[1, 1 + idk]))")
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

    # The Z-factor can be approximated by (1 - δs[ξ])⁻¹ (WARNING: we should expand as a power series instead!)
    zapprox = 1 ./ zinv

    # Power series terms for (1 + δM[ξ]) = (1 + δm[ξ])⁻¹
    δM = Measurement{Float64}[
        -δm[1],
        δm[1]^2 - δm[2],
        -δm[1]^3 + 2 * δm[1] * δm[2] - δm[3],
        δm[1]^4 - 3 * δm[1]^2 * δm[2] + 2 * δm[1] * δm[3] - δm[4],
    ]
    # Power series terms for (m⋆/m)[ξ] = (1 - δs[ξ]) ∘ (1 + δm[ξ])⁻¹ = (1 - δs[ξ]) ∘ (1 + δM[ξ])
    δr = Measurement{Float64}[
        1.0,
        δM[1] - δs[1],
        δM[2] - δM[1] * δs[1] - δs[2],
        δM[3] - δM[2] * δs[1] - δM[1] * δs[2] - δs[3],
        δM[4] - δM[3] * δs[1] - δM[2] * δs[2] - δM[1] * δs[3] - δs[4],
    ]
    println(δM)

    # Aggregate terms in power series for (m⋆/m)[ξ]
    mass_ratios = accumulate(+, δr)
    # mass_ratios = accumulate(+, δr; init=1.0)
    # Print results
    for i in eachindex(mass_ratios)
        N = i - 1
        println(δr[i].val, "\t", δr[i].err)
        println(mass_ratios[i].val, "\t", mass_ratios[i].err)
        println("\n(N = $N)\nδ(m⋆/m) = $(δr[i])\nm⋆/m = $(mass_ratios[i])")
    end
    println()
    println(δr)
    println(mass_ratios)

    if isSave
        println("Current working directory: $(pwd())")
        println("Saving data to JLD2...")
        # jldopen("data/mass_ratio_from_sigma.jld2", "a+"; compress=true) do f
        # jldopen("data/mass_ratio_from_sigma_gridtest.jld2", "a+"; compress=true) do f
        jldopen("data/mass_ratio_from_sigma_kF_gridtest.jld2", "a+"; compress=true) do f
            key = "$(UEG.short(para))_idk=$(idk)"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (para, ngrid, kgrid, mass_ratios)
        end
        jldopen("data/inverse_zfactor.jld2", "a+"; compress=true) do f
            key = "$(UEG.short(para))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (para, ngrid, kgrid, zinv)
        end
        jldopen("data/zfactor_approx.jld2", "a+"; compress=true) do f
            key = "$(UEG.short(para))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (para, ngrid, kgrid, zapprox)
        end
    end
    println("Done!")
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

        kF = para.kF
        mass_ratios = []
        if haskey(f, "has_taylor_factors") == false
            error(
                "Data missing key 'has_taylor_factors', process with script 'add_taylor_factors_to_counterterm_data.jl'!",
            )
        end
        has_taylor_factors_mass::Bool = f["has_taylor_factors"]
        for key in keys(f)
            key == "has_taylor_factors" && continue
            if UEG.paraid(f[key][1]) == UEG.paraid(para)
                htf_str = has_taylor_factors_mass ? "with" : "without"
                print("Found data $(htf_str) Taylor factors...")
                # process_mass_ratio(f[key], isSave)
                for idk in idks
                    mass_ratio = process_mass_ratio(
                        f[key],
                        isSave,
                        has_taylor_factors_mass;
                        idk=idk,
                        method=method,
                    )
                    push!(mass_ratios, pass_ratio)
                end
                println("done!")
            end
        end
    end
end
