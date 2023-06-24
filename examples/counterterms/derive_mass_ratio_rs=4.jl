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

const order = 4  # C^{(1)}_{N≤5} includes CTs up to 3rd order
const beta = 40.0
const rs = 4.0
const orders = [1, 2, 3, 4]
const optimal_lambdas = [0.5, 0.625, 0.75, 1.0]

function paraid_no_mass2(p::UEG.ParaMC)
    return Dict(
        "dim" => p.dim,
        "rs" => p.rs,
        "beta" => p.beta,
        "Fs" => p.Fs,
        "Fa" => p.Fa,
        "massratio" => p.massratio,
        "spin" => p.spin,
        "isFock" => p.isFock,
        "isDynamic" => p.isDynamic,
    )
end

function short_multilambda(p::UEG.ParaMC)
    return join(["$(k)_$(v)" for (k, v) in sort(paraid_no_mass2(p))], "_") * "_lambdas_$optimal_lambdas"
end

# opt1 = Measurements.value.(mass_ratios_N_vs_lambda[3])[lambdas .== 0.5]
# opt2 = Measurements.value.(mass_ratios_N_vs_lambda[3])[lambdas .== 0.625]
# opt3 = Measurements.value.(mass_ratios_N_vs_lambda[4])[lambdas .== 0.75]
# opt4 = Measurements.value.(mass_ratios_N_vs_lambda[5])[lambdas .== 1.0]

# Momentum spacing for finite-difference derivative of Sigma (in units of para.kF)
δK = 0.005  # spacings n*δK = 0.15–0.3 not relevant for rs = 1.0 => reduce δK by half

# We estimate the derivative wrt k using grid points kgrid[ikF] and kgrid[ikF + idk]
idks = 1:15

# kgrid indices & spacings
dks = δK * collect(idks)

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
const filename = "data/data_mass_ratio$(ct_string)_kF_gridtest.jld2"
const parafilename = "data/para.csv"

# Save the results to JLD2?
isSave = true

for idk in idks
    print("Processing mass ratio...")
    @assert idk ∈ idks
    @assert method in [:forward, :central]

    for mass2 in zip(orders, optimal_lambdas)
        f = jldopen(filename, "r")
        para, ngrid, kgrid, data = datatuple
        printstyled(UEG.short(para); color=:yellow)

        # Get Fermi index
        ikF = searchsortedfirst(kgrid, para.kF)
        println("ikF = $ikF")

        # Max order in RPT calculation
        max_order = para.order
        println("Max order: ", max_order)

        # Reexpand merged data in powers of μ
        ct_filename = "data/data_Z$(ct_string)_kF_opt.jld2"
        z, μ = UEG_MC.load_z_mu(para; ct_filename=ct_filename, parafilename=parafilename)
        # Add Taylor factors to CT data
        for (p, v) in z
            z[p] = v / (factorial(p[2]) * factorial(p[3]))
        end
        for (p, v) in μ
            μ[p] = v / (factorial(p[2]) * factorial(p[3]))
        end

        # println(data)

        # Convert data to a Dict of measurements with interaction counterterms merged
        _data = Dict{keytype(data),valtype(data)}()
        for (k, v) in data
            _data[k] = v / (factorial(k[2]) * factorial(k[3]))
        end
        merged_data = CounterTerm.mergeInteraction(_data)
        # println()
        # println([k for (k, _) in merged_data])
        # println(merged_data)

    end

    # δzi, δμ, _ = CounterTerm.sigmaCT(max_order - 1, μ, z; verbose=1)
    δzi, δμ, _ = CounterTerm.sigmaCT(max_order, μ, z; verbose=1)
    println("Computed δμ: ", δμ)

    # Total Z to order max_order
    z = 1 - sum(δzi)
    println("δzi:\n", δzi, "\n", "z = ", z)

    # Renormalize self-energy data (ngrid, kgrid) to get Sigma to order N in RPT
    Σ_renorm = CounterTerm.chemicalpotential_renormalization(max_order, merged_data, δμ)

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
        @assert kgrid[ikF + 1] ≈ para.kF + para.kF * δK
        @assert kgrid[ikF + idk] - kgrid[ikF] ≈ para.kF * dks[idk]

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
        jldopen("data/mass_ratio_from_sigma_kF_gridtest_rs4.jld2", "a+"; compress=true) do f
            key = "$(short_multilambda(para))_idk=$(idk)"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (para, ngrid, kgrid, mass_ratios)
        end
        jldopen("data/inverse_zfactor_rs4.jld2", "a+"; compress=true) do f
            key = "$(short_multilambda(para))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (para, ngrid, kgrid, zinv)
        end
        jldopen("data/zfactor_approx_rs4.jld2", "a+"; compress=true) do f
            key = "$(short_multilambda(para))"
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
