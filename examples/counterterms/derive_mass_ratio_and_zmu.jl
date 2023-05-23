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

# Physical params matching data for SOSEM observables
order = [4]  # C^{(1)}_{N≤5} includes CTs up to 3rd order
rs = [1.0]
mass2 = [1.0]
beta = [40.0]

# Momentum spacing for finite-difference derivative of Sigma (in units of kF)
# δK = 5e-6
# δK = 0.025
δK = 0.001

# We estimate the derivative wrt k using grid points kgrid[ikF] and kgrid[ikF + idk]
idks = 1:3
# idks = [3]

# kgrid indices & spacings
dks = [1, 5, 10] * δK
dkscales = [1, 5, 10]
# idks = 1:4
# dks = [1, 2, 4, 8] * δK
# dkscales = [1, 2, 4, 8]

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

# const filename = "data_mass_ratio_and_Z$(ct_string).jld2"
const filename = "data_mass_ratio$(ct_string).jld2"
const parafilename = "para.csv"

function zfactor(data, β)
    return @. (imag(data[2, 1]) - imag(data[1, 1])) / (2π / β)
end

function mu(data)
    return real(data[1, 1])
end

# function process_mass_ratio(datatuple, δz, δμ, isSave)
function process_mass_ratio(datatuple, isSave; idk=1)
    print("Processing mass ratio...")
    # df = UEG_MC.fromFile(parafilename)
    para, ngrid, kgrid, data = datatuple
    printstyled(UEG.short(para); color=:yellow)
    println()

    # Max order in RPT calculation
    max_order = para.order
    println("Max order: ", max_order)

    # Reexpand merged data in powers of μ
    ct_filename = "data_Z$(ct_string).jld2"
    z, μ = UEG_MC.load_z_mu(para; ct_filename=ct_filename, parafilename=parafilename)
    # Add Taylor factors to CT data
    for (p, v) in z
        z[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    for (p, v) in μ
        μ[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    # δz, δμ = CounterTerm.sigmaCT(max_order - 1, μ, z; verbose=1)
    δz, δμ = CounterTerm.sigmaCT(max_order, μ, z; verbose=1)
    println("Computed δμ: ", δμ)

    # Total Z to order max_order
    z = 1 - sum(δz)
    println("δz:\n", δz, "\n", "z = ", z)

    # println(data)

    # Convert data to a Dict of measurements with interaction counterterms merged
    _data = Dict{keytype(data),valtype(data)}()
    for (k, v) in data
        _data[k] = v / (factorial(k[2]) * factorial(k[3]))
    end
    merged_data = CounterTerm.mergeInteraction(_data)
    println()
    println([k for (k, _) in merged_data])
    println(merged_data)

    # Renormalize self-energy data (ngrid, kgrid) to get Sigma to order N in RPT
    Σ_renorm = CounterTerm.chemicalpotential_renormalization(max_order, merged_data, δμ)
    # Σ_renorm =
    #     UEG_MC.chemicalpotential_renormalization_sigma(merged_data, δμ; max_order=max_order)

    # Aggregate the full results for Σₓ up to order N
    # Σ_total = UEG_MC.aggregate_orders(Σ_renorm)

    println()
    println(Σ_renorm)
    # println(Σ_total)

    # Compute shifts δm and δs for each order n in RPT
    δm = Measurement{Float64}[]
    δs = Measurement{Float64}[]
    for n in 1:max_order
        # Σ_N = Σ_total[n]  # bug!
        Σ_n = Σ_renorm[n]
        # ∂ₖReΣ(k, ikₙ = 0) at the Fermi surface k = kF
        # NOTE: n1 = 0, k1 = kF, k2 = kF * (1 + δK), k2 - k1 = kF * δK
        # idk = 1
        @assert idk ∈ 1:3
        @assert kgrid[1] ≈ para.kF
        @assert kgrid[2] ≈ para.kF + para.kF * δK
        @assert kgrid[1 + idk] - kgrid[1] ≈ para.kF * dks[idk]
        # ds_dk = (real(Σ_n[1, 1]) - real(Σ_n[1, 1 + idk])) / (kgrid[1] - kgrid[1 + idk])
        ds_dk = (real(Σ_n[1, 1 + idk]) - real(Σ_n[1, 1])) / (kgrid[1 + idk] - kgrid[1])
        dm = (para.me / para.kF) * ds_dk
        println("ds_dk = ", ds_dk)
        # Use existing Z-factor data
        ds = δz[n]
        # Recompute Z-factor from new Σ run
        # ds = (para.β / 2π) * (imag(Σ_n[2, 1]) - imag(Σ_n[1, 1])) # = zfactor(Σ_n, para.β) = δ(1/z)
        push!(δm, dm)
        push!(δs, ds)
    end
    println()
    println(δs)
    println(δm)

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
        jldopen("mass_ratio_from_sigma.jld2", "a+"; compress=true) do f
            key = "$(UEG.short(para))_idk=$(idk)"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (para, ngrid, kgrid, mass_ratios)
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
        for key in keys(f)
            if UEG.paraid(f[key][1]) == UEG.paraid(para)
                println("Found matching key, processing...")
                # process_mass_ratio(f[key], isSave)
                for idk in idks
                    mass_ratio = process_mass_ratio(f[key], isSave; idk=idk)
                    push!(mass_ratios, mass_ratio)
                end
                println("done!")
            end
        end
    end
end
