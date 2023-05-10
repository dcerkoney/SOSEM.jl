using CodecZlib
using ElectronLiquid
using Measurements
using Printf
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
δK = 0.025

# kgrid indices & spacings
idks = 1:4
dks = [1, 2, 4, 8] * δK
dkscales = [1, 2, 4, 8]

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
# const parafilename = "para_mass_ratio_and_Z.csv"
const filename = "data_mass_ratio$(ct_string).jld2"
const parafilename = "para_mass_ratio.csv"
# const parafilename = "para.csv"

function zfactor(data, β)
    return @. (imag(data[2, 1]) - imag(data[1, 1])) / (2π / β)
end

function mu(data)
    return real(data[1, 1])
end

# function massratio(data, β, kF)
#     kamp = Σ.mesh[2][k_label]
#     z = zfactor(data, β)

#     Σ_freq = dlr_to_imfreq(to_dlr(Σ), [0, 1])
#     k1, k2 = k_label, k_label + 1
#     while abs(Σ.mesh[2][k2] - Σ.mesh[2][k1]) < δK
#         k2 += 1
#     end
#     # @assert kF < kgrid.grid[k1] < kgrid.grid[k2] "k1 and k2 are not on the same side! It breaks $kF > $(kgrid.grid[k1]) > $(kgrid.grid[k2])"
#     sigma1 = real(Σ_freq[1, k1] + Σ_ins[1, k1])
#     sigma2 = real(Σ_freq[1, k2] + Σ_ins[1, k2])
#     ds_dk = (sigma1 - sigma2) / (Σ.mesh[2][k1] - Σ.mesh[2][k2])
#     mass_ratio = 1.0 / z / (1 + me / kamp * ds_dk)
#     return mass_ratio
# end

# function process_mass_ratio(datatuple, δz, δμ, isSave)
function process_mass_ratio(datatuple, isSave)
    print("Processing mass ratio...")
    # df = UEG_MC.fromFile(parafilename)
    para, ngrid, kgrid, data = datatuple
    printstyled(UEG.short(para); color=:yellow)
    println()

    # Max order in RPT calculation
    max_order = para.order

    # Reexpand merged data in powers of μ
    ct_filename = "data_Z$(ct_string).jld2"
    z, μ = UEG_MC.load_z_mu(para; ct_filename=ct_filename)
    # Add Taylor factors to CT data
    for (p, v) in z
        z[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    for (p, v) in μ
        μ[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    δz, δμ = CounterTerm.sigmaCT(max_order - 1, μ, z; verbose=1)
    # δz, δμ = CounterTerm.sigmaCT(max_order, μ, z; verbose=1)
    println("Computed δμ: ", δμ)

    # Total Z to order max_order
    z = 1 - sum(δz)
    println("δz:\n", δz, "\n", "z = ", z)

    # println(data)

    # Convert data to a Dict of measurements with interaction counterterms merged
    _data = Dict{keytype(data),valtype(data)}()
    for (k, v) in data
        _data[k] = v / (factorial(k[2]) * factorial(k[3]))
        # data[k] = v / (factorial(k[2]) * factorial(k[3]))
    end
    merged_data = CounterTerm.mergeInteraction(_data)
    # merged_data = CounterTerm.mergeInteraction(data)
    println([k for (k, _) in merged_data])
    println(merged_data)

    # Renormalize self-energy data (ngrid, kgrid) to get Sigma to order N in RPT
    sigma_renorm =
        UEG_MC.chemicalpotential_renormalization_sigma(merged_data, δμ; max_order=max_order)
    println(sigma_renorm)

    # Aggregate the full results for Σₓ up to order N
    sigma_N_total = UEG_MC.aggregate_orders(sigma_renorm)

    println(sigma_N_total)

    # Compute m⋆/m for each order N in RPT
    mass_ratios = []
    for N in 1:max_order
        # Σ and Z to order N in RPT
        Σ_N = sigma_N_total[N]
        z_N = 1 / (1 + sum(δz[1:(N - 1)]))
        # z_N = 1 / (1 - sum(δz[1:(N - 1)]))
        @assert size(Σ_N) == (1, 5)

        # ∂ₖReΣ(k, ikₙ = 0) at the Fermi surface k = kF
        # NOTE: n1 = 0, k1 = kF, k2 = kF * (1 + δK), k2 - k1 = kF * δK
        idk = 1
        # ds_dk = (real(Σ_N[1, 1]) - real(Σ_N[1, 1 + idk])) / (kgrid[1] - kgrid[1 + idk])
        ds_dk = (real(Σ_N[1, 1 + idk]) - real(Σ_N[1, 1])) / (kgrid[1 + idk] - kgrid[1])
        @assert idk ∈ 1:4
        @assert kgrid[1] ≈ para.kF
        @assert kgrid[2] ≈ para.kF + para.kF * δK
        @assert kgrid[1 + idk] - kgrid[1] ≈ para.kF * dks[idk]

        println("\n(N=$N)")
        println("Z = $z_N")
        println("ReΣ(kF, 0) = $(real(Σ_N[1, 1]))")
        println("ReΣ(kF + $(dkscales[idk])δK * kF, 0) = $(real(Σ_N[1, 1 + idk]))")
        println("ds_dk = $ds_dk")

        # Compute m⋆/m
        mass_ratio = 1.0 / z_N / (1 + (para.me / para.kF) * ds_dk)
        println("(N = $N) m⋆/m: ", mass_ratio)
        push!(mass_ratios, mass_ratio)
    end

    if isSave
        println("Current working directory: $(pwd())")
        println("Saving data to JLD2...")
        jldopen("mass_ratio_from_sigma.jld2", "a+"; compress=true) do f
            key = "$(UEG.short(para))"
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

# function process_zmu(datatuple, isSave)
#     print("Processing Z and μ...")
#     df = UEG_MC.fromFile(parafilename)
#     para, _, _, data = datatuple
#     printstyled(UEG.short(para); color=:yellow)
#     println()

#     _mu = Dict()
#     for (p, val) in data
#         _mu[p] = mu(val) / (factorial(p[2]) * factorial(p[3]))
#     end
#     _z = Dict()
#     for (p, val) in data
#         _z[p] = zfactor(val, para.β) / (factorial(p[2]) * factorial(p[3]))
#     end

#     for p in sort([k for k in keys(data)])
#         println("$p: μ = $(mu(data[p]))   z = $(zfactor(data[p], para.β))")
#     end

#     δz, δμ = CounterTerm.sigmaCT(para.order, _mu, _z; isfock=isFock, verbose=1)
#     println("zfactor: ", δz)
#     println("mu: ", δμ)

#     ############# save to csv  #################
#     # println(df)
#     for o in keys(data)
#         println("Adding order $o")
#         # global df
#         paraid = UEG.paraid(para)
#         df = CounterTerm.appendDict(
#             df,
#             paraid,
#             Dict(
#                 "order" => o,
#                 "μ" => _mu[o].val,
#                 "μ.err" => _mu[o].err,
#                 "Σw" => _z[o].val,
#                 "Σw.err" => _z[o].err,
#             );
#             replace=true,
#         )
#     end

#     # println("new dataframe\n$df")
#     if isSave
#         println("Current working directory: $(pwd())")
#         println("Saving results...")
#         UEG_MC.toFile(df, parafilename)
#     end
#     println("Done!")
#     return δz, δμ
# end

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
        for key in keys(f)
            if UEG.paraid(f[key][1]) == UEG.paraid(para)
                println("Found matching key, processing...")
                # δz, δμ = process_zmu(f[key], isSave)
                # process_mass_ratio(f[key], δz, δμ, isSave)
                process_mass_ratio(f[key], isSave)
            end
        end
    end
end
