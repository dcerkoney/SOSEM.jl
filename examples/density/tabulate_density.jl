using SOSEM
using CodecZlib
using JLD2
using FeynmanDiagram
using ElectronLiquid
using Measurements
using DataStructures

"""MC tabulation of the total density."""
function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    # Debug mode
    if isinteractive()
        ENV["JULIA_DEBUG"] = Main
    end

    # Total loop order N
    # orders = [0, 1, 2, 3]
    orders = [1, 2, 3]
    min_order = minimum(orders)
    max_order = maximum(orders)
    sort!(orders)

    # Settings
    alpha = 3.0
    print = 0
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 1e9

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = false

    # Remove Fock insertions?
    isFock = true

    # Ignore measured mu/lambda partitions?
    fix_mu = false
    fix_lambda = true

    # Inverse temperature in units of EF
    beta = 40.0

    # Compare with results from EFT_UEG?
    compare_eft = false

    # Set green4 to zero for benchmarking?
    no_green4 = false
    no_green4_str = no_green4 ? "_no_green4" : ""

    # UEG parameters for MC integration
    loadparam = ParaMC(;
        order=max_order,
        rs=1.0,
        beta=beta,
        mass2=1.0,
        isDynamic=false,
        isFock=isFock,  # remove Fock insertions
    )
    @debug "β * EF = $(loadparam.beta), β = $(loadparam.β), EF = $(loadparam.EF)"

    # Distinguish results with different counterterm schemes used in the original run
    ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
    if renorm_mu
        ct_string *= "_mu"
    end
    if renorm_lambda
        ct_string *= "_lambda"
    end
    ct_string_short = ct_string
    if isFock
        ct_string *= "_noFock"
    end

    # Load the raw data
    savename =
        "results/data/density_n=$(loadparam.order)_rs=$(loadparam.rs)_beta_ef=$(loadparam.beta)_" *
        "lambda=$(loadparam.mass2)_neval=$(neval)_$(solver)$(ct_string)$(no_green4_str)"
    println(savename)
    orders, param, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(loadparam))"
        return f[key]
    end

    if compare_eft
        # Load the EFT_UEG data (NOTE: EFT data is already in Dict format!)
        savename = "results/data/data_n$(no_green4_str)"
        param_eft, data_eft = jldopen("$savename.jld2", "a+") do f
            key = "$(UEG.short(loadparam))"
            return f[key]
        end
        println(param_eft)
        println(data_eft)
    end

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = sort(UEG_MC.restodict(res, partitions))
    if compare_eft
        data_eft = sort(data_eft)
    end

    # Add overall spin summation (factor of 2)
    n0 = param.kF^3 / (3 * pi^2)
    for (k, v) in data
        data[k] = [2 * v / n0]
    end
    if compare_eft
        for (k, v) in data_eft
            data_eft[k] = [2 * v / n0]
        end
    end

    println(n0)
    println("\nPartitions SOSEM (n_loop, n_μ, n_λ):\n")
    for Pm in keys(data)
        # fix_mu && Pm[2] > 0 && continue
        # Pm[3] > 0 && continue  # No lambda counterterms
        println("$((Pm[1]+1, Pm[3], Pm[2])):\t$(data[Pm][1])")
    end
    println("\nPartitions EFT (n_loop, n_μ, n_λ):\n")
    if compare_eft
        for Pm in keys(data_eft)
            # fix_mu && Pm[2] > 0 && continue
            # Pm[3] > 0 && continue  # No lambda counterterms
            println("$((Pm[1]+1, Pm[3], Pm[2])):\t$(data_eft[Pm][1])")
        end
    end
    println("Done!")
    return

    # Zero out partitions with mu renorm if present (fix mu)
    if renorm_mu == false || fix_mu
        for P in keys(data)
            if P[2] > 0
                println("Fixing mu without lambda renorm, ignoring n_k partition $P")
                data[P] = zero(data[P])
            end
        end
        if compare_eft
            for P in keys(data_eft)
                if P[2] > 0
                    println("Fixing mu without lambda renorm, ignoring n_k partition $P")
                    data_eft[P] = zero(data_eft[P])
                end
            end
        end
    end
    # Zero out partitions with lambda renorm if present (fix lambda)
    if renorm_lambda == false || fix_lambda
        for P in keys(data)
            if P[3] > 0
                println("No lambda renorm, ignoring n_k partition $P")
                data[P] = zero(data[P])
            end
        end
        if compare_eft
            for P in keys(data_eft)
                if P[3] > 0
                    println("No lambda renorm, ignoring n_k partition $P")
                    data_eft[P] = zero(data_eft[P])
                end
            end
        end
    end

    # Set bare result manually using exact non-interacting density if not available
    n0 = param.kF^3 / (3 * pi^2)
    if haskey(data, (0, 0, 0)) == false
        data[(0, 0, 0)] = [measurement(n0, 0.0)]
    end
    if haskey(data_eft, (0, 0, 0)) == false
        data_eft[(0, 0, 0)] = [measurement(n0, 0.0)]
    end
    if min_order == 0
        n0_calc = data[(0, 0, 0)][1]
        zscore = stdscore(n0_calc, n0)
        compare_eft && println("\n(SOSEM)")
        println("\nExact n0:\t$n0")
        println("Computed n0:\t$n0_calc")
        println("Standard score for n0:\t$zscore")
        @assert zscore ≤ 20
        if compare_eft
            n0_calc_eft = data_eft[(0, 0, 0)][1]
            zscore_eft = stdscore(n0_calc_eft, n0)
            println("\n(EFT_UEG)")
            println("\nExact n0:\t$n0")
            println("Computed n0:\t$n0_calc_eft")
            println("Standard score for n0:\t$zscore_eft")
            @assert zscore_eft ≤ 20
        end
    end

    # Merge interaction and loop orders
    merged_data = CounterTerm.mergeInteraction(data)
    if compare_eft
        merged_data_eft = CounterTerm.mergeInteraction(data_eft)
    end

    # Measure partitions in units of the non-interacting density
    scaled_data = Dict()
    for (k, v) in data
        scaled_data[k] = v / n0
    end
    scaled_merged_data = Dict()
    for (k, v) in merged_data
        scaled_merged_data[k] = v / n0
    end
    if compare_eft
        scaled_data_eft = Dict()
        for (k, v) in data_eft
            scaled_data_eft[k] = v / n0
        end
        scaled_merged_data_eft = Dict()
        for (k, v) in merged_data_eft
            scaled_merged_data_eft[k] = v / n0
        end
    end

    # Use Pengcheng's convention
    scaled_merged_shifted_data = OrderedDict()
    for (k, v) in merged_data
        scaled_merged_shifted_data[(k[1] + 1, k[2])] = v / n0
    end
    sort!(scaled_merged_shifted_data)
    scaled_merged_shifted_data_eft = OrderedDict()
    for (k, v) in merged_data_eft
        scaled_merged_shifted_data_eft[(k[1] + 1, k[2])] = v / n0
    end
    sort!(scaled_merged_shifted_data_eft)

    compare_eft && println("\n(SOSEM)")
    println("\nDensity partitions in units of n0 (n_loop, n_μ, n_λ):\n")
    for P in keys(scaled_data)
        fix_mu && P[2] > 0 && continue
        fix_lambda && P[3] > 0 && continue
        println(" • Partition $P:\t$(scaled_data[P][1])")
    end
    println("\nInteraction-merged density partitions in units of n0 (n_loop, n_μ, n_λ):\n")
    for Pm in keys(scaled_merged_shifted_data)
        fix_mu && Pm[2] > 0 && continue
        println(" • Partition $Pm:\t$(scaled_merged_shifted_data[Pm][1])")
    end

    if compare_eft
        println("\n(EFT_UEG)")
        println("\nDensity partitions in units of n0 (n_loop, n_μ, n_λ):\n")
        for P in keys(scaled_data_eft)
            fix_mu && P[2] > 0 && continue
            fix_lambda && P[3] > 0 && continue
            println(" • Partition $P:\t$(scaled_data_eft[P][1])")
        end
        println(
            "\nInteraction-merged density partitions in units of n0 (n_loop, n_μ, n_λ):\n",
        )
        for Pm in keys(scaled_merged_shifted_data_eft)
            fix_mu && Pm[2] > 0 && continue
            println(" • Partition $Pm:\t$(scaled_merged_shifted_data_eft[Pm][1])")
        end
    end

    # Load counterterm data
    ct_filename = "examples/counterterms/data_Z$(ct_string_short).jld2"
    z, μ = UEG_MC.load_z_mu(param; ct_filename=ct_filename)
    # Zero out partitions with mu renorm if present (fix mu)
    if renorm_mu == false || fix_mu
        for P in keys(μ)
            if P[2] > 0
                println("Fixing mu without lambda renorm, ignoring μ partition $P")
                μ[P] = zero(μ[P])
            end
        end
    end
    # Zero out partitions with lambda renorm if present (fix lambda)
    if renorm_lambda == false || fix_lambda == true
        for P in keys(μ)
            if P[3] > 0
                println("No lambda renorm, ignoring μ partition $P")
                μ[P] = zero(μ[P])
            end
        end
    end
    δz, δμ = CounterTerm.sigmaCT(max_order, μ, z; isfock=isFock, verbose=1)
    println("Computed δμ: ", δμ)

    # Reexpand merged data in powers of μ
    δn = UEG_MC.chemicalpotential_renormalization_green(
        merged_data,
        δμ;
        min_order=0,
        max_order=max_order,
    )
    scaled_δn = Dict()
    for (k, v) in δn
        scaled_δn[k] = v / n0
    end
    if compare_eft
        δn_eft = UEG_MC.chemicalpotential_renormalization_green(
            merged_data_eft,
            δμ;
            min_order=0,
            max_order=max_order,
        )
        scaled_δn_eft = Dict()
        for (k, v) in δn_eft
            scaled_δn_eft[k] = v / n0
        end
    end

    compare_eft && println("\n(SOSEM)")
    println("\nOrder-by-order density shifts (β ϵF = $beta):\n")
    for o in keys(δn)
        # Luttinger's theorem ⟹ δn_0 = n0, δn_m ≡ 0 ∀ m > 0
        δn_exact = o == 0 ? n0 : 0.0
        rounded_zscore = round(stdscore(δn[o][1], δn_exact); digits=3)
        println(" • Order $o:\t$(δn[o][1])\t(z-score = $rounded_zscore)")
    end

    if compare_eft
        println("\n(EFT_UEG)")
        println("\nOrder-by-order density shifts (β ϵF = $beta):\n")
        for o in keys(δn_eft)
            # Luttinger's theorem ⟹ δn_0 = n0, δn_m ≡ 0 ∀ m > 0
            δn_exact = o == 0 ? n0 : 0.0
            rounded_zscore_eft = round(stdscore(δn_eft[o][1], δn_exact); digits=3)
            println(" • Order $o:\t$(δn_eft[o][1])\t(z-score = $rounded_zscore_eft)")
        end
    end

    # println("\nOrder-by-order density shifts in units of n0:\n")
    # for o in keys(scaled_δn)
    #     println(" • Order $o:\t$(scaled_δn[o])")
    # end
end

main()
