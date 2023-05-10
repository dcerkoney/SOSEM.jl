using SOSEM
using CodecZlib
using JLD2
using FeynmanDiagram
using ElectronLiquid
using Measurements
using DataStructures

function c1l2_over_eTF2_vlambda_vlambda(l)
    m = sqrt(l)
    I1 = (l / (l + 4) + log((l + 4) / l) - 1) / 4
    I2 = (l^2 / (l + 4) - (l + 4) + 2l * log((l + 4) / l)) / 48
    I3 = (π / 2m + 2 / (l + 4) - atan(2 / m) / m) / 3
    # I1 = (l / (l + 4) - log(l / (l + 4)) - 1) / 4
    # I2 = (l^2 / (l + 4) - (l + 4) - 2l * log(l / (l + 4))) / 64
    # I3 = 2(2 / (l + 4) - atan(2 / m) / m) / 3
    return (I1 + I2 + I3)
end

"""l = λ / kF^2"""
function c1l2_over_eTF2_v_vlambda(l)
    m = sqrt(l)
    return (π / 3m - 1 / 12) + (l / 12 + 1) * log((4 + l) / l) / 4 - (2 / 3m) * atan(2 / m)
end

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
    orders = [1, 2, 3, 4]
    # orders = [1]
    sort!(orders)

    # Plot total results for orders min_order_plot ≤ ξ ≤ max_order_plot
    min_order = minimum(orders)
    max_order = maximum(orders)
    min_order_plot = 1
    max_order_plot = 4

    # Settings
    alpha = 3.0
    print = 0
    solver = :vegasmc

    # Number of evals below and above kF
    neval = 5e10

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Remove Fock insertions?
    isFock = false

    # UEG parameters for MC integration
    loadparam = ParaMC(;
        order=max_order,
        rs=1.0,
        beta=40.0,
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
        "results/data/c1l_n=$(loadparam.order)_rs=$(loadparam.rs)_beta_ef=$(loadparam.beta)_" *
        "lambda=$(loadparam.mass2)_neval=$(neval)_$(solver)$(ct_string)"
    println(savename)
    orders, param, partitions, res = jldopen("$savename.jld2", "a+") do f
        key = "$(UEG.short(loadparam))"
        return f[key]
    end
    println(partitions)

    # Convert results to a Dict of measurements at each order with interaction counterterms merged
    data = UEG_MC.restodict(res, partitions)
    println(data)

    # Add Taylor factors 1 / (n_μ! n_λ!)
    for (k, v) in data
        data[k] = v / (factorial(k[2]) * factorial(k[3]))
        # # Extra minus sign for missing factor of (-1)^F = -1?
        # data[k] = [-v / (factorial(k[2]) * factorial(k[3]))]
    end
    println(data)

    println("\nPartitions (n_loop, n_λ, n_μ):\n")
    for P in keys(data)
        println(" • Partition $P:\t$(data[P][1])")
    end

    # Merge interaction and loop orders
    merged_data = CounterTerm.mergeInteraction(data)

    println("\nInteraction-merged partitions (n_loop, n_μ, n_λ):\n")
    for Pm in keys(merged_data)
        println(" • Partition $Pm:\t$(merged_data[Pm][1])")
    end

    # Load counterterm data
    ct_filename = "examples/counterterms/data_Z$(ct_string_short).jld2"
    z, μ = UEG_MC.load_z_mu(param; ct_filename=ct_filename)
    # Add Taylor factors to CT data
    for (p, v) in z
        z[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    for (p, v) in μ
        μ[p] = v / (factorial(p[2]) * factorial(p[3]))
    end
    δz, δμ = CounterTerm.sigmaCT(max_order, μ, z; isfock=isFock, verbose=1)
    println("Computed δμ: ", δμ)

    # Reexpand merged data in powers of μ
    c1l = UEG_MC.chemicalpotential_renormalization(
        merged_data,
        δμ;
        n_min=1,
        min_order=min_order,
        max_order=max_order,
    )
    c1l_total = UEG_MC.aggregate_orders(c1l)

    println("\nOrder-by-order local moment contributions:\n")
    for o in keys(c1l)
        println(" • n = $o:\t$(c1l[o][1])")
    end

    println("\nTotal local moment vs order N:\n")
    for o in sort(collect(keys(c1l_total)))
        println(" • N = $o:\t$(c1l_total[o][1])")
    end

    if min_order == 1
        # calc = data[(1, 0, 0)][1]
        calc = c1l_total[1][1]
        exact = c1l2_over_eTF2_v_vlambda(param.mass2 / param.kF^2)
        zscore = stdscore(calc, exact)
        println("\nExact c1l2:\t$exact")
        println("Computed c1l2:\t$calc")
        println("Standard score for c1l2:\t$zscore")
        # @assert zscore ≤ 20
    end

    println("Done!")
    return
end

main()
