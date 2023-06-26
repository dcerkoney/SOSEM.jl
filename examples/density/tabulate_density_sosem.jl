using CodecZlib
using ElectronLiquid
using DataStructures
using JLD2
using SOSEM

function main()
    # Total loop order N
    orders = [1, 2, 3, 4]
    # orders = [0, 1, 2, 3, 4]
    # orders = [4, 5]
    # orders = [4]
    min_order = minimum(orders)
    max_order = maximum(orders)
    sort!(orders)

    # Settings
    alpha = 3.0
    print = 0
    solver = :vegasmc

    # Number of evals below and above kF
    # neval = 1e7
    neval = 1e9
    # neval = 1e10

    # Enable/disable interaction and chemical potential counterterms
    renorm_mu = true
    renorm_lambda = true

    # Remove Fock insertions?
    isFock = false

    # Ignore measured mu/lambda partitions?
    fix_mu = false
    fix_lambda = false

    # Inverse temperature in units of EF
    beta = 40.0

    # Compare with results from EFT_UEG?
    compare_eft = false

    # Set green4 to zero for benchmarking?
    no_green4 = false
    # no_green4 = true
    no_green4_str = no_green4 ? "_no_green4" : ""

    # Optionally give specific partition(s) to build
    # build_partitions = [(1, 2, 1), (1, 3, 0)]
    # build_partitions = [(2, 0, 2), (2, 0, 3)]
    build_partitions = nothing
    partn_string = ""
    if isnothing(build_partitions) == false
        for P in build_partitions
            partn_string *= "_" * join(P)
        end
    end

    # UEG parameters for MC integration
    loadparam = ParaMC(;
        order=max_order,
        rs=1.0,
        beta=beta,
        # mass2=1.0,
        mass2=0.6,
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
        "lambda=$(loadparam.mass2)_neval=$(neval)_$(solver)$(ct_string)" *
        "$(no_green4_str)$(partn_string)"
    println(savename)

    # NOTE: Since we use the SOSEM script for the density integration, 
    #       the Taylor factors are post-processed in all three cases
    has_taylor_factors_list = [false]
    run_types = ["EFT_UEG (post-bugfix)"]
    savenames = [savename * "_EFT_UEG_bugfix"]
    # has_taylor_factors_list = [false, false]
    # run_types = ["SOSEM", "EFT_UEG (post-bugfix)"]
    # savenames = [savename * "_SOSEM", savename * "_EFT_UEG_bugfix"]
    # has_taylor_factors_list = [false, false, false]
    # run_types = ["SOSEM", "EFT_UEG", "EFT_UEG (post-bugfix)"]
    # savenames = [savename * "_SOSEM", savename * "_EFT_UEG", savename * "_EFT_UEG_bugfix"]
    for (has_taylor_factors, run_type, savename) in
        zip(has_taylor_factors_list, run_types, savenames)
        # println(savename)
        orders, param, partitions, res = jldopen("$savename.jld2", "a+") do f
            key = "$(UEG.short(loadparam))"
            return f[key]
        end
        # println(partitions)

        # Convert results to a Dict of measurements at each order with interaction counterterms merged
        data = sort(UEG_MC.restodict(res, partitions))

        # Add overall spin summation (factor of 2) and factor of 1 / (n_μ! n_λ!)
        n0 = param.kF^3 / (3 * pi^2)
        for (k, v) in data
            # 2 from spin factor and units of n0
            if has_taylor_factors
                # data[k] = 2 * v
                data[k] = 2 * v / n0
            else
                # data[k] = 2 * v / (factorial(k[2]) * factorial(k[3]))
                data[k] = 2 * v / n0 / (factorial(k[2]) * factorial(k[3]))
            end
        end
        
        println("\n$(run_type) partitions (n_loop + 1, n_λ, n_μ):")
        for P in keys(data)
            sum(P) > 3 && continue
            println("$((P[1] + 1, P[3], P[2])):\t$(data[P][1])")
        end
    end
end

main()