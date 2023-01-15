using ElectronLiquid
using Measurements
using Printf
using JLD2
using SOSEM

# Change to counterterm directory
cd("$(ENV["SOSEM_CEPH"])/examples/counterterms")

# Physical params matching data for SOSEM observables
order = [3]  # C^{(1)}_{N≤5} includes CTs up to 3rd order

# Grid-search 1: rs, mass2
# rs = LinRange(0.1, 2.0, 5)
rs = [1.0]
#mass2 = LinRange(1.0, 5.0, 5)
mass2 = [0.5, 1.5]
beta = [40.0]

# Grid-search 2: rs, beta
#rs = LinRange(0.1, 2.0, 5)
#mass2 = [<lambda_opt>]
#beta = [20.0, 40.0, 100.0]

# Post-search: finer, wider rs mesh at chosen beta and mass2
#rs = LinRange(0.1, 3.0, 21)
#mass2 = [<lambda_opt>]
#beta = [<beta_opt>]

const filename = "data_Z.jld2"
const parafilename = "para.csv"

function zfactor(data, β)
    return @. (imag(data[2, 1]) - imag(data[1, 1])) / (2π / β)
end

function mu(data)
    return real(data[1, 1])
end

function process(datatuple, isSave)
    print("processing...")
    df = UEG_MC.fromFile(parafilename)
    para, _, _, data = datatuple
    printstyled(UEG.short(para); color=:yellow)
    println()

    for p in sort([k for k in keys(data)])
        println("$p: μ = $(mu(data[p]))   z = $(zfactor(data[p], para.β))")
    end

    _mu = Dict()
    for (p, val) in data
        _mu[p] = mu(val)
    end
    _z = Dict()
    for (p, val) in data
        _z[p] = zfactor(val, para.β)
    end

    dzi, _, _ = CounterTerm.sigmaCT(para.order, _mu, _z)
    println("zfactor: ", dzi)

    ############# save to csv  #################
    # println(df)
    for o in keys(data)
        # println(o)
        # global df
        paraid = UEG.paraid(para)
        df = CounterTerm.appendDict(
            df,
            paraid,
            Dict(
                "order" => o,
                "μ" => _mu[o].val,
                "μ.err" => _mu[o].err,
                "Σw" => _z[o].val,
                "Σw.err" => _z[o].err,
            );
            replace=true,
        )
    end

    # println("new dataframe\n$df")
    return isSave && UEG_MC.toFile(df, parafilename)
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
        )

        kF = para.kF
        for key in keys(f)
            if UEG.paraid(f[key][1]) == UEG.paraid(para)
                process(f[key], isSave)
            end
        end
    end
end
