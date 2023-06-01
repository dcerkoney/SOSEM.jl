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
# order = [5]  # C^{(1)}_{N≤6} includes CTs up to 5th order
# rs = [1.0]
# mass2 = [1.0]
# beta = [40.0]

# For lambda optimization
order = [4]  # C^{(1)}_{N≤5} includes CTs up to 4th order
beta = [40.0]
rs = [2.0]
mass2 = [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

# Enable/disable interaction and chemical potential counterterms
renorm_mu = true
renorm_lambda = true

# Remove Fock insertions?
isFock = false

# Select finite difference method
# method = "backward"
method = "forward"
# method = "central"

# Distinguish results with different counterterm schemes
ct_string = (renorm_mu || renorm_lambda) ? "_with_ct" : ""
if renorm_mu
    ct_string *= "_mu"
end
if renorm_lambda
    ct_string *= "_lambda"
end

# const filename = "data/data_Z$(ct_string).jld2"
# const filename = "data/data_Z$(ct_string)_kF.jld2"
const filename = "data/data_Z$(ct_string)_kF_opt.jld2"
const parafilename = "data/para.csv"

function zfactor(data, β; method="backward", verbose=false)
    @assert method in ["backward", "forward", "central"]
    @assert size(data)[1] == 3 "Expect ngrid = [-1, 0, 1]!"
    verbose && println("Calculating zfactor shift using $method FD method")
    if method == "backward"
        dsigma_dw = @. (imag(data[2, 1]) - imag(data[1, 1])) / (2π / β)  # [-1, 0]
    elseif method == "forward"
        dsigma_dw = @. (imag(data[3, 1]) - imag(data[2, 1])) / (2π / β)  # [0, 1]
    else # method == "central"
        dsigma_dw = @. (imag(data[3, 1]) - imag(data[1, 1])) / (4π / β)  # [-1, 1]
    end
    return dsigma_dw
end

function zfactor_old(data, β; verbose=false)
    @assert size(data)[1] == 2 "Expect ngrid = [0, 1]!"
    verbose && println("Calculating zfactor shift using forward FD method")
    # return @. (imag(data[2, 1]) - imag(data[1, 1])) / (2π / β)  # [-1, 0]
    return @. (imag(data[3, 1]) - imag(data[2, 1])) / (2π / β)  # [0, 1]
end

function mu(data)
    return real(data[1, 1])
end

function process(datatuple, isSave; method="backward")
    print("processing...")
    df = UEG_MC.fromFile(parafilename)
    para, ngrid, kgrid, data = datatuple
    printstyled(UEG.short(para); color=:yellow)
    println()

    # Using Z = Z_kF for all k
    @assert kgrid == [para.kF] "Expect kgrid = [kF]!"

    # Specializing Z-factor calculation based on ngrid
    if ngrid ∉ [[0, 1], [-1, 0, 1]]
        error("ngrid = $ngrid not supported!")
    end

    _mu = Dict()
    for (p, val) in data
        _mu[p] = mu(val) / (factorial(p[2]) * factorial(p[3]))
    end
    _z = Dict()
    for (p, val) in data
        if ngrid == [-1, 0, 1]
            zres = zfactor(val, para.β; method=method, verbose=true)
        else
            zres = zfactor_old(val, para.β; verbose=true)
        end
        _z[p] = zres / (factorial(p[2]) * factorial(p[3]))
    end

    for p in sort([k for k in keys(data)])
        if ngrid == [-1, 0, 1]
            zprint = zfactor(data[p], para.β; method=method)
            println("$p: μ = $(mu(data[p]))   z = $zprint")
        else
            zprint = zfactor_old(data[p], para.β)
            println("$p: μ = $(mu(data[p]))   z = $zprint")
        end
    end

    dzi, _, _ = CounterTerm.sigmaCT(para.order, _mu, _z; isfock=isFock, verbose=1)
    println("zfactor: ", dzi)

    ############# save to csv  #################
    # println(df)
    for o in keys(data)
        println("Adding order $o")
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
    if isSave
        println("Current working directory: $(pwd())")
        println("Saving results...")
        UEG_MC.toFile(df, parafilename)
        println("Done!")
    end
    return
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
        for key in keys(f)
            if UEG.paraid(f[key][1]) == UEG.paraid(para)
                process(f[key], isSave; method=method)
            end
        end
    end
end
