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
# rs = [2.0]
# mass2 = [1.75]
# mass2 = [1.5, 1.75, 2.0]
# mass2 = [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
### rs = 3 ###
# rs = [3.0]
# mass2 = [2.0, 2.5, 3.0, 3.5, 4.0]
### rs = 4 ###
# rs = [4.0]
# mass2 = [3.0, 3.5, 4.0, 4.5, 5.0]
### rs = 5 ###
rs = [5.0]
mass2 = [4.0, 4.5, 5.0, 5.5, 6.0]

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

# const filename = "data/data_Z$(ct_string).jld2"
# const filename = "data/data_Z$(ct_string)_kF.jld2"
const filename = "data/data_Z$(ct_string)_kF_opt.jld2"
const parafilename = "data/para.csv"

"""
Calculate the Z-factor shift using finite-difference methods 
(assumes the first two grid points of data in dimension 1 are [-1, 0] or [0, 1]).
"""
function zfactor(data, β)
    return @. (imag(data[2, 1]) - imag(data[1, 1])) / (2π / β)
end

function mu(data)
    return real(data[1, 1])
end

function process(datatuple, isSave)
    print("processing...")
    df = UEG_MC.fromFile(parafilename)
    para, ngrid, kgrid, data = datatuple
    printstyled(UEG.short(para); color=:yellow)
    println()

    # Using Z = Z_kF for all k
    @assert kgrid == [para.kF] "Expect kgrid = [kF], kgrid = $kgrid is not supported!"

    # Specializing Z-factor calculation based on ngrid
    if ngrid ∉ [[-1, 0], [0, 1], [-1, 0, 1]]
        error("Expect ngrid = [-1, 0] or [-1, 0, 1], ngrid = $ngrid is not supported!")
    end
    if ngrid == [0, 1]
        @warn "ngrid = $ngrid is deprecated, use [-1, 0] instead!"
    elseif ngrid == [-1, 0, 1]
        @warn "Using [-1, 0] data for Z-factor calculation, ignoring last grid point!"
    end

    _mu = Dict()
    for (p, val) in data
        _mu[p] = mu(val) / (factorial(p[2]) * factorial(p[3]))
    end
    _z = Dict()
    for (p, val) in data
        _z[p] = zfactor(val, para.β) / (factorial(p[2]) * factorial(p[3]))
    end

    for p in sort([k for k in keys(data)])
        println("$p: μ = $(mu(data[p]))   z = $(zfactor(data[p], para.β))")
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
                process(f[key], isSave)
            end
        end
    end
end
