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

# order = [5]
order = [4]
beta = [40.0]

### rs = 1 ###
rs = [1.0]
# mass2 = [1.0]
mass2 = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

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
# rs = [5.0]
# N = 5
# mass2 = [0.8125, 0.875, 0.9375]
# N = 4
# mass2 = [0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5]
# mass2 = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]

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

# # For testing has_taylor_factors == false
# const filename = "data/before_taylor_factors/data_Z$(ct_string)_kF.jld2.bak"

# # SOSEM data
# const filename = "data/data_Z$(ct_string).jld2"

# rs = 1
# const filename = "data/data_Z$(ct_string)_kF.jld2"

# rs = 2, 3, 4, 5
# const filename = "data/data_Z$(ct_string)_kF_opt.jld2"
# const filename = "data/data_Z$(ct_string)_kF_opt_archive1.jld2"

# Old parafile (mixed ngrid)
# const parafilename = "data/para.csv"

# New parafile for ngrid = [-1, 0] only
# const parafilename = "data/para_m10.csv"


# Test of [-1, 0] and [0, 1] grids at rs = 1
const filename_m10 = "../../results/effective_mass_ratio/rs=1/ngrid_test/data_Z_with_ct_mu_lambda_kF_with_factors_m10.jld2"
const filename_0p1 = "../../results/effective_mass_ratio/rs=1/ngrid_test/data_Z_with_ct_mu_lambda_kF_with_factors_0p1.jld2"
const parafilename_m10 = "../../results/effective_mass_ratio/rs=1/ngrid_test/para_rs=1_m10.csv"
const parafilename_0p1 = "../../results/effective_mass_ratio/rs=1/ngrid_test/para_rs=1_0p1.csv"
# filename = filename_m10
# parafilename = parafilename_m10
filename = filename_0p1
parafilename = parafilename_0p1

"""
Calculate the Z-factor shift using finite-difference methods 
(i.e., by averaging data at n=0 and n=-1)
"""
function zfactor_m10(data, β)
    return @. (imag(data[2, 1]) - imag(data[1, 1])) / (2π / β)
end

"""
Calculate the Z-factor shift using finite-difference methods via only data at n=0,
using the symmetry d[n = -1] = -d[n = 0].

- `idx_n0`: index into data where n=0
"""
function zfactor_0(data, β; idx_n0)
    return @. imag(data[idx_n0, 1]) / (π / β)
end

function mu(data)
    return real(data[1, 1])
end

function process(datatuple, isSave, has_taylor_factors)
    print("processing...")
    df = UEG_MC.fromFile(parafilename)
    para, ngrid, kgrid, data = datatuple
    printstyled(UEG.short(para); color=:yellow)
    println()

    # Using Z = Z_kF for all k
    @assert kgrid == [para.kF] "Expect kgrid = [kF], kgrid = $kgrid is not supported!"

    # Specializing Z-factor calculation based on ngrid
    # if ngrid ∉ [[-1, 0], [-1, 0, 1]]
    if ngrid ∉ [[-1, 0], [0, 1], [-1, 0, 1]]
        error(
            "Expect ngrid = [1, 0], [-1, 0] or [-1, 0, 1], ngrid = $ngrid is not supported!",
        )
    end
    if ngrid == [0, 1]
        @warn "ngrid = $ngrid is deprecated, use [-1, 0] instead!"
        # zfactor = (data, β) -> zfactor_0(data, β; idx_n0=1)
        zfactor = (data, β) -> zfactor_m10(data, β)  # use [0, 1] as in old data
    elseif ngrid == [-1, 0, 1]
        @warn "Using [-1, 0] data for Z-factor calculation, ignoring last grid point!"
        zfactor = (data, β) -> zfactor_m10(data, β)
    else # ngrid == [-1, 0]
        zfactor = (data, β) -> zfactor_m10(data, β)
    end

    _mu = Dict()
    _z = Dict()
    for (p, val) in data
        if has_taylor_factors
            _mu[p] = mu(val)
            _z[p] = zfactor(val, para.β)
        else
            _mu[p] = mu(val) / (factorial(p[2]) * factorial(p[3]))
            _z[p] = zfactor(val, para.β) / (factorial(p[2]) * factorial(p[3]))
        end
    end

    for p in sort([k for k in keys(data)])
        println("$p: μ = $(mu(data[p]))   z = $(zfactor(data[p], para.β))")
    end

    dzi, dmu, dz = CounterTerm.sigmaCT(para.order, _mu, _z; isfock=isFock, verbose=1)
    println("zfactor: ", dzi)
    println("dmu: ", dmu)
    println("dz: ", dz)

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
        if haskey(f, "has_taylor_factors") == false
            error(
                "Data missing key 'has_taylor_factors', process with script 'add_taylor_factors_to_counterterm_data.jl'!",
            )
        end
        has_taylor_factors::Bool = f["has_taylor_factors"]
        for key in keys(f)
            key == "has_taylor_factors" && continue
            if UEG.paraid(f[key][1]) == UEG.paraid(para)
                htf_str = has_taylor_factors ? "with" : "without"
                print("Found data $(htf_str) Taylor factors...")
                process(f[key], isSave, has_taylor_factors)
            end
        end
    end
end
