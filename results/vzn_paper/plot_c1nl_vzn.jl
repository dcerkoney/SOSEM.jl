using CodecZlib
using DataFrames
using DelimitedFiles
using ElectronGas
using ElectronLiquid
using Interpolations
using JLD2
using Lehmann
using LsqFit
using Measurements
using Parameters
using PyCall
using SOSEM

const vzn_dir = "results/vzn_paper"

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt
@pyimport mpl_toolkits.axes_grid1.inset_locator as il

function load_csv(filename)
    # assumes csv format: (x, y)
    d = readdlm(filename, ',')
    @assert ndims(d) == 2
    xdata = d[:, 1]
    ydata = d[:, 2]
    return xdata, ydata
end

function average(filename)
    # assumes csv format: (x, y)
    d = readdlm(filename, ',')
    @assert ndims(d) == 2
    ydata = d[:, 2]
    return sum(ydata) / length(ydata)
end

function load_c1_hf(filename="$vzn_dir/c1_hf.csv")
function load_c1_obqmc(filename="$vzn_dir/c1_ob-qmc.csv")
function get_c1l_qmc(filename="$vzn_dir/c1_local_qmc.csv")
function get_c1l_vs(filename="$vzn_dir/c1_local_vs.csv")

# function get_c1l_hf(filename="$vzn_dir/c1l_over_rs2_rpa.csv")
#     # Grab local value at rs = 5 from plot vs rs
#     local c1l_hf
#     for (rs, c1l_hf_over_rs2) in csv
#         if rs ≈ 5
#             c1l_hf = rs^2 * c1l_hf_over_rs2
#         end
#     end
#     return c1l_hf
# end

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    # rs = 5 for VZN SOSEM plots
    rs_vzn = 5.0

    # Load full SOSEM data in HF and OB-QMC approximations
    c1_hf    = load_csv("$vzn_dir/c1_hf.csv")
    c1_obqmc = load_csv("$vzn_dir/c1_ob-qmc.csv")
    println("C⁽¹⁾ (HF)\n: $c1_hf")
    println("C⁽¹⁾ (OB-QMC)\n: $c1_obqmc")

    # Load QMC local moment
    c1l_qmc = average("$vzn_dir/c1_local_qmc.csv")
    # Estimate HF local moment using linear interpolation on data vs rs
    c1l_over_rs2_hf = load_csv("$vzn_dir/c1l_over_rs2_rpa.csv")
    c1l_hf = rs_vzn^2 * c1l_over_rs2_hf[:, end]
    println("C⁽¹⁾ˡ (HF): $c1l_hf")
    println("C⁽¹⁾ˡ (QMC): $c1l_qmc")

    # Subtract local contribution to obtain non-local moment
    # TODO: verify that this is the correct local term;
    #       do we need to extract it from c1l vs rs data instead?
    c1nl_qmc = c1_obqmc .- c1l_qmc

    # Same as above for HF result; use exact HF value at k = 0 to deduce the local moment measured by VZN in the HF approximation
    c1nl_hf = c1_hf .- c1l_hf

    println("C⁽¹⁾ⁿˡ (HF)\n: $c1nl_hf")
    println("C⁽¹⁾ⁿˡ (QMC)\n: $c1nl_qmc")

    return
end

main()