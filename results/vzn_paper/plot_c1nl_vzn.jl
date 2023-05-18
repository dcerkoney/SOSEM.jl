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

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

const vzn_dir = "results/vzn_paper"

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

# get_c1l_qmc() = load_csv(filename="$vzn_dir/c1_local_qmc.csv")
# load_c1_hf() = load_csv(filename="$vzn_dir/c1_hf.csv")
# load_c1_obqmc() = load_csv(filename="$vzn_dir/c1_ob-qmc.csv")
# function get_c1l_vs(filename="$vzn_dir/c1_local_vs.csv")

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

    # Plot the results?
    plot = false

    # rs = 5 for VZN SOSEM plots
    rs_vzn = 5.0

    param = UEG.ParaMC(; rs=5.0, beta=40.0, isDynamic=false)
    param_rs5 = param
    param_rs1 = UEG.ParaMC(; rs=1.0, beta=40.0, isDynamic=false)

    # Load QMC local moment
    c1l_qmc_over_EF2 = average("$vzn_dir/c1_local_qmc.csv")
    println("C⁽¹⁾ˡ (QMC): $c1l_qmc_over_EF2")

    # Load full SOSEM data in HF and OB-QMC approximations
    k_kf_grid_hf, c1_hf_over_EF2 = load_csv("$vzn_dir/c1_hf.csv")
    k_kf_grid_qmc, c1_qmc_over_EF2 = load_csv("$vzn_dir/c1_ob-qmc.csv")
    println("C⁽¹⁾ (HF):\n $c1_hf_over_EF2")
    println("C⁽¹⁾ (QMC):\n $c1_qmc_over_EF2")

    # Subtract local contribution to obtain HF/QMC non-local moments
    # NOTE: VZN define C⁽¹⁾(HF) as the sum of the HF non-local moment,
    #       and the OB-QMC local moment (since C⁽¹⁾ˡ(HF) is divergent)
    c1nl_qmc_over_EF2 = c1_qmc_over_EF2 .- c1l_qmc_over_EF2
    c1nl_hf_over_EF2 = c1_hf_over_EF2 .- c1l_qmc_over_EF2

    println("C⁽¹⁾ⁿˡ (HF):\n $c1nl_hf_over_EF2")
    println("C⁽¹⁾ⁿˡ (QMC):\n $c1nl_qmc_over_EF2")

    # Change from units of eF^2 to eTF^2
    eTF = param.qTF^2 / (2 * param.me)
    c1l_qmc_over_eTF2 = c1l_qmc_over_EF2 * (param.EF / eTF)^2
    c1nl_qmc_over_eTF2 = c1nl_qmc_over_EF2 * (param.EF / eTF)^2
    c1nl_hf_over_eTF2 = c1nl_hf_over_EF2 * (param.EF / eTF)^2
    println("C⁽¹⁾ˡ / eTF² (QMC, rs = 5, direct): $c1l_qmc_over_eTF2")

    k_kf_grid_qmc, c1l_qmc_over_rs2 = load_csv("$vzn_dir/c1l_over_rs2_qmc.csv")
    P = sortperm(k_kf_grid_qmc)
    c1l_qmc_over_rs2_interp = linear_interpolation(k_kf_grid_qmc[P], c1l_qmc_over_rs2[P]; extrapolation_bc=Line())
    c1l_qmc_rs1 = c1l_qmc_over_rs2_interp(1.0)
    c1l_qmc_rs5 = 25 * c1l_qmc_over_rs2_interp(5.0)

    eTF_rs1 = param_rs1.qTF^2 / (2 * param_rs1.me)
    eTF_rs5 = param_rs5.qTF^2 / (2 * param_rs5.me)
    c1l_qmc_rs1_over_eTF2 = c1l_qmc_rs1 * (param_rs1.EF / eTF_rs1)^2
    c1l_qmc_rs5_over_eTF2 = c1l_qmc_rs5 * (param_rs5.EF / eTF_rs5)^2
    println("C⁽¹⁾ˡ (QMC, rs = 1): $c1l_qmc_rs1")
    println("C⁽¹⁾ˡ (QMC, rs = 5): $c1l_qmc_rs5")
    println("C⁽¹⁾ˡ / eTF² (QMC, rs = 1): $c1l_qmc_rs1_over_eTF2")
    println("C⁽¹⁾ˡ / eTF² (QMC, rs = 5): $c1l_qmc_rs5_over_eTF2")
    return

    if plot
        # Use LaTex fonts for plots
        plt.rc("text"; usetex=true)
        plt.rc("font"; family="serif")
        # Plot the results in units of eTF2 using matplotlib
        fig, ax = plt.subplots()
        ax.plot(k_kf_grid_hf, c1nl_hf_over_eTF2, "o-"; markersize=2, label="HF")
        ax.plot(k_kf_grid_qmc, c1nl_qmc_over_eTF2, "o-"; markersize=2, label="QMC")
        ax.set_xlim(0, 3)
        ax.set_xlabel("\$k / k_F\$")
        ax.set_ylabel("\$C^{(1)nl} / \\epsilon^2_{\\mathrm{TF}}\$")
        ax.legend(; loc="best")
        plt.tight_layout()
        plt.show()
        fig.savefig("$vzn_dir/c1nl_vzn.pdf")
    end
    return
end

main()