using CodecZlib
using DelimitedFiles
using ElectronLiquid
using ElectronGas
using GreenFunc
using Interpolations
using JLD2
using Measurements
using PyCall
using SOSEM

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

# NOTE: Call from main project directory

const vzn_dir = "results/vzn_paper"

function load_csv(filename)
    # assumes csv format: (x, y)
    d = readdlm(filename, ',')
    @assert ndims(d) == 2
    xdata = d[:, 1]
    ydata = d[:, 2]
    return xdata, ydata
end

function get_Fs(rs)
    if rs < 1.0 || rs > 5.0
        return get_Fs_DMC(rs)
    else
        return get_Fs_PW(rs)
    end
end

"""
Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ from DMC data of 
Moroni, Ceperley & Senatore (1995) [Phys. Rev. Lett. 75, 689].
"""
function get_Fs_DMC(rs)
    SOSEM.@todo
end

"""
Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ via interpolation of the 
compressibility ratio data of Perdew & Wang (1992) [Phys. Rev. B 45, 13244].
"""
function get_Fs_PW(rs)
    if rs < 1.0 || rs > 5.0
        @warn "The Perdew-Wang interpolation for Fs may " *
              "be inaccurate outside the metallic regime!"
    end
    kappa0_over_kappa = 1.0025 - 0.1721rs - 0.0036rs^2
    # F⁰ₛ = κ₀/κ - 1
    return kappa0_over_kappa - 1.0
end

function get_sigma_rpa_and_rpa_fl_wn_dyn_k0(param::Parameter.Para)
    sigma_tau_dynamic, sigma_tau_instant = SelfEnergy.G0W0(param)
    sigma_dlr_dynamic = to_dlr(sigma_tau_dynamic)
    dlr = sigma_dlr_dynamic.mesh[1].dlr
    kgrid_rpa = sigma_dlr_dynamic.mesh[2]

    # Get the static and dynamic components of the self-energy
    sigma_rpa_static = sigma_tau_instant[1, :]
    sigma_rpa_wn_dynamic = to_imfreq(sigma_dlr_dynamic)

    # Positive Matsubara frequencies
    wns = dlr.ωn[dlr.ωn .≥ 0]

    # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
    rs = round(param.rs; sigdigits=13)
    Fs = get_Fs(rs)
    println("Fermi liquid parameter at rs = $(rs): Fs = $Fs")

    # Get RPA+FL self-energy
    sigma_tau_fl_dynamic, sigma_tau_fl_instant =
        SelfEnergy.G0W0(param; int_type=:ko_const, Fs=Fs)
    sigma_dlr_fl_dynamic = to_dlr(sigma_tau_fl_dynamic)
    dlr_fl = sigma_dlr_fl_dynamic.mesh[1].dlr
    kgrid_rpa_fl = sigma_dlr_fl_dynamic.mesh[2]

    # Get the static and dynamic components of the self-energy
    sigma_rpa_fl_static = sigma_tau_fl_instant[1, :]
    sigma_rpa_fl_wn_dynamic = to_imfreq(sigma_dlr_fl_dynamic)

    # Positive Matsubara frequencies
    wns_fl = dlr_fl.ωn[dlr_fl.ωn .≥ 0]
    @assert wns ≈ wns_fl

    # Nondimensionalize frequencies and self-energy by eTF
    eTF = param.qTF^2 / (2 * param.me)

    # Self-energies at k = 0 and k = kF
    sigma_rpa_wn_dynamic_k0 = sigma_rpa_wn_dynamic[:, 1][dlr.ωn .≥ 0] / eTF
    sigma_rpa_fl_wn_dynamic_k0 = sigma_rpa_fl_wn_dynamic[:, 1][dlr_fl.ωn .≥ 0] / eTF

    return wns, sigma_rpa_wn_dynamic_k0, sigma_rpa_fl_wn_dynamic_k0
end

function get_rpa_and_rpa_fl_tails_k0(wns_over_eTF, param::Parameter.Para)
    rs = round(param.rs; sigdigits=13)

    # Nondimensionalize frequencies and self-energy by eTF
    eTF = param.qTF^2 / (2 * param.me)

    # Get RPA value of the local moment at this rs from VZN paper
    k_kf_grid_rpa, c1l_rpa_over_rs2 = load_csv("$vzn_dir/c1l_over_rs2_rpa.csv")
    P = sortperm(k_kf_grid_rpa)
    c1l_rpa_over_rs2_interp =
        linear_interpolation(k_kf_grid_rpa[P], c1l_rpa_over_rs2[P]; extrapolation_bc=Line())
    c1l_rpa_vzn = c1l_rpa_over_rs2_interp(rs) * rs^2 * param.EF^2
    c1l_rpa_over_eTF2_vzn = c1l_rpa_vzn / eTF^2

    # # Non-dimensionalize bare and RPA+FL non-local moments (stored in Hartree a.u.!)
    sosem_au =
        np.load("results/data/soms_rs=$(rs)_beta_ef=40.0.npz")
    # Non-dimensionalize rs = 2 quadrature results by Thomas-Fermi energy
    # param_au = Parameter.atomicUnit(0, rs, 3)    # (dimensionless T, rs)
    param_au = Parameter.atomicUnit(1 / param.beta, rs, 3)    # (dimensionless T, rs)
    eTF_au = param_au.qTF^2 / (2 * param_au.me)

    # Bare and RPA(+FL) results (stored in Hartree a.u.)
    c1nl_rpa =
        (sosem_au.get("rpa_b") + sosem_au.get("bare_c") + sosem_au.get("bare_d")) / eTF_au^2
    c1nl_rpa_fl =
        (sosem_au.get("rpa+fl_b") + sosem_au.get("bare_c") + sosem_au.get("bare_d")) /
        eTF_au^2

    # RPA(+FL) means and error bars
    c1nl_rpa_means, c1nl_rpa_stdevs =
        Measurements.value.(c1nl_rpa), Measurements.uncertainty.(c1nl_rpa)
    c1nl_rpa_fl_means, c1nl_rpa_fl_stdevs =
        Measurements.value.(c1nl_rpa_fl), Measurements.uncertainty.(c1nl_rpa_fl)

    # Get local RPA moment from HEG SOMS data for comparison
    c1l_rpa_hartrees = sosem_au.get("rpa_a_T=0")[1]
    c1l_rpa_hartrees *= -1.0  # TODO: Fix overall sign error on local moment in get_heg_som.py
    c1l_rpa_over_eTF2 = c1l_rpa_hartrees / eTF_au^2
    c1l_rpa = 4 * c1l_rpa_hartrees
    @assert c1l_rpa_over_eTF2 ≈ c1l_rpa / eTF^2

    # Get local RPA+FL moment from HEG SOMS data (no VZN data available!)
    c1l_rpa_fl_hartrees = sosem_au.get("rpa+fl_a_T=0")[1]
    c1l_rpa_fl_hartrees *= -1.0  # TODO: Fix overall sign error on local moment in get_heg_som.py
    c1l_rpa_fl_over_eTF2 = c1l_rpa_fl_hartrees / eTF_au^2
    c1l_rpa_fl = 4 * c1l_rpa_fl_hartrees
    @assert c1l_rpa_fl_over_eTF2 ≈ c1l_rpa_fl / eTF^2

    # Compare heg_soms and VZN results for RPA local moment
    println("C⁽¹⁾ˡ (RPA, rs = $(rs)): $c1l_rpa_vzn")
    println("C⁽¹⁾ˡ / eTF² (RPA, rs = $(rs)): $c1l_rpa_over_eTF2_vzn")
    println("(SOM) C⁽¹⁾ˡ (RPA, rs = $(rs)): $c1l_rpa")
    println("(SOM) C⁽¹⁾ˡ / eTF² (RPA, rs = $(rs)): $c1l_rpa_over_eTF2")

    # Print heg_soms results for RPA+FL local moment
    println("(SOM) C⁽¹⁾ˡ (RPA+FL, rs = $(rs)): $c1l_rpa_fl")
    println("(SOM) C⁽¹⁾ˡ / eTF² (RPA+FL, rs = $(rs)): $c1l_rpa_fl_over_eTF2")

    # Tail fit to -ImΣ using RPA results
    rpa_tail_k0 = (c1l_rpa_over_eTF2 + c1nl_rpa_means[1]) ./ wns_over_eTF
    rpa_tail_k0_err = c1nl_rpa_stdevs[1] ./ wns_over_eTF
    # TODO: Mixing VZN and SOSEM dataf here---switch to pure VZN and compare!
    # rpa_tail_k0 = (c1l_rpa_over_eTF2_vzn + c1nl_rpa_means[1]) ./ wns_over_eTF
    # rpa_tail_k0_err = c1nl_rpa_stdevs[1] ./ wns_over_eTF

    # Tail fit to -ImΣ using RPA+FL results
    rpa_fl_tail_k0 = (c1l_rpa_fl_over_eTF2 + c1nl_rpa_fl_means[1]) ./ wns_over_eTF
    rpa_fl_tail_k0_err = c1nl_rpa_fl_stdevs[1] ./ wns_over_eTF

    return rpa_tail_k0, rpa_tail_k0_err, rpa_fl_tail_k0, rpa_fl_tail_k0_err
end

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    rslist_rpa_and_rpa_fl = [1.0, 2.0, 5.0]
    rs_qmc = 1.0
    beta = 40.0
    mass2 = 1.0
    solver = :vegasmc
    expand_bare_interactions = 0

    neval_c1l = 1e10
    neval_c1b0 = 5e10
    neval_c1b = 5e10
    neval_c1c = 5e10
    neval_c1d = 5e10
    neval = max(neval_c1b0, neval_c1b, neval_c1c, neval_c1d)

    # Use SOSEM data to max order N calculated in RPT
    N = 5

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    # Distinguish results with fixed vs re-expanded bare interactions
    intn_str = ""
    if expand_bare_interactions == 2
        intn_str = "no_bare_"
    elseif expand_bare_interactions == 1
        intn_str = "one_bare_"
    end

    # Filename for new JLD2 format
    filename =
    # "results/data/rs=$(rs_qmc)_beta_ef=$(beta)_" *
        "archives/rs=$(rs_qmc)_beta_ef=$(beta)_lambda=$(mass2)_" *
        "$(intn_str)$(solver)_with_ct_mu_lambda_archive1"

    # Load SOSEM data
    local param1, kgrid, k_kf_grid, c1l_N_mean, c1l_N_total, c1nl_N_total
    # Load the data for each observable
    f = jldopen("$filename.jld2", "r")
    param1 = f["c1d/N=$N/neval=$neval_c1d/param"]
    kgrid = f["c1d/N=$N/neval=$neval_c1d/kgrid"]
    c1l_N_total = f["c1l/N=$N/neval=$neval_c1l/meas"][1]
    if N == 2
        c1nl_N_total =
            f["c1b0/N=$N/neval=$neval_c1b0/meas"] +
            f["c1c/N=$N/neval=$neval_c1c/meas"] +
            f["c1d/N=$N/neval=$neval_c1d/meas"]
    else
        c1nl_N_total =
            f["c1b0/N=$N/neval=$neval_c1b0/meas"] +
            f["c1b/N=$N/neval=$neval_c1b/meas"] +
            f["c1c/N=$N/neval=$neval_c1c/meas"] +
            f["c1d/N=$N/neval=$neval_c1d/meas"]
    end
    close(f)  # close file

    # Get dimensionless k-grid (k / kF)
    k_kf_grid = kgrid / param1.kF

    # Get means and error bars from the result up to this order
    c1l_N_mean, c1l_N_stdev =
        Measurements.value.(c1l_N_total), Measurements.uncertainty.(c1l_N_total)
    c1nl_N_means, c1nl_N_stdevs =
        Measurements.value.(c1nl_N_total), Measurements.uncertainty.(c1nl_N_total)
    @assert length(k_kf_grid) == length(c1nl_N_means) == length(c1nl_N_stdevs)

    darkcolors = ["C1", "C2", "C3"]
    for (i, rs) in enumerate(rslist_rpa_and_rpa_fl)
        println("\nPlotting data for rs = $rs...")
        fig, ax = plt.subplots()
        # Get the G0W0 self-energy and corresponding DLR grid from the ElectronGas package
        # NOTE: Here we need to be careful to generate Σ_G0W0 for the *bare* theory, i.e.,
        #       to use an ElectronGas.Parameter object where Λs (mass2) is zero!
        bpara = Parameter.rydbergUnit(1 / beta, rs, 3)

        wns, sigma_rpa_wn_dyn_k0, sigma_rpa_fl_wn_dyn_k0 =
            get_sigma_rpa_and_rpa_fl_wn_dyn_k0(bpara)

        # Nondimensionalize frequencies and self-energies by eTF
        eTF = bpara.qTF^2 / (2 * bpara.me)
        wns_over_eTF = wns / eTF
        sigma_rpa_wn_dyn_k0_over_eTF = sigma_rpa_wn_dyn_k0 / eTF
        sigma_rpa_fl_wn_dyn_k0_over_eTF = sigma_rpa_fl_wn_dyn_k0 / eTF

        # Plot vs ωₙ / EF
        wns_over_EF = wns / bpara.EF

        rpa_tail_k0, rpa_tail_k0_err, rpa_fl_tail_k0, rpa_fl_tail_k0_err =
            get_rpa_and_rpa_fl_tails_k0(wns_over_eTF, bpara)

        # RPA(+FL)
        ax.plot(
            wns_over_EF,
            -imag(sigma_rpa_wn_dyn_k0_over_eTF),
            "C$(i-1)";
            label="\$RPA\$ (\$r_s=$rs\$)",
        )
        ax.plot(
            wns_over_EF,
            -imag(sigma_rpa_fl_wn_dyn_k0_over_eTF),
            "C$(i)";
            label="\$RPA+FL\$ (\$r_s=$rs\$)",
        )
        # RPA(+FL) tails
        ax.plot(
            wns_over_EF,
            rpa_tail_k0,
            "C$(i-1)";
            linestyle="dashed",
            label="\$C^{(1)}_{RPA} / \\omega_n \\epsilon_{\\mathrm{TF}}\$ (\$r_s=$rs\$)",
        )
        ax.fill_between(
            wns_over_EF,
            (rpa_tail_k0 - rpa_tail_k0_err),
            (rpa_tail_k0 + rpa_tail_k0_err);
            color="C$(i-1)",
            alpha=0.4,
        )
        ax.plot(
            wns_over_EF,
            rpa_fl_tail_k0,
            "C$(i)";
            linestyle="dashed",
            label="\$C^{(1)}_{RPA+FL} / \\omega_n \\epsilon_{\\mathrm{TF}}\$ (\$r_s=$rs\$)",
        )
        ax.fill_between(
            wns_over_EF,
            (rpa_fl_tail_k0 - rpa_fl_tail_k0_err),
            (rpa_fl_tail_k0 + rpa_fl_tail_k0_err);
            color="C$(i)",
            alpha=0.4,
        )
        # Include QMC result at rs = 1
        if rs == 1.0
            # The high-frequency tail of ImΣ(iωₙ) is -C⁽¹⁾ / ωₙ.
            # Here we non-dimensionalize the self-energy and tails in units of eTF
            tail_k0 = (c1l_N_mean + c1nl_N_means[1]) ./ wns_over_eTF
            tail_k0_err = (c1l_N_stdev + c1nl_N_stdevs[1]) ./ wns_over_eTF
            ax.plot(
                wns_over_EF,
                tail_k0,
                "k";
                linestyle="dashed",
                label="\$C^{(1)}_N / \\omega_n \\epsilon_{\\mathrm{TF}}\$ (\$r_s=1\$)",
            )
            ax.fill_between(
                wns_over_EF,
                (tail_k0 - tail_k0_err),
                (tail_k0 + tail_k0_err);
                color="k",
                alpha=0.4,
            )
        end
        ax.set_xlim(0, 50)
        if rs == 1.0
            ax.set_ylim(; bottom=0, top=0.07)
        elseif rs == 2.0
            ax.set_ylim(; bottom=0, top=0.21)
        elseif rs == 5.0
            ax.set_ylim(; bottom=0, top=0.35)
        end
        ax.legend(; loc="best")
        ax.set_xlabel("\$\\omega_n / \\epsilon_F\$")
        ax.set_ylabel(
            "\$-\\mathrm{Im}\\Sigma(k = 0, i\\omega_n) / \\epsilon_{\\mathrm{TF}}\$",
        )
        plt.tight_layout()
        fig.savefig(
            "results/self_energy_fits/sigma_tail_comparisons_" *
            "N=$(N)_rs=$(rs)_beta_ef=$(beta)_" *
            "lambda=$(mass2)_neval=$(neval)_$(solver)_k=0.pdf",
        )
    end
    plt.close("all")

    # # The same but plot wn*Σ vs C1
    # for (i, rs) in enumerate(rslist_rpa_and_rpa_fl)
    #     println("Plotting data for rs = $rs...")
    #     fig, ax = plt.subplots()
    #     # Get the G0W0 self-energy and corresponding DLR grid from the ElectronGas package
    #     # NOTE: Here we need to be careful to generate Σ_G0W0 for the *bare* theory, i.e.,
    #     #       to use an ElectronGas.Parameter object where Λs (mass2) is zero!
    #     bpara = Parameter.rydbergUnit(1 / beta, rs, 3)

    #     wns, sigma_rpa_wn_dyn_k0, sigma_rpa_fl_wn_dyn_k0 =
    #         get_sigma_rpa_and_rpa_fl_wn_dyn_k0(bpara)

    #     # Nondimensionalize frequencies and self-energies by eTF
    #     eTF = bpara.qTF^2 / (2 * bpara.me)
    #     wns_over_eTF = wns / eTF
    #     sigma_rpa_wn_dyn_k0_over_eTF = sigma_rpa_wn_dyn_k0 / eTF
    #     sigma_rpa_fl_wn_dyn_k0_over_eTF = sigma_rpa_fl_wn_dyn_k0 / eTF

    #     # Plot vs ωₙ / EF
    #     wns_over_EF = wns / bpara.EF

    #     rpa_tail_k0, rpa_tail_k0_err, rpa_fl_tail_k0, rpa_fl_tail_k0_err =
    #         get_rpa_and_rpa_fl_tails_k0(wns_over_eTF, bpara)

    #     # RPA(+FL)
    #     ax.plot(
    #         wns_over_EF,
    #         -imag(sigma_rpa_wn_dyn_k0_over_eTF),
    #         "C$(i-1)";
    #         label="\$RPA\$ (\$r_s=$rs\$)",
    #     )
    #     ax.plot(
    #         wns_over_EF,
    #         -imag(sigma_rpa_fl_wn_dyn_k0_over_eTF),
    #         "C$(i-1)";
    #         label="\$RPA+FL\$ (\$r_s=$rs\$)",
    #     )
    #     # RPA(+FL) tails
    #     ax.plot(
    #         wns_over_EF,
    #         rpa_tail_k0,
    #         "C$(i-1)";
    #         linestyle="dotted",
    #         label="\$C^{(1)}_{RPA} / \\omega_n \\epsilon_{\\mathrm{TF}}\$ (\$r_s=$rs\$)",
    #     )
    #     ax.fill_between(
    #         wns_over_EF,
    #         rpa_tail_k0 - rpa_tail_k0_err,
    #         rpa_tail_k0 + rpa_tail_k0_err;
    #         color="C$(i-1)",
    #         alpha=0.4,
    #     )
    #     ax.plot(
    #         wns_over_EF,
    #         rpa_fl_tail_k0,
    #         "C$(i-1)";
    #         linestyle="dashdot",
    #         label="\$C^{(1)}_{RPA+FL} / \\omega_n \\epsilon_{\\mathrm{TF}}\$ (\$r_s=$rs\$)",
    #     )
    #     ax.fill_between(
    #         wns_over_EF,
    #         rpa_fl_tail_k0 - rpa_fl_tail_k0_err,
    #         rpa_fl_tail_k0 + rpa_fl_tail_k0_err;
    #         color="C$(i-1)",
    #         alpha=0.4,
    #     )
    #     # Include QMC result at rs = 1
    #     if rs == 1.0
    #         # The high-frequency tail of ImΣ(iωₙ) is -C⁽¹⁾ / ωₙ.
    #         # Here we non-dimensionalize the self-energy and tails in units of eTF
    #         tail_k0 = -(c1l_N_mean + c1nl_N_means[1]) ./ wns_over_eTF
    #         tail_k0_err = (c1l_N_stdev + c1nl_N_stdevs[1]) ./ wns_over_eTF
    #         ax.plot(
    #             wns_over_EF,
    #             tail_k0,
    #             "C$(i-1)";
    #             linestyle="dashed",
    #             label="\$C^{(1)}_N / \\omega_n \\epsilon_{\\mathrm{TF}}\$ (\$r_s=1\$)",
    #         )
    #         ax.fill_between(
    #             wns_over_EF,
    #             tail_k0 - tail_k0_err,
    #             tail_k0 + tail_k0_err;
    #             color="C$(i-1)",
    #             alpha=0.4,
    #         )
    #     end
    #     ax.set_xlim(0, 50)
    #     ax.set_ylim(; bottom=0, top=1.0)
    #     ax.legend(; loc="best")
    #     ax.set_xlabel("\$\\omega_n / \\epsilon_F\$")
    #     ax.set_ylabel("\$-\\omega_n\\mathrm{Im}\\Sigma(k = 0, i\\omega_n) / \\epsilon_{\\mathrm{TF}}\$")
    #     plt.tight_layout()
    #     fig.savefig(
    #         "results/self_energy_fits/sigma_times_wn_tail_comparisons_" *
    #         "N=$(N)_rs=$(rs)_beta_ef=$(beta)_" *
    #         "lambda=$(mass2)_neval=$(neval)_$(solver)_k=0.pdf",
    #     )
    # end
    # plt.close("all")

    # TODO: Compare ReΣ(∞) and C⁽⁰⁾_{N - 1}
    return
end

main()
