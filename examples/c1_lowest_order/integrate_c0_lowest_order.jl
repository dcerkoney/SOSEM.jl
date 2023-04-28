using CodecZlib
using CompositeGrids
using ElectronGas
using ElectronLiquid
using ElectronLiquid.UEG: ParaMC, short
using JLD2
using Lehmann
using LinearAlgebra
using MCIntegration
using Measurements
using PyCall
using SOSEM.UEG_MC: lindhard

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

function fermi_k(k, param::ParaMC)
    return -Spectral.kernelFermiT.(-1e-8, k^2 / (2 * param.me) - param.μ, param.β)
end

function fermi_ek(ek, param::ParaMC)
    return -Spectral.kernelFermiT.(-1e-8, ek - param.μ, param.β)
end

"""
Exact expression for the Fock (exchange) self-energy at
T = 0 in terms of the dimensionless Lindhard function.
"""
function fock_self_energy_exact_zero_temp(k, param::ParaMC)
    # The (dimensionful) value at k = 0 is minus the Thomas-Fermi energy
    eTF = param.qTF^2 / (2 * param.me)
    return -eTF * lindhard(k / param.kF)
end

"""
Exact expression for the Fock (exchange) self-energy
in terms of the dimensionless Lindhard function.
"""
function fock_self_energy_exact(k, p::ParaMC)
    # The (dimensionful) value at k = 0 is minus the Thomas-Fermi energy
    eTF = p.qTF^2 / (2 * p.me)
    return -eTF * lindhard(k / p.kF)
end

"""Integrand for the Fock self-energy non-dimensionalized by E²_{TF} ~ q⁴_{TF}."""
function integrand_bf(vars, config)
    # Sample internal momentum (the Fock diagram is instantaneous)
    K, ExtKidx = vars
    R, Theta = K
    # R, Theta, Phi = K

    # Unpack userdata
    param, kgrid = config.userdata

    # External momentum via random index into kgrid (wlog, we place it along the x-axis)
    ik = ExtKidx[1]
    k = kgrid[ik]

    r = R[1] / (1 - R[1])
    θ = Theta[1]
    phifactor = sin(θ) / (1 - R[1])^2

    # Phase-space and Jacobian factor
    factor = phifactor * param.e0^2 / param.ϵ0 / (2π)^2

    # f_{\mathbf{k} + \mathbf{q}}
    kpq2 = k^2 + 2 * k * r * cos(θ) + r^2
    fermi_kpq = -Spectral.kernelFermiT.(-1e-8, kpq2 / (2 * param.me) - param.μ, param.β)

    # Return the non-dimensionalized Fock integrand
    eTF = param.qTF^2 / (2 * param.me)  # Thomas-Fermi energy
    return -factor * fermi_kpq / eTF
end

"""Measurement for a single diagram tree (without CTs, fixed order in V)."""
function measure_bf(vars, obs, weights, config)
    # ExtK bin index
    ik = vars[2][1]
    obs[1][ik] += weights[1]
    return
end

"""Brute force 2D integration"""
function fock_self_energy_finite_temp_bf(
    kgrid,
    param::ParaMC;
    neval,
    print=-1,
    solver=:vegas,
    alpha=3.0,
)
    # Grid size
    n_kgrid = length(kgrid)

    # Setup integration variables
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    # Phi = Continuous(0.0, 2π; alpha=alpha)
    # K = CompositeVar(R, Theta, Phi)
    K = CompositeVar(R, Theta)
    # Bin in external momentum
    ExtKidx = Discrete(1, n_kgrid; alpha=alpha)
    var = (K, ExtKidx)

    # Temp array for momenta K & Q
    # varK = zeros(2, 2)
    # varK = zeros(3, 2)
    # println(varK)

    res = integrate(
        integrand_bf;
        solver=solver,
        measure=measure_bf,
        neval=neval,
        print=print,
        # Config kwargs
        userdata=(param, kgrid),
        var=var,
        obs=[zeros(n_kgrid)],
    )
    return res
end

# """Brute force 2D integration"""
# function fock_self_energy_finite_temp_v1(
#     kgrid,
#     param::ParaMC;
#     neval,
#     print=-1,
#     solver=:vegas,
#     alpha=3.0,
# )
#     function integrand_v1(var, config)
#         # Unpack variables
#         y, theta, ExtKidx = var
#         # y, nu, ExtKidx = var

#         # External momentum magnitude via random index into kgrid
#         ik = ExtKidx[1]
#         k = kgrid[ik]

#         # q = kF y / (1 - y)
#         q = y[1] / (1 - y[1])
#         jacobian = 1 / (1 - y[1])^2

#         # Σₓ(k) ∼ ∫ q² V(q) f(k+q)
#         ekpq = (k^2 + 2 * k * q * cos(theta[1]) + q^2) / (2 * param.me) - param.μ
#         # ekpq = (k^2 + 2 * k * q * nu[1] + q^2) / (2 * param.me) - param.μ
#         prefactors = param.e0^2 / param.ϵ0 / (2π)^2

#         # Non-dimensionalize result by dividing by Thomas-Fermi energy
#         eTF = param.qTF^2 / (2 * param.me)
#         exact = -lindhard.(k / param.kF)
#         return (-prefactors * jacobian * (fermi_ek(ekpq, param) - (ekpq ≤ 0)) / eTF) + exact
#     end

#     # Setup integration variables
#     Y = Continuous(0.0, 1.0; alpha=alpha)
#     Theta = Continuous(0.0, 1π; alpha=alpha)
#     # NU = Continuous(-1.0, 1.0; alpha=alpha)
#     ExtKidx = Discrete(1, length(kgrid); alpha=alpha)  # bin external momentum
#     var = (Y, Theta, ExtKidx)
#     # var = (Y, NU, ExtKidx)

#     function measure_v1(var, obs, weights, config)
#         # ExtK bin index
#         ik = var[3][1]
#         obs[1][ik] += weights[1]
#         return
#     end
#     res = integrate(
#         integrand_v1;
#         solver=solver,
#         measure=measure_v1,
#         neval=neval,
#         print=print,
#         # Config kwargs
#         var=var,
#         obs=[zeros(length(kgrid))],
#     )
#     return res
# end

# """Brute force 2D integration"""
# function fock_self_energy_finite_temp_v2(
#     kgrid,
#     param::ParaMC;
#     neval,
#     print=-1,
#     solver=:vegas,
#     alpha=3.0,
# )
#     function integrand_v2(var, config)
#         # Unpack variables
#         y, theta, ExtKidx = var
#         # y, nu, ExtKidx = var

#         # External momentum magnitude via random index into kgrid
#         ik = ExtKidx[1]
#         k = kgrid[ik]

#         # q = kF y / (1 - y)
#         q = y[1] / (1 - y[1])
#         jacobian = 1 / (1 - y[1])^2

#         Vkpq_q2 = (param.e0^2 / param.ϵ0) * q^2 / (k^2 - 2 * k * q * cos(theta[1]) + q^2)
#         println(Vkpq_q2)

#         # Σₓ(k) ∼ ∫ q² V(q) f(k+q)
#         # ekpq = (k^2 + 2 * k * q * cos(theta[1]) + q^2) / (2 * param.me) - param.μ
#         # ekpq = (k^2 + 2 * k * q * nu[1] + q^2) / (2 * param.me) - param.μ

#         # Non-dimensionalize result by dividing by Thomas-Fermi energy
#         eTF = param.qTF^2 / (2 * param.me)
#         exact = -lindhard.(k / param.kF)
#         return (jacobian * Vkpq_q2 * (fermi_k(q, param) - (q ≤ param.kF)) / (2π)^2 / eTF) +
#                exact
#     end

#     # Setup integration variables
#     Y = Continuous(0.0, 1.0; alpha=alpha)
#     Theta = Continuous(0.0, 1π; alpha=alpha)
#     # NU = Continuous(-1.0, 1.0; alpha=alpha)
#     ExtKidx = Discrete(1, length(kgrid); alpha=alpha)  # bin external momentum
#     var = (Y, Theta, ExtKidx)
#     # var = (Y, NU, ExtKidx)

#     function measure_v2(var, obs, weights, config)
#         # ExtK bin index
#         ik = var[3][1]
#         obs[1][ik] += weights[1]
#         return
#     end
#     res = integrate(
#         integrand_v2;
#         solver=solver,
#         measure=measure_v2,
#         neval=neval,
#         print=print,
#         # Config kwargs
#         var=var,
#         obs=[zeros(length(kgrid))],
#     )
#     return res
# end

"""
Calculate Hartree-Fock exchange self-energy at finite temperature using special-purpose integration.
"""
function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    # Debug mode
    if isinteractive()
        ENV["JULIA_DEBUG"] = SOSEM
    end

    # UEG parameters for MC integration
    param = ParaMC(; rs=1.0, beta=40.0, mass2=0.0, isDynamic=false)
    @debug "β * EF = $(param.beta), β = $(param.β), EF = $(param.EF)"

    # K-mesh for measurement
    minK = 0.2 * param.kF
    Nk, korder = 4, 4
    kgrid =
        CompositeGrid.LogDensedGrid(
            :uniform,
            [0.0, 3 * param.kF],
            [param.kF],
            Nk,
            minK,
            korder,
        ).grid
    k_kf_grid = kgrid / param.kF

    # Number of integrand evaluations
    neval = 1e8
    alpha = 3.0
    solver = :vegas
    # solver = :vegasmc

    # Plot the result?
    plot = true

    # Compare integration strategies
    # res = fock_self_energy_finite_temp_v1(
    # res = fock_self_energy_finite_temp_v2(
    res = fock_self_energy_finite_temp_bf(
        kgrid,
        param;
        neval=neval,
        solver=solver,
        alpha=alpha,
    )

    # Save to JLD2 on main thread
    if !isnothing(res)
        # The nondimensionalized Fock self-energy is the negative Lindhard function
        exact = -lindhard.(kgrid / param.kF)

        # Check the MC result at k = 0 against the exact (non-dimensionalized)
        # Fock (exhange) self-energy: Σx(0) / E_{TF} = -F(0) = -1
        means, stdevs = res.mean, res.stdev
        meas = measurement.(means, stdevs)
        scores = stdscore.(meas, exact)
        score_k0 = scores[1]
        worst_score = argmax(abs, scores)

        # Finite-temperature error
        abs_err = abs.((means - exact) ./ exact)
        max_abs_err = maximum(abs_err)
        println("Absolute errors:\n$abs_err")

        # Summarize the result
        println("""
            Σₓ(k) ($solver):
               • Exact value    (k = 0): $(exact[1])
               • Measured value (k = 0): $(meas[1])
               • Standard score (k = 0): $score_k0
               • Worst standard score: $worst_score
               • Maximum absolute error: $max_abs_err
            """)

        savename =
            "sigma_fock_finite_temp_rs=$(param.rs)_" *
            "beta_ef=$(param.beta)_neval=$(neval)_$(solver)"
        jldopen("$savename.jld2", "a+"; compress=true) do f
            key = "$(UEG.short(param))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            return f[key] = (param, kgrid, res)
        end

        # Plot the result
        if plot
            fig, ax = plt.subplots()
            # Compare with exact non-dimensionalized function (-F(k / kF)) at T = 0
            k_kf_grid_fine = range(0.0; stop=maximum(k_kf_grid), length=1000)
            ax.plot(k_kf_grid_fine, -lindhard.(k_kf_grid_fine), "k"; label="(exact)")
            ax.plot(k_kf_grid, means, "o-"; color="C0", label="($solver)")
            ax.fill_between(
                k_kf_grid,
                means - stdevs,
                means + stdevs;
                color="C0",
                alpha=0.4,
            )
            ax.legend(; loc="best")
            ax.set_xlabel("\$k / k_F\$")
            ax.set_ylabel("\$\\Sigma_{F}(k) \\,/\\, \\epsilon_{\\mathrm{TF}}\$")
            ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
            plt.tight_layout()
            fig.savefig(
                "sigma_fock_finite_temp_rs=$(param.rs)_" *
                "beta_ef=$(param.beta)_neval=$(neval)_$(solver).pdf",
            )
            plt.close("all")
        end
    end
end

main()
