using CompositeGrids
using Cuba
using ElectronGas
using ElectronLiquid.UEG: ParaMC, short
using FastGaussQuadrature
using JLD2
using Lehmann
using Measurements

function fermi(k, param::ParaMC)
    return -Spectral.kernelFermiT.(-1e-8, k .^ 2 / (2 * param.me) .- param.μ, param.β)
end

"""Integrand for the Fock self-energy non-dimensionalized by E²_{TF} ~ q⁴_{TF}."""
function integrand_fock(vars, config)
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
function measure_fock(vars, obs, weights, config)
    # ExtK bin index
    ik = vars[2][1]
    obs[1][ik] += weights[1]
    return
end

function fock_self_energy_finite_temp(
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
    K = CompositeVar(R, Theta)
    # Bin in external momentum
    ExtKidx = Discrete(1, n_kgrid; alpha=alpha)
    var = (K, ExtKidx)

    res = integrate(
        integrand_fock;
        solver=solver,
        measure=measure_fock,
        neval=neval,
        print=print,
        # Config kwargs
        userdata=(param, kgrid),
        var=var,
        obs=[zeros(n_kgrid)],
    )
    return res
end

function c1d_finite_temp(kgrid, param::ParaMC; neval, print=-1, solver=:vegas, alpha=3.0)
    """Integrand for the Fock self-energy non-dimensionalized by E²_{TF} ~ q⁴_{TF}."""
    function integrand_c1d(vars, config)
        # Sample internal momentum (the Fock diagram is instantaneous)
        K, ExtKidx = vars
        R, Theta, Phi = K

        # Unpack userdata
        varK = config.userdata

        # External momentum via random index into kgrid (wlog, we place it along the x-axis)
        ik = ExtKidx[1]
        varK[1, 1] = kgrid[ik]

        phifactor = 1.0
        r = R[1] / (1 - R[1])
        θ = Theta[1]
        ϕ = Phi[1]
        varK[1, 2] = r * sin(θ) * cos(ϕ)
        varK[2, 2] = r * sin(θ) * sin(ϕ)
        varK[3, 2] = r * cos(θ)

        # phifactor *= r^2 * sin(θ) / (1 - R[1])^2
        phifactor *= sin(θ) / (1 - R[1])^2

        # Phase-space and Jacobian factor
        factor = phifactor * param.e0^2 / param.ϵ0 / (2π)^3

        # Return the non-dimensionalized Fock integrand
        eTF = param.qTF^2 / (2 * param.me)  # Thomas-Fermi energy
        kpq = norm(varK[:, 1] + varK[:, 2])  # |\mathbf{k} + \mathbf{q}|
        fermi_kpq =
            -Spectral.kernelFermiT.(-1e-8, kpq^2 / (2 * param.me) - param.μ, param.β)
        integrand = return -factor * integrand / eTF^2
    end

    """Measurement for a single diagram tree (without CTs, fixed order in V)."""
    function measure_c1d(vars, obs, weights, config)
        # ExtK bin index
        ik = vars[2][1]
        obs[1][ik] += weights[1]
        return
    end

    # Grid size
    n_kgrid = length(kgrid)

    # Setup integration variables
    R = Continuous(0.0, 1.0; alpha=alpha)
    Theta = Continuous(0.0, 1π; alpha=alpha)
    Phi = Continuous(0.0, 2π; alpha=alpha)
    K = CompositeVar(R, Theta, Phi)
    # Bin in external momentum
    ExtKidx = Discrete(1, n_kgrid; alpha=alpha)
    var = (K, ExtKidx)

    # Temp array for momenta K & Q
    varK = zeros(3, 2)

    res = integrate(
        integrand_c1d;
        solver=solver,
        measure=measure_c1d,
        neval=neval,
        print=print,
        # Config kwargs
        userdata=varK,
        var=var,
        obs=[zeros(n_kgrid)],
    )
    return res
end

function integrand_c1b02(k, param::ParaMC)
    I1 = 0
    I2 = -2 * integrand_c1d2(k, param)
    return I1 + I2
end

function integrand_c1c2(kgrid, param::ParaMC)
    return -fock_self_energy_finite_temp(kgrid, param) .^ 2
end

function integrand_c1d2(k, param::ParaMC)
    return # integrate(integrand, kp ∈ [0, ∞])
end

"""
Calculate each lowest-order SOSEM observable at finite temperature using special-purpose integrators.
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
    neval = 5e10

    # Save results to JLD2 archive?
    save = true

    # Re-expanding EOM bare Coulomb interactions V[V_λ]?
    expand_bare_interactions = false
    # TODO: Add functionality for one or two re-expanded bare interaction lines V[V_λ]
    @assert expand_bare_interactions == false

    # Save to JLD2 on main thread
    if save
        savename = "results/data/c1_lowest_order_rs=$(param.rs)_beta_ef=$(param.beta)_neval=$neval"
        jldopen("$savename.jld2", "a+"; compress=true) do f
            key = "$(short(param))"
            if haskey(f, key)
                @warn("replacing existing data for $key")
                delete!(f, key)
            end
            for (obsname, obs) in zip(["c1b0", "c1c", "c1d"], [c1b0, c1c, c1d])
                f["$key/$obsname"] = obs
            end
            f["$key/param"] = param
            return
        end
    end
end

main()
