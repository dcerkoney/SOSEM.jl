using CodecZlib
using JLD2
using Measurements
using PyCall
using PyPlot
using ElectronLiquid
using ElectronGas
using TaylorSeries
using SOSEM

# For style "science"
@pyimport scienceplots

# const filename = "data_K.jld2"
# const parafilename = "para_wn_1minus0.csv"
# const Zrenorm = true    # turn on to renormalize the Z factor

const filename = "data/data_K_with_ct_mu_lambda.jld2"
const ct_filename = "data/data_Z_with_ct_mu_lambda_kF.jld2"
# const ct_filename = "data/data_Z_with_ct_mu_lambda.jld2"
const parafilename = "data/para.csv"
# const Zrenorm = false     # turn off Z renormalization 
const Zrenorm = true        # turn on Z renormalization 

# Vibrant qualitative colour scheme from https://personal.sron.nl/~pault/
const cdict = Dict([
    "blue" => "#0077BB",
    "cyan" => "#33BBEE",
    "teal" => "#009988",
    "orange" => "#EE7733",
    "red" => "#CC3311",
    "magenta" => "#EE3377",
    "grey" => "#BBBBBB",
]);

function loaddata(para, FileName=filename)
    key = UEG.short(para)
    f = jldopen(FileName, "r")
    # println(key)
    # println(keys(f))
    p, ngrid, kgrid, sigma = f[key]
    # println(sigma)
    order = p.order
    _partition = UEG.partition(para.order)
    rdata, idata = Dict(), Dict()
    for (ip, p) in enumerate(_partition)
        rdata[p] = real(sigma[p][:, :])
        idata[p] = imag(sigma[p][:, :])
    end
    return ngrid, kgrid, rdata, idata
end

function renormalize(para, sigma, Zrenorm)
    dim, β, kF = para.dim, para.β, para.kF
    mu, sw = UEG_MC.getSigma(para; parafile=parafilename)
    # mu, sw = CounterTerm.getSigma(para, parafile=parafilename)
    ############ z renormalized  ##########################
    dzi, dmu, dz = CounterTerm.sigmaCT(para.order, mu, sw)
    println(para.order)
    println(dmu)
    sigma = CounterTerm.chemicalpotential_renormalization(para.order, sigma, dmu)
    return sigma
end

function process(para, Zrenorm)
    dim, β, kF = para.dim, para.β, para.kF

    ngrid, kgrid, rdata, idata = loaddata(para, filename)

    return renormalize(para, rdata, Zrenorm),
    renormalize(para, idata, Zrenorm),
    kgrid,
    ngrid
end

function spline(x, y, e)
    # signal = pyimport("scipy.signal")
    interp = pyimport("scipy.interpolate")
    # yfit = signal.savgol_filter(y, 5, 3)
    w = 1.0 ./ e
    kidx = searchsortedfirst(x, 0.5)
    _x, _y = deepcopy(x[kidx:end]), deepcopy(y[kidx:end])
    _w = 1.0 ./ e[kidx:end]

    #enforce the boundary condition: the derivative at k=0 is zero
    pushfirst!(_x, 0.01)
    pushfirst!(_x, 0.0)
    kidx = searchsortedfirst(_x, 0.7)
    yavr = sum(y[1:kidx] .* w[1:kidx]) / sum(w[1:kidx])
    pushfirst!(_y, yavr)
    pushfirst!(_y, yavr)
    pushfirst!(_w, _w[1] * 10000)
    pushfirst!(_w, _w[1] * 10000)

    # generate knots with spline without constraints
    spl = interp.UnivariateSpline(_x, _y; w=_w, k=3)
    __x = collect(LinRange(0.0, x[end], 100))
    yfit = spl(__x)
    return __x, yfit
end

function dΣ_dϵk_n(sigma, order, kgrid, para)
    dk = [
        (sigma[o][1, 2:end] .- sigma[o][1, 1]) ./ (kgrid[2:end] .^ 2 / (2 * para.me))
        for o in 1:order
    ]
    return kgrid[2:end], sum(dk)
end

function dΣ_dω_n(sigma, order, ki, para)
    return [(sigma[o][2, ki] .- sigma[o][1, ki]) ./ (2π / para.β) for o in 1:order]
    # return [(sigma[o][1, ki]) ./ (π / para.β) for o in 1:order]
end

function get_mstar_m_over_z(sigma, order, kgrid, para; Zrenorm=true)
    kF_label = searchsortedfirst(kgrid, kF)
    mstar_m_over_z_N = []
    for ki in 1:length(kgrid)
        # Setup mass shifts as power series in ξ
        δmi = dΣ_dϵk_n(sigma, order, ki, para)
        δm = Taylor1([δmi...], order)
        # Derive z[ξ] (/ z_kF[ξ])
        mass_series = 1 + δm
        if Zrenorm
            # Setup zinv shifts as power series in ξ
            δsi = dΣ_dω_n(sigma, order, ki, para)
            δs = Taylor1([δsi...], order)
            # Derive z[ξ] (/ z_kF[ξ])
            zinv_series = 1 + δs
            zkF = zinv_series[kF_label]
            z = 1 / zinv_series
            if Zrenorm
                mass_series *= zkF
            end
        end
        # Evaluate (m*/mZ)[ξ] to order ξᴺ
        mass_coeffs = [getcoeff(mass_series, o) for o in 1:order]
        push!(mstar_m_over_z_N, sum(mass_coeffs))
    end
    return kgrid, mstar_m_over_z_N
end

function get_zinv(sigma, order, kgrid; Zrenorm=true)
    kF_label = searchsortedfirst(kgrid, kF)
    # dw = [sum(dΣ_dω_n(sigma, order, ki, para)) for ki in 1:length(kgrid)]
    zinv_N = []
    for ki in 1:length(kgrid)
        δsi = sum(dΣ_dω_n(sigma, order, ki, para))
        δs = Taylor1([δsi...], order)
        # Derive (1/z)[ξ] (⋅ z_kF[ξ])
        zinv_series = 1 + δs
        zkF = zinv_series[kF_label]
        z_series = 1 / zinv_series
        if Zrenorm
            z_series *= zkF
        end
        # Evaluate (1/z)[ξ] to order ξᴺ
        zinv_coeffs = [getcoeff(z_series, o) for o in 1:order]
        push!(zinv_N, sum(zinv_coeffs))
    end
    # zinv_N = [(sigma[o][1, :]) ./ (π / para.β) for o in 1:order]
    return kgrid, zinv_N
end

function get_z(sigma, order, kgrid; Zrenorm=true)
    kF_label = searchsortedfirst(kgrid, kF)
    zs = []
    for ki in 1:length(kgrid)
        # Setup zinv shifts as power series in ξ
        δsi = dΣ_dω_n(sigma, order, ki, para)
        δs = Taylor1([δsi...], order)
        # Derive z_k[ξ] (/ z_kF[ξ])
        zinv_series = 1 + δs
        zkF = zinv_series[kF_label]
        z_series = 1 / zinv_series
        if Zrenorm
            z_series *= zkF
        end
        # Evaluate z_k[ξ] to order ξᴺ
        z_coeffs = [getcoeff(z_series, o) for o in 1:order]
        push!(zs, sum(z_coeffs))
    end
    return kgrid, zs
end

function sw(sigma, order, kgrid; Zrenorm=true)
    kF_label = searchsortedfirst(kgrid, kF)
    # dw = [sum(dΣ_dω_n(sigma, order, ki, para)) for ki in 1:length(kgrid)]
    dw = []
    for ki in 1:length(kgrid)
        dwi = sum(dΣ_dω_n(sigma, order, ki, para))
        push!(dw, dwi)
    end
    # dw = [(sigma[o][1, :]) ./ (π / para.β) for o in 1:order]
    return kgrid, dw
end

# power series of 1/z
function sw_inv(sigma, order, kgrid; Zrenorm=true)
    kF_label = searchsortedfirst(kgrid, kF)
    dw = []
    for ki in 1:length(kgrid)
        δzi = dΣ_dω_n(sigma, order, ki, para)
        zi = Taylor1([1.0, δzi...], order)
        zkF = zi[kF_label]
        z = 1 / zi
        if Zrenorm
            z *= zkF
        end
        δz = [getcoeff(z, o) for o in 1:order]
        push!(dw, sum(δz[1:end]))
    end
    return kgrid, dw
end

function plotS_k(para, rSw_k, iSw_k, kgrid; Zrenorm=true)
    # Get order-dependent params and ReΣ / ImΣ
    paralist = para isa Vector ? para : repeat([para], para.order)
    rSw_ks = rSw_k isa Vector ? rSw_k : repeat([rSw_k], para.order)
    iSw_ks = iSw_k isa Vector ? iSw_k : repeat([iSw_k], para.order)

    para = paralist[1]  # All params except lambda are the same for each order
    dim, β, kF = para.dim, para.β, para.kF
    kF_label = searchsortedfirst(kgrid, kF)
    # zk[1] = zk[1] .- zk[1][kF_label]
    # zk[2] = zk[2] .- zk[2][kF_label]

    # plot = pyimport("plot")
    style = PyPlot.matplotlib."style"
    style.use(["science", "std-colors"])
    color = ["green", cdict["blue"], cdict["red"], "black"]
    #cmap = get_cmap("Paired")
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    # Use LaTex fonts for plots
    rcParams["font.size"] = 16
    rcParams["mathtext.fontset"] = "cm"
    # rcParams["font.family"] = "Times New Roman"

    figure(; figsize=(12, 4))
    # figure(; figsize=(8, 4))

    # Plot ∂_ϵₖ Σ(k, iω0) ≈ (Σ(k, iω0) - Σ(0, iω0)) / ϵ_k,
    # multiplying by (1/z)[ξ] if Zrenorm is true
    subplot(1, 3, 3)
    for o in 1:(para.order)
        _kgrid, dsigma_despk = dΣ_dϵk_n(rSw_ks[o], o, kgrid, para)
        y = [-m.val for m in dsigma_despk]
        e = [m.err for m in dsigma_despk]
        errorbar(
            _kgrid / kF,
            y;
            yerr=e,
            color=color[o],
            capsize=4,
            fmt="o",
            markerfacecolor="none",
            label="Order $o",
        )

        _x, _y = spline(_kgrid / kF, y, e)
        plot(_x, _y; color=color[o], linestyle="--")
    end
    xlim([kgrid[1] / kF, 2.0])
    if para.rs == 1.0
        ylim([-0.02, 0.225])
    elseif para.rs == 2.0
        ylim([-0.14, 0.225])
    end
    xlabel("\$k/k_F\$")
    if Zrenorm
        ylabel(
            "\$\\frac{1}{z} \\cdot \\left(\\Sigma(k, i\\omega_0) - \\Sigma(0, i\\omega_0)\\right)/(k^2/2m)\$",
        )
    else
        ylabel(
            "\$\\left(\\Sigma(k, i\\omega_0) - \\Sigma(0, i\\omega_0)\\right)/(k^2/2m)\$",
        )
    end
    legend()
    PyPlot.tight_layout()

    # Plot Z(k) = 1 - ∂_iω Σ(k, iω)|_{ω=0},
    # multiplying by (1/z)[ξ] if Zrenorm is true
    subplot(1, 3, 2)
    for o in 1:(para.order)
        # _kgrid, s_w = sw_inv(iSw_ks[o], o, kgrid)
        # z = [(1.0 + e) for e in s_w]
        # # z = [(1.0 - e) for e in s_w]

        # _kgrid, s_w = sw(iSw_ks[o], o, kgrid)
        # z = [1.0 / (1.0 + e) for e in s_w]
        y = [m.val for m in z]
        e = [m.err for m in z]

        kidx = searchsortedfirst(_kgrid, kF)

        println("order $o at k=$(_kgrid[1]/kF): $(z[1])")
        println("order $o at k=$(_kgrid[kidx]/kF): $(z[kidx])")

        errorbar(
            _kgrid / kF,
            y;
            yerr=e,
            color=color[o],
            capsize=4,
            fmt="o",
            markerfacecolor="none",
            label="Order $o",
        )

        _x, _y = spline(_kgrid / kF, y, e)
        plot(_x, _y; color=color[o], linestyle="--")
    end
    xlim([kgrid[1] / kF, 2.0])
    # ylim([0.98, 1.275])
    if para.rs == 1.0
        ylim([0.73, 1.02])
    elseif para.rs == 2.0
        ylim([0.78, 1.02])
    end
    xlabel("\$k/k_F\$")
    if Zrenorm
        ylabel(
            "\$\\frac{1}{z} \\cdot Z(k) = \\frac{1}{z} \\cdot \\left( 1 - \\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\\right)^{-1}\$",
        )
    else
        ylabel(
            "\$Z(k) = \\left( 1 - \\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\\right)^{-1}\$",
        )
    end
    legend()
    PyPlot.tight_layout()
    println()

    # Plot (1/Z)(k) = (1 - ∂_iω Σ(k, iω)|_{ω=0})^{-1},
    # multiplying by z[ξ] if Zrenorm is true
    subplot(1, 3, 1)
    for o in 1:(para.order)
        _kgrid, s_w = sw(iSw_ks[o], o, kgrid)
        z = [(1.0 + e) for e in s_w]
        # z = [(1.0 - e) for e in s_w]

        # _kgrid, s_w = sw(iSw_ks[o], o, kgrid)
        # z = [1.0 / (1.0 + e) for e in s_w]
        y = [z.val for z in z]
        e = [z.err for z in z]
        kidx = searchsortedfirst(_kgrid, kF)
        println("order $o at k=$(_kgrid[1]/kF): $(z[1])")
        println("order $o at k=$(_kgrid[kidx]/kF): $(z[kidx])")

        errorbar(
            _kgrid / kF,
            y;
            yerr=e,
            color=color[o],
            capsize=4,
            fmt="o",
            markerfacecolor="none",
            label="Order $o",
        )

        _x, _y = spline(_kgrid / kF, y, e)
        plot(_x, _y; color=color[o], linestyle="--")
    end
    xlim([kgrid[1] / kF, 2.0])
    if para.rs == 1.0
        ylim([0.98, 1.275])
    elseif para.rs == 2.0
        ylim([0.98, 1.18])
    end
    if Zrenorm
        ylabel(
            "\$z \\cdot Z^{-1}(k) = z \\cdot \\left(1 - \\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\\right)\$",
        )
    else
        ylabel(
            "\$Z^{-1}(k) = 1 - \\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\$",
        )
    end
    xlabel("\$k/k_F\$")
    legend()
    PyPlot.tight_layout()

    zrenormstr = Zrenorm ? "_zrenorm" : ""
    return savefig(
        "sigma_k_iw0_rs=$(para.rs)_lambda=$(para.mass2)_beta=$(para.beta)$(zrenormstr).pdf",
    )

    # zkF, μkF, has_taylor_factors =
    #     UEG_MC.load_z_mu(para; ct_filename=ct_filename, parafilename=parafilename)
    # # Add Taylor factors to CT data
    # if has_taylor_factors == false
    #     println("Adding Taylor factors to CT data...")
    #     for (p, v) in zkF
    #         zkF[p] = v / (factorial(p[2]) * factorial(p[3]))
    #     end
    #     for (p, v) in μkF
    #         μkF[p] = v / (factorial(p[2]) * factorial(p[3]))
    #     end
    # else
    #     println("Taylor factors already present in CT data...")
    # end
    # δzikF, δμkF, _ = CounterTerm.sigmaCT(para.order, μkF, zkF; verbose=1)
    # # The inverse Z-factor is (1 - δs[ξ])
    # zinvkF = Measurement{Float64}[1; accumulate(+, δzikF; init=1)]

    # println("\nδμ(k = kF):\n")
    # for (o, dmu) in enumerate(δμkF)
    #     println(" • (N = $o) $dmu")
    # end
    # println("\nδ(1/z)(k = kF):\n")
    # for (o, dzinv) in enumerate(δzikF)
    #     println(" • (N = $o) $dzinv")
    # end
    # println("\n(1/z)(k = kF):\n")
    # for (op1, zinv) in enumerate(zinvkF)
    #     println(" • (N = $(op1 - 1)) $zinv")
    # end
    # println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    para = ParaMC(; rs=1.0, beta=40.0, Fs=-0.0, order=4, mass2=1.0, isDynamic=false)
    # para = ParaMC(; rs=2.0, beta=40.0, Fs=-0.0, order=4, mass2=1.75, isDynamic=false)
    
    # Using single lambda for all orders at rs=1,2
    rSw_k, iSw_k, kgrid, ngrid = process(para, false)
    kF_label = searchsortedfirst(kgrid, para.kF)
    for k in keys(rSw_k)
        println(k, ": ", rSw_k[k][1, kF_label])
    end
    plotS_k(para, rSw_k, iSw_k, kgrid; Zrenorm=Zrenorm)

    # # Using order-by-order lambda optima for rs=3,4,5
    # para1 = ParaMC(; rs=4.0, beta=40.0, Fs=-0.0, order=4, mass2=0.5625, isDynamic=false)
    # para2 = ParaMC(; rs=4.0, beta=40.0, Fs=-0.0, order=4, mass2=0.625, isDynamic=false)
    # para3 = ParaMC(; rs=4.0, beta=40.0, Fs=-0.0, order=4, mass2=0.75, isDynamic=false)
    # para4 = ParaMC(; rs=4.0, beta=40.0, Fs=-0.0, order=4, mass2=1.0, isDynamic=false)
    # paralist = [para1, para2, para3, para4]
    # local kgrid, ngrid
    # rSw_ks = []
    # iSw_ks = []
    # for para in paralist
    #     rSw_k, iSw_k, kgrid, ngrid = process(para, false)
    #     push!(rSw_ks, rSw_k)
    #     push!(iSw_ks, iSw_k)
    # end
    # kF_label = searchsortedfirst(kgrid, para.kF)
    # for k in keys(rSw_k)
    #     println(k, ": ", rSw_k[k][1, kF_label])
    # end
    # plotS_k(paralist, rSw_ks, iSw_ks, kgrid, Zrenorm)
end