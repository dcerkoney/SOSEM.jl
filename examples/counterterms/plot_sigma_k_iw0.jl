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

# const filename = "data/data_K_with_ct_mu_lambda.jld2"
# const ct_filename = "data/data_Z_with_ct_mu_lambda_kF.jld2"
# const ct_filename = "data/data_Z_with_ct_mu_lambda_kF_opt.jld2"
# const ct_filename = "data/data_Z_with_ct_mu_lambda.jld2"
# const parafilename = "data/para.csv"
# const parafilename = "data/para_0m1.csv"

# const Zrenorm = false     # turn off Z renormalization 
const Zrenorm = true        # turn on Z renormalization 

# Test of [-1, 0] and [0, 1] grids at rs = 1
const filename_m10 = "../../results/effective_mass_ratio/rs=1/ngrid_test/data_K_with_ct_mu_lambda_with_factors_m10.jld2"
const filename_0p1 = "../../results/effective_mass_ratio/rs=1/ngrid_test/data_K_with_ct_mu_lambda_with_factors_0p1.jld2"
const parafilename_m10 = "../../results/effective_mass_ratio/rs=1/ngrid_test/para_rs=1_m10.csv"
const parafilename_0p1 = "../../results/effective_mass_ratio/rs=1/ngrid_test/para_rs=1_0p1.csv"
filename = filename_m10
parafilename = parafilename_m10
# filename = filename_0p1
# parafilename = parafilename_0p1

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
        # rdata[p] = real(sigma[p][:, :])
        # idata[p] = imag(sigma[p][:, :])
    end
    return ngrid, kgrid, rdata, idata
end

function renormalize(para, sigma, Zrenorm)
    dim, β, kF = para.dim, para.β, para.kF

    # # Add Taylor factors to CT data
    # # ct_filename = "data/data_Z_with_ct_mu_lambda_kF.jld2"
    # ct_filename = "data/data_Z_with_ct_mu_lambda_kF_opt.jld2"
    # # ct_filename = "data/data_Z_with_ct_mu_lambda.jld2"
    # # ct_filename = "data/data_Z_with_ct_mu_lambda_kF.jld2"
    # # ct_filename = "data/data_Z_with_ct_mu_lambda_kF_opt_archive1.jld2"
    # z, μ, has_taylor_factors_zmu = UEG_MC.load_z_mu_old(para; ct_filename=ct_filename)
    # if true
    # # if has_taylor_factors_zmu == false
    #     for (p, v) in z
    #         z[p] = v / (factorial(p[2]) * factorial(p[3]))
    #     end
    #     for (p, v) in μ
    #         μ[p] = v / (factorial(p[2]) * factorial(p[3]))
    #     end
    # end
    # # Get μ and z counterterms
    # δzi, δμ, δz = CounterTerm.sigmaCT(max_order, μ, z; verbose=1)
    # println("δzi:\n", δzi, "\nδμ:\n", δμ, "\nδz:\n", δz)

    mu, sw = UEG_MC.getSigma(para; parafile=parafilename)
    δzi, δμ, δz = CounterTerm.sigmaCT(para.order, mu, sw)
    println("δzi:\n", δzi, "\nδμ:\n", δμ, "\nδz:\n", δz, "\n")

    if para.rs == 1.0 && para.mass2 == 1.0 && contains(parafilename, "m10")
        @assert stdscore(δzi[4] - (0.04071 ± 0.00045), 0) < 5
        @assert stdscore(δμ[4] - (-0.05946 ± 0.00089), 0) < 5
        @assert stdscore(δz[4] - (-0.03220 ± 0.00450), 0) < 5
    elseif para.rs == 1.0 && para.mass2 == 1.0 && contains(parafilename, "0p1")
        @assert stdscore(δzi[4] - (0.00801 ± 0.00018), 0) < 5
        @assert stdscore(δμ[4] - (-0.05956 ± 0.00028), 0) < 5
        @assert stdscore(δz[4] - (-0.00523 ± 0.00018), 0) < 5
    end
    # error("done testing")    

    # sigma = CounterTerm.chemicalpotential_renormalization(para.order, sigma, δμ)
    # return sigma

    # sigma_R = sigma * z if zrenorm is True, else sigma_R = sigma
    # sigma_R = CounterTerm.renormalization(para.order, sigma, δμ, δz; zrenorm=Zrenorm)
    sigma_R = CounterTerm.renormalization(para.order, sigma, δμ, δz; zrenorm=false)
    return sigma_R
end

function process(para, Zrenorm)
    dim, β, kF = para.dim, para.β, para.kF

    ngrid, kgrid, rdata, idata = loaddata(para, filename)

    rdata_R = renormalize(para, rdata, Zrenorm)
    idata_R = renormalize(para, idata, Zrenorm)

    return rdata_R, idata_R, kgrid, ngrid
end

# function renormalize(para, sigma, Zrenorm)
#     dim, β, kF = para.dim, para.β, para.kF

#     mu, sw = CounterTerm.getSigma(para; parafile=parafilename)
#     δzi, δμ, δz = CounterTerm.sigmaCT(para.order, mu, sw)
#     println("δzi:\n", δzi, "\nδμ:\n", δμ, "\nδz:\n", δz, "\n")

#     # sigma_R = sigma * z if zrenorm is True, else sigma_R = sigma
#     sigma_R = CounterTerm.renormalization(para.order, sigma, δμ, δz; zrenorm=Zrenorm)
#     return sigma_R
# end

# function process(para, Zrenorm)
#     dim, β, kF = para.dim, para.β, para.kF

#     ngrid, kgrid, rdata, idata = loaddata(para, filename)

#     rdata_R = renormalize(para, rdata, Zrenorm)
#     idata_R = renormalize(para, idata, Zrenorm)

#     return rdata_R, idata_R, kgrid, ngrid
# end

#  • δzi:
#     (N = 1): 0.0 ± 8.1e-11
#     (N = 2): 0.044576 ± 1.5e-5
#     (N = 3): 0.022092 ± 3.1e-5
#     (N = 4): 0.013545 ± 9.9e-5

#  • δμ:
#     (N = 1): -0.50463 ± 0.00018
#     (N = 2): -0.2588 ± 0.00014
#     (N = 3): -0.12428 ± 0.00012
#     (N = 4): -0.07251 ± 0.00027

#  • δz:
#     (N = 1): 0.0 ± 8.1e-11
#     (N = 2): -0.044576 ± 1.5e-5
#     (N = 3): -0.022092 ± 3.1e-5
#     (N = 4): -0.011558 ± 9.9e-5

function sk(sigma, order, kgrid, para)
    dk = [
        (sigma[o][1, 2:end] .- sigma[o][1, 1]) ./ (kgrid[2:end] .^ 2 / (2 * para.me))
        for o in 1:order
    ]
    return kgrid[2:end], sum(dk)
end

function delta_z(sigma, order, ki, para)
    return [(sigma[o][2, ki] .- sigma[o][1, ki]) ./ (2π / para.β) for o in 1:order]
    # return [(sigma[o][1, ki]) ./ (π / para.β) for o in 1:order]
end

# power series of z-factor
function sw(sigma, order, kgrid, para)
    dw = [sum(delta_z(sigma, order, ki, para)) for ki in 1:length(kgrid)]
    # dw = [(sigma[o][1, :]) ./ (π / para.β) for o in 1:order]
    return kgrid, dw
end

# power series of 1/z
function sw_inv(sigma, order, kgrid, para)
    dw = []
    for ki in 1:length(kgrid)
        δzi = delta_z(sigma, order, ki, para)
        zi = Taylor1([1.0, δzi...], order)
        z = 1 / zi
        δz = [getcoeff(z, o) for o in 1:order]
        push!(dw, sum(δz[1:end]))
    end
    return kgrid, dw
end

function plotS_k(para, rSw_k, iSw_k, kgrid; Zrenorm=true)
    # # Get order-dependent params and ReΣ / ImΣ
    # paralist = para isa Vector ? para : repeat([para], para.order)
    # rSw_ks = rSw_k isa Vector ? rSw_k : repeat([rSw_k], para.order)
    # iSw_ks = iSw_k isa Vector ? iSw_k : repeat([iSw_k], para.order)

    # para = paralist[1]  # All params except lambda are the same for each order
    # dim, β, kF = para.dim, para.β, para.kF
    # kF_label = searchsortedfirst(kgrid, kF)
    # # zk[1] = zk[1] .- zk[1][kF_label]
    # # zk[2] = zk[2] .- zk[2][kF_label]

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

    kF = para.kF

    # Plot ∂_ϵₖ Σ(k, iω0) ≈ (Σ(k, iω0) - Σ(0, iω0)) / ϵ_k,
    # multiplying by (1/z)[ξ] if Zrenorm is true
    subplot(1, 2, 2)
    # subplot(1, 3, 3)
    for o in 1:(para.order)
        _kgrid, s_k = sk(rSw_k, o, kgrid, para)
        z = s_k
        y = [-z.val for z in z]
        e = [z.err for z in z]
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
        ylim([-0.02, 0.2])
    elseif para.rs == 2.0
        ylim([-0.14, 0.225])
    end
    xlabel("\$k/k_F\$")
    if Zrenorm
        ylabel(
            "\$z \\cdot \\left(\\Sigma(k, i\\omega_0) - \\Sigma(0, i\\omega_0)\\right)/(k^2/2m)\$",
        )
    else
        ylabel(
            "\$\\left(\\Sigma(k, i\\omega_0) - \\Sigma(0, i\\omega_0)\\right)/(k^2/2m)\$",
        )
    end
    legend()
    PyPlot.tight_layout()

    # # Plot Z(k) = 1 - ∂_iω Σ(k, iω)|_{ω=0},
    # # multiplying by (1/z)[ξ] if Zrenorm is true
    # subplot(1, 3, 2)
    # for o in 1:(para.order)
    #     _kgrid, s_w = sw_inv(iSw_k, o, kgrid, para)
    #     z = s_w
    #     # z = [(1.0 + e) for e in s_w]
    #     # _kgrid, s_w = sw(iSw_k, o, kgrid)
    #     # z = [1.0 / (1.0 + e) for e in s_w]
    #     y = [z.val for z in z]
    #     e = [z.err for z in z]
    #     kidx = searchsortedfirst(_kgrid, kF)

    #     println("order $o at k=$(_kgrid[1]/kF): $(z[1])")
    #     println("order $o at k=$(_kgrid[kidx]/kF): $(z[kidx])")

    #     errorbar(
    #         _kgrid / kF,
    #         y;
    #         yerr=e,
    #         color=color[o],
    #         capsize=4,
    #         fmt="o",
    #         markerfacecolor="none",
    #         label="Order $o",
    #     )

    #     _x, _y = spline(_kgrid / kF, y, e)
    #     plot(_x, _y; color=color[o], linestyle="--")
    # end
    # xlim([kgrid[1] / kF, 2.0])
    # # ylim([0.98, 1.275])
    # if para.rs == 1.0
    #     ylim([0.73, 1.02])
    # elseif para.rs == 2.0
    #     ylim([0.78, 1.02])
    # end
    # xlabel("\$k/k_F\$")
    # if Zrenorm
    #     ylabel(
    #         "\$\\frac{1}{z} \\cdot Z(k) = \\frac{1}{z} \\cdot \\left( 1 - \\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\\right)^{-1}\$",
    #     )
    # else
    #     ylabel(
    #         "\$Z(k) = \\left( 1 - \\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\\right)^{-1}\$",
    #     )
    # end
    # legend()
    # PyPlot.tight_layout()
    # println()

    # Plot (1/Z)(k) = (1 - ∂_iω Σ(k, iω)|_{ω=0})^{-1},
    # multiplying by z[ξ] if Zrenorm is true
    subplot(1, 2, 1)
    # subplot(1, 3, 1)

    mu2, sw2 = UEG_MC.getSigma(para; parafile=parafilename)
    δzi, δμ, δz = CounterTerm.sigmaCT(para.order, mu2, sw2)

    for o in 1:(para.order)
        _kgrid, s_w = sw(iSw_k, o, kgrid, para)
        # z = s_w
        z = [e - sum(δzi[i] for i in 1:o) for e in s_w]
        # z = [(1.0 + e) for e in s_w]
        # z = [(1.0 - e) for e in s_w]

        # _kgrid, s_w = sw(iSw_k, o, kgrid)
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
        if Zrenorm
            ylim([-0.08, 0.015])
        else
            ylim([-0.01, 0.1])
        end
    elseif para.rs == 2.0
        ylim([0.98, 1.18])
    end
    # if para.rs == 1.0
    #     ylim([0.98, 1.275])
    # elseif para.rs == 2.0
    #     ylim([0.98, 1.18])
    # end
    if Zrenorm
        ylabel(
            "\$-(z - 1) \\cdot \\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\$",
        )
    else
        ylabel(
            "\$-\\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\$",
        )
    end
    # if Zrenorm
    #     ylabel(
    #         "\$z \\cdot Z^{-1}(k) = z \\cdot \\left(1 - \\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\\right)\$",
    #     )
    # else
    #     ylabel(
    #         "\$Z^{-1}(k) = 1 - \\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\$",
    #     )
    # end
    xlabel("\$k/k_F\$")
    legend()
    PyPlot.tight_layout()

    zrenormstr = Zrenorm ? "_zrenorm" : ""
    savefig(
        "../../results/sigma_k_iw0/sigma_k_iw0_rs=$(para.rs)_lambda=$(para.mass2)_beta=$(para.beta)$(zrenormstr).pdf",
        # "../../results/effective_mass_ratio/rs=1/ngrid_test/sigma_k_iw0_rs=$(para.rs)_lambda=$(para.mass2)_beta=$(para.beta)$(zrenormstr)_m10.pdf",
        # "../../results/effective_mass_ratio/rs=1/ngrid_test/sigma_k_iw0_rs=$(para.rs)_lambda=$(para.mass2)_beta=$(para.beta)$(zrenormstr)_0p1.pdf",
    )
    return

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
    # para = ParaMC(; rs=1.0, beta=40.0, Fs=-0.0, order=4, mass2=1.0, isDynamic=false)
    para = ParaMC(; rs=1.0, beta=40.0, Fs=-0.0, order=4, mass2=1.75, isDynamic=false)

    # para = ParaMC(; rs=2.0, beta=40.0, Fs=-0.0, order=4, mass2=1.75, isDynamic=false)

    # Using single lambda for all orders at rs=1,2
    rSw_k, iSw_k, kgrid, ngrid = process(para, Zrenorm)
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