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

# Physical parameters
const rs = 3.0
# const rs = 4.0
const Fs = -0.0
const beta = 40.0
const max_order_plot = 5

# const Zrenorm = false     # turn off Z renormalization 
const Zrenorm = true        # turn on Z renormalization 

const filename = "data/data_K.jld2"
const parafilename = "data/para.csv"

# Calculated lambda optima and associated max run orders for each calculation
const lambda_opt = Dict(
    1.0 => [2.0, 2.0, 2.0, 2.0, 2.0],
    2.0 => [1.75, 1.75, 1.75, 1.75, 1.75],
    3.0 => [1.75, 1.75, 1.75, 1.75, 1.75],
    # 3.0 => [0.75, 0.75, 1.0, 1.25, 1.75],
    4.0 => [0.625, 0.625, 0.75, 1.0, 1.125],
    #
)
const max_orders = Dict(
    1.0 => [5, 5, 5, 5, 5],
    2.0 => [5, 5, 5, 5, 5],
    3.0 => [5, 5, 5, 5, 5],
    # 3.0 => [2, 2, 3, 4, 5],
    4.0 => [2, 2, 3, 4, 5],
    #
)

# Vibrant qualitative colour scheme from https://personal.sron.nl/~pault/
const cdict = Dict([
    "orange" => "#EE7733",
    "blue" => "#0077BB",
    "cyan" => "#33BBEE",
    "magenta" => "#EE3377",
    "red" => "#CC3311",
    "teal" => "#009988",
    "grey" => "#BBBBBB",
]);

function spline_k(x, y, e)
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

function spline_orders(x, y, e)
    # signal = pyimport("scipy.signal")
    interp = pyimport("scipy.interpolate")
    # yfit = signal.savgol_filter(y, 5, 3)
    w = 1.0 ./ e
    # generate knots with spline without constraints
    spl = interp.UnivariateSpline(x, y; w=w, k=3)
    __x = collect(LinRange(x[1], x[end], 100))
    yfit = spl(__x)
    return __x, yfit
end

function loaddata(para, FileName=filename)
    key = UEG.short(para)
    f = jldopen(FileName, "r")
    ngrid, kgrid, sigma = f[key]
    rdata, idata = Dict(), Dict()
    for p in UEG.partition(para.order)
        rdata[p] = real(sigma[p][:, :])
        idata[p] = imag(sigma[p][:, :])
    end
    return ngrid, kgrid, rdata, idata
end

function renormalize(para, sigma, Zrenorm)
    zpara = deepcopy(para)
    zpara.order = 5
    mu, sw = UEG_MC.getSigma(zpara; parafile=parafilename)
    δzi, δμ, δz = CounterTerm.sigmaCT(para.order, mu, sw)
    println("Max order: $(para.order)")
    println("δzi:\n", δzi, "\nδμ:\n", δμ, "\nδz:\n", δz, "\n")
    # sigma_R = sigma * z if zrenorm is True, else sigma_R = sigma
    return CounterTerm.renormalization(para.order, sigma, δμ, δz; zrenorm=Zrenorm)
end

function process(para, Zrenorm)
    ngrid, kgrid, rdata, idata = loaddata(para, filename)
    rdata_R = renormalize(para, rdata, Zrenorm)
    idata_R = renormalize(para, idata, Zrenorm)
    return rdata_R, idata_R, kgrid, ngrid
end

function ds_dk(sigma, order, kgrid)
    # derivative ds/dk from non-uniform first-order central difference method
    dsdk = [
        (sigma[o][1, 3:end] - sigma[o][1, 1:(end - 1)]) /
        (kgrid[3:end] - kgrid[1:(end - 1)]) for o in 1:order
    ]
    return kgrid[2:end], sum(dsdk)
end

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

function plotS_k(paralist::Vector{ParaMC}, rSw_ks, iSw_ks, kgrid; Zrenorm=true)
    # max_order_plot = maximum([p.order for p in paralist])
    @assert length(paralist) == length(rSw_ks) == length(iSw_ks) == max_order_plot

    para = paralist[end]  # Physical parameters at maximum plot order
    lambdas = [p.mass2 for p in paralist]
    kF = para.kF

    # plot = pyimport("plot")
    style = PyPlot.matplotlib."style"
    style.use(["science", "std-colors"])
    color = [
        "k",
        cdict["orange"],
        cdict["blue"],
        cdict["cyan"],
        cdict["magenta"],
        cdict["red"],
        cdict["teal"],
    ]
    # color = [cdict["blue"], cdict["orange"], "green", cdict["red"], "black"]
    # color = ["green", cdict["blue"], cdict["red"], "black"]
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
    subplot(1, 2, 2)
    # subplot(1, 3, 3)
    for (o, p) in enumerate(paralist)
        zpara = deepcopy(p)
        zpara.order = 5
        mu_data, sw_data = UEG_MC.getSigma(zpara; parafile=parafilename)
        δzi, δμ, δz = CounterTerm.sigmaCT(p.order, mu_data, sw_data)

        _kgrid, s_k = sk(rSw_ks[o], o, kgrid, p)
        s_k = -s_k  # we measure -Σ
        # z = s_k
        if Zrenorm
            zF = 1 + sum(δz[i] for i in 1:o)
            z = [(zF + e) for e in s_k]
        else
            z = [(1 + e) for e in s_k]
        end
        # y = [-z.val for z in z]
        y = [z.val for z in z]
        e = [z.err for z in z]
        errorbar(
            _kgrid / kF,
            y;
            yerr=e,
            color=color[o],
            capsize=4,
            # markersize=4,
            fmt="o",
            markerfacecolor="none",
            label="\$N = $o\$",
            zorder=10 * o + 3,
        )

        _x, _y = spline_k(_kgrid / kF, y, e)
        plot(_x, _y; color=color[o], linestyle="--")
    end
    xlim([kgrid[1] / kF, 2.0])
    if para.rs == 1.0
        # ylim([nothing, 0.28])
        # ylim([-0.07, 0.23])
        if Zrenorm
            ylim([nothing, 1.2])
        else
            ylim([nothing, 1.35])
        end
    elseif para.rs == 2.0
        # ylim([-0.07, 0.25])
        # ylim([nothing, 0.3])
        if Zrenorm
            ylim([nothing, 1.2])
        else
            ylim([nothing, 1.35])
        end
    elseif para.rs == 3.0
        if Zrenorm
            ylim([nothing, 1.15])
        else
            ylim([0.98, 1.32])
        end
    elseif para.rs == 4.0
        # ylim([-0.07, 0.25])
        # ylim([nothing, 0.3])
        if Zrenorm
            ylim([nothing, 1.15])
        else
            ylim([0.98, 1.32])
        end
    end
    xlabel("\$k/k_F\$")
    if Zrenorm
        ylabel(
            "\$z \\cdot \\epsilon(k)/(k^2/2m)\$",
            # "\$z \\cdot \\left(\\Sigma(k, i\\omega_0) - \\Sigma(0, i\\omega_0)\\right)/(k^2/2m)\$",
        )
    else
        ylabel(
            "\$\\epsilon(k)/(k^2/2m)\$",
            # "\$\\left(\\Sigma(k, i\\omega_0) - \\Sigma(0, i\\omega_0)\\right)/(k^2/2m)\$",
        )
    end
    legend(; ncol=2, loc="upper right")
    PyPlot.tight_layout()

    # Plot (1/Z)(k) = (1 - ∂_iω Σ(k, iω)|_{ω=0})^{-1},
    # multiplying by z[ξ] if Zrenorm is true
    subplot(1, 2, 1)
    # subplot(1, 3, 1)

    for (o, p) in enumerate(paralist)
        zpara = deepcopy(p)
        zpara.order = 5
        mu_data, sw_data = UEG_MC.getSigma(zpara; parafile=parafilename)
        δzi, δμ, δz = CounterTerm.sigmaCT(p.order, mu_data, sw_data)

        _kgrid, s_w = sw(iSw_ks[o], o, kgrid, p)

        # a = zF or 1 depending on whether Zrenorm is true
        local a
        if Zrenorm
            a = 1.0 + sum(δz[i] for i in 1:o)
            @assert a ≤ 1.0
        else
            a = 1.0
        end
        z = [a + e for e in s_w]
        y = [z.val for z in z]
        e = [z.err for z in z]

        # if Zrenorm
        #     z = [e - sum(δzi[i] for i in 1:o) for e in s_w]
        # else
        #     z = s_w
        # end
        # y = [-z.val for z in z]
        # e = [z.err for z in z]

        kidx = searchsortedfirst(_kgrid, kF)
        println("order $o at k=$(_kgrid[1]/kF): $(z[1])")
        println("order $o at k=$(_kgrid[kidx]/kF): $(z[kidx])")

        errorbar(
            _kgrid / kF,
            y;
            yerr=e,
            color=color[o],
            capsize=4,
            # markersize=4,
            fmt="o",
            markerfacecolor="none",
            label="\$N = $o\$",
            zorder=10 * o + 3,
        )

        _x, _y = spline_k(_kgrid / kF, y, e)
        plot(_x, _y; color=color[o], linestyle="--")
    end
    xlim([kgrid[1] / kF, 2.0])
    if para.rs == 1.0
        if Zrenorm
            ylim([0.9175, 1.1])
        else
            ylim([0.99, 1.21])
            # ylim([0.79, 1.01])
        end
    elseif para.rs == 2.0
        if Zrenorm
            ylim([0.9175, 1.1])
        else
            ylim([0.99, 1.21])
            # ylim([0.79, 1.01])
        end
    elseif para.rs == 3.0
        if Zrenorm
            ylim([0.9375, 1.075])
        else
            ylim([0.99, 1.16])
            # ylim([0.79, 1.01])
        end
    elseif para.rs == 4.0
        if Zrenorm
            ylim([0.9175, 1.1])
        else
            ylim([0.99, 1.21])
            # ylim([0.79, 1.01])
        end
    end
    # if para.rs == 1.0
    #     if Zrenorm
    #         ylim([-0.01, 0.09])
    #     else
    #         ylim([-0.17, 0.02])
    #     end
    # elseif para.rs == 2.0
    #     if Zrenorm
    #         ylim([-0.015, 0.09])
    #         # ylim([-0.015, 0.06])
    #     else
    #         ylim([-0.22, 0.02])
    #         # ylim([-0.17, 0.02])
    #     end
    # elseif para.rs == 4.0
    #     if Zrenorm
    #         # ylim([-0.015, 0.09])
    #         # ylim([-0.015, 0.06])
    #     else
    #         ylim([-0.24, 0.02])
    #         # ylim([-0.17, 0.02])
    #     end
    # end
    if Zrenorm
        ylabel(
            "\$z \\cdot Z^{-1}(k)\$",
            # "\$(z - 1) \\cdot \\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\$",
        )
    else
        ylabel(
            "\$Z^{-1}(k)\$",
            # "\$\\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\$",
        )
    end
    xlabel("\$k/k_F\$")
    if Zrenorm
        legend(; ncol=2, loc="upper left")
    else
        legend(; ncol=2, loc="best")
    end
    PyPlot.tight_layout()

    zrenormstr = Zrenorm ? "_zrenorm" : ""
    lambdastr = allequal(lambdas) ? "$(lambdas[1])" : "$(lambdas)"
    savefig(
        "../../results/sigma_k_iw0/sigma_k_iw0_rs=$(para.rs)_lambdas=$(lambdastr)_beta=$(para.beta)$(zrenormstr).pdf",
    )
    return
end

function plotS_k(para::ParaMC, rSw_k, iSw_k, kgrid; Zrenorm=true)
    duplicate(v) = repeat([v], para.order)
    plotS_k(duplicate(para), duplicate(rSw_k), duplicate(iSw_k), kgrid; Zrenorm=Zrenorm)
    return
end

function plotS_k0(paralist; Zrenorm=true, ktarget=0.5)
    lambdas = [para.mass2 for para in paralist]
    para = paralist[1]
    local kval
    s_ks = []
    s_ws = []
    for para in paralist
        # Using single lambda for all orders at rs=1,2
        rSw_k, iSw_k, kgrid, ngrid = process(para, Zrenorm)
        kF_label = searchsortedfirst(kgrid, para.kF)
        for k in keys(rSw_k)
            println(k, ": ", rSw_k[k][1, kF_label])
        end
        s_k_Ns = []
        s_w_Ns = []
        for o in 1:(para.order)
            _kgrid1, s_k = sk(rSw_k, o, kgrid, para)
            ikval1 = searchsortedfirst(_kgrid1, para.kF * ktarget)
            kval = _kgrid1[ikval1] / para.kF

            _kgrid2, s_w = sw(iSw_k, o, kgrid, para)
            ikval2 = searchsortedfirst(_kgrid2, para.kF * ktarget)
            kval2 = _kgrid2[ikval2] / para.kF
            @assert kval == kval2

            push!(s_k_Ns, s_k[ikval1])
            push!(s_w_Ns, s_w[ikval2])
        end
        push!(s_ks, s_k_Ns)
        push!(s_ws, s_w_Ns)
        println("lambda = $(para.mass2), ktarget = $(ktarget)kF, kval = $(kval)kF")
    end

    # plot = pyimport("plot")
    style = PyPlot.matplotlib."style"
    style.use(["science", "std-colors"])
    color = [
        "k",
        cdict["orange"],
        cdict["blue"],
        cdict["cyan"],
        cdict["magenta"],
        cdict["red"],
        cdict["teal"],
    ]
    # color = [cdict["blue"], cdict["orange"], "green", cdict["red"], "black"]
    # color = ["green", cdict["blue"], cdict["red"], "black"]
    #cmap = get_cmap("Paired")
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    # Use LaTex fonts for plots
    rcParams["font.size"] = 16
    rcParams["mathtext.fontset"] = "cm"
    # rcParams["font.family"] = "Times New Roman"

    figure(; figsize=(12, 4))

    # Plot ∂_ϵₖ Σ(k, iω0) ≈ (Σ(k, iω0) - Σ(0, iω0)) / ϵ_k,
    # multiplying by (1/z)[ξ] if Zrenorm is true
    sp1 = subplot(1, 2, 2)
    for (i, p) in enumerate(paralist)
        lambda = p.mass2
        orders = collect(1:(p.order))
        mu_data, sw_data = UEG_MC.getSigma(p; parafile=parafilename)
        _, _, δz = CounterTerm.sigmaCT(p.order, mu_data, sw_data)
        if Zrenorm
            zF = [1 + sum(δz[i] for i in 1:o) for o in orders]
            z = [(zF[o] - e) for (o, e) in enumerate(s_ks[i])]
        else
            z = [(1 - e) for e in s_ks[i]]
        end
        # y = [-z.val for z in z]
        y = [z.val for z in z]
        e = [z.err for z in z]

        # z = s_ks[i]
        # y = [-z.val for z in z]
        # e = [z.err for z in z]
        errorbar(
            orders,
            y;
            yerr=e,
            color=color[i + 1],
            capsize=4,
            # markersize=4,
            fmt="o",
            markerfacecolor="none",
            label="\$\\lambda = $lambda\$",
            zorder=10 * i + 3,
        )
        _x, _y = spline_orders(orders, y, e)
        plot(_x, _y; color=color[i + 1], linestyle="--")
    end
    # xlim([kgrid[1] / kF, 2.0])
    if para.rs == 1.0
        # ylim([nothing, 0.28])
        # ylim([-0.07, 0.23])
    elseif para.rs == 2.0
        # ylim([-0.07, 0.25])
    end
    xlabel("\$N\$")
    annotate(
        "\$k = $(round(kval; sigdigits=3)) k_F\$";
        xy=[1; 0],
        xycoords="axes fraction",
        xytext=[0.875, 0.34],
        textcoords="axes fraction",
        # textcoords="offset points",
        fontsize=14.0,
        ha="right",
        va="bottom",
    )
    if Zrenorm
        ylabel(
            "\$z \\cdot \\epsilon(k) / (k^2/2m)\$",
            # "\$z \\cdot \\left(\\Sigma(k, i\\omega_0) - \\Sigma(0, i\\omega_0)\\right)/(k^2/2m)\$",
        )
    else
        ylabel(
            "\$\\epsilon(k) / (k^2/2m)\$",
            # "\$\\left(\\Sigma(k, i\\omega_0) - \\Sigma(0, i\\omega_0)\\right)/(k^2/2m)\$",
        )
    end
    legend(; ncol=2, loc="best")
    PyPlot.tight_layout()

    # Plot (1/Z)(k) = (1 - ∂_iω Σ(k, iω)|_{ω=0})^{-1},
    # multiplying by z[ξ] if Zrenorm is true
    sp2 = subplot(1, 2, 1)
    for (i, p) in enumerate(paralist)
        lambda = p.mass2
        orders = collect(1:(p.order))
        mu_data, sw_data = UEG_MC.getSigma(p; parafile=parafilename)
        δzi, _, _ = CounterTerm.sigmaCT(p.order, mu_data, sw_data)
        if Zrenorm
            z = [e - sum(δzi[j] for j in 1:o) for (o, e) in enumerate(s_ws[i])]
        else
            z = s_ws[i]
        end
        y = [-z.val for z in z]
        e = [z.err for z in z]
        errorbar(
            orders,
            y;
            yerr=e,
            color=color[i + 1],
            capsize=4,
            # markersize=4,
            fmt="o",
            markerfacecolor="none",
            label="\$\\lambda = $lambda\$",
            zorder=10 * i + 3,
        )
        _x, _y = spline_orders(orders, y, e)
        plot(_x, _y; color=color[i + 1], linestyle="--")
    end
    # xlim([kgrid[1] / kF, 2.0])
    if para.rs == 1.0
        if Zrenorm
            # ylim([-0.01, 0.09])
            # ylim([-0.09, 0.012])
            # ylim([-0.1, 0.012])
        else
            # ylim([-0.17, 0.02])
        end
    elseif para.rs == 2.0
        # ylim([-0.12, 0.01])
        # ylim([0.98, 1.18])
    end
    if Zrenorm
        ylabel(
            "\$(z - 1) \\cdot \\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\$",
            # "\$-(z - 1) \\cdot \\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\$",
        )
    else
        ylabel(
            "\$\\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\$",
            # "\$-\\partial_{i\\omega}\\operatorname{Im}\\Sigma(k, i\\omega)\\big|_{\\omega = 0}\$",
        )
    end
    xlabel("\$N\$")
    if Zrenorm
        annotate(
            "\$k = $(round(kval; sigdigits=3)) k_F\$";
            xy=[1; 0],
            xycoords="axes fraction",
            xytext=[0.125, 0.5],
            textcoords="axes fraction",
            # textcoords="offset points",
            fontsize=14.0,
            ha="left",
            va="top",
        )
    else
        annotate(
            "\$k = $(round(kval; sigdigits=3)) k_F\$";
            xy=[1; 0],
            xycoords="axes fraction",
            xytext=[0.125, 0.25],
            textcoords="axes fraction",
            # textcoords="offset points",
            fontsize=14.0,
            ha="left",
            va="top",
        )
    end
    legend(; ncol=2, loc="best")
    PyPlot.tight_layout()

    zrenormstr = Zrenorm ? "_zrenorm" : ""
    savefig(
        "../../results/sigma_k_iw0/sigma_iw0_k=$(round(kval; sigdigits=3))kF_rs=$(para.rs)_lambda=$(lambdas)_beta=$(para.beta)$(zrenormstr).pdf",
    )

    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    @assert haskey(lambda_opt, rs) "Lambda optima for rs = $(rs) not found!"
    @assert length(lambda_opt[rs]) ≥ max_order_plot "Lambda optimum at N = $max_order_plot not found for rs = $(rs)!"

    # Parameters for each run differ only by order and lambda
    lambdas = lambda_opt[rs][1:max_order_plot]
    orders = max_orders[rs][1:max_order_plot]
    paralist = [
        ParaMC(; rs=rs, beta=beta, Fs=Fs, order=order, mass2=lambda, isDynamic=false)
        for (order, lambda) in zip(orders, lambdas)
    ]

    ### Using order-by-order lambda optima ###
    local kgrid, ngrid
    rSw_ks, iSw_ks = [], []
    for para in paralist
        rSw_k, iSw_k, kgrid, ngrid = process(para, Zrenorm)
        push!(rSw_ks, rSw_k)
        push!(iSw_ks, iSw_k)
    end
    # plotS_k(paralist, rSw_ks, iSw_ks, kgrid; Zrenorm=Zrenorm)
    
    ### Using fixed lambda = λ*₅ for all orders ###
    # rs = 1
    para = ParaMC(; rs=1.0, beta=40.0, Fs=-0.0, order=5, mass2=2.0, isDynamic=false)
    rSw_k, iSw_k, kgrid, ngrid = process(para, Zrenorm)
    plotS_k(para, rSw_k, iSw_k, kgrid; Zrenorm=Zrenorm)
    # rs = 2
    para = ParaMC(; rs=2.0, beta=40.0, Fs=-0.0, order=5, mass2=1.75, isDynamic=false)
    rSw_k, iSw_k, kgrid, ngrid = process(para, Zrenorm)
    plotS_k(para, rSw_k, iSw_k, kgrid; Zrenorm=Zrenorm)
    # rs = 3
    para = ParaMC(; rs=3.0, beta=40.0, Fs=-0.0, order=5, mass2=1.75, isDynamic=false)
    rSw_k, iSw_k, kgrid, ngrid = process(para, Zrenorm)
    plotS_k(para, rSw_k, iSw_k, kgrid; Zrenorm=Zrenorm)
    # rs = 4
    para = ParaMC(; rs=4.0, beta=40.0, Fs=-0.0, order=5, mass2=1.125, isDynamic=false)
    rSw_k, iSw_k, kgrid, ngrid = process(para, Zrenorm)
    plotS_k(para, rSw_k, iSw_k, kgrid; Zrenorm=Zrenorm)

    # # Using single lambda for all orders
    # para = ParaMC(; rs=1.0, beta=40.0, Fs=-0.0, order=5, mass2=2.0, isDynamic=false)
    # para = ParaMC(; rs=1.0, beta=40.0, Fs=-0.0, order=5, mass2=3.5, isDynamic=false)
    # para = ParaMC(; rs=2.0, beta=40.0, Fs=-0.0, order=5, mass2=1.75, isDynamic=false)
    # para = ParaMC(; rs=2.0, beta=40.0, Fs=-0.0, order=5, mass2=2.5, isDynamic=false)
    # para = ParaMC(; rs=4.0, beta=40.0, Fs=-0.0, order=5, mass2=1.5, isDynamic=false)
    # rSw_k, iSw_k, kgrid, ngrid = process(para, Zrenorm)
    # kF_label = searchsortedfirst(kgrid, para.kF)
    # for k in keys(rSw_k)
    #     println(k, ": ", rSw_k[k][1, kF_label])
    # end
    # plotS_k(para, rSw_k, iSw_k, kgrid; Zrenorm=Zrenorm)

    # # Compare results for two calculations using single lambda at all orders
    ### rs = 1 ###
    # mass2list = [2.0, 3.5]
    # paralist = [
    #     ParaMC(; rs=1.0, beta=40.0, Fs=-0.0, order=5, mass2=mass2, isDynamic=false) for
    #     mass2 in mass2list
    # ]
    ### rs = 2 ###
    # mass2list = [1.75, 2.5]
    # paralist = [
    #     ParaMC(; rs=2.0, beta=40.0, Fs=-0.0, order=5, mass2=mass2, isDynamic=false) for
    #     mass2 in mass2list
    # ]
    ### rs = 4 ###
    # mass2list = [1.5]
    # paralist = [
    #     ParaMC(; rs=2.0, beta=40.0, Fs=-0.0, order=5, mass2=mass2, isDynamic=false) for
    #     mass2 in mass2list
    # ]
    # plotS_k0(paralist; Zrenorm=Zrenorm, ktarget=0.5)

end