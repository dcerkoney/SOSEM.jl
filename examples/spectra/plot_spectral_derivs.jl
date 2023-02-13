using ElectronGas
using ElectronLiquid
using ElectronLiquid.Propagator: green2, green3
using Lehmann
using PyCall

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt

function main()
    # UEG parameters
    rs = 1.0
    beta = 40.0
    mass2 = 1.0
    param = ParaMC(; rs=rs, beta=beta, mass2=mass2, isDynamic=false)

    # Use LaTex fonts for plots
    plt.rc("text"; usetex=true)
    plt.rc("font"; family="serif")

    taus = LinRange(0.0, param.β, 500)

    # Plot the occupation number for each partition
    fig, ax = plt.subplots()
    # ξₖ ≡ ϵₖ - μ₀
    # xi_k = 0.0
    xi_k = 1.0
    # First order
    ax.plot(taus / param.β, green2.(xi_k, taus, param.β); label="\$n=1\$ (green2)")
    ax.plot(
        taus / param.β,
        -Spectral.kernelFermiT_dω.(taus, xi_k, param.β);
        label="\$n=1\$ (\$-\$kernelFermiT\$\\_\\mathrm{d}\\omega\$)",
        linestyle="--",
    )
    # Second order
    ax.plot(taus / param.β, green3.(xi_k, taus, param.β); label="\$n=2\$ (green3)")
    ax.plot(
        taus / param.β,
        Spectral.kernelFermiT_dω2.(taus, xi_k, param.β) / 2;
        label="\$n=2\$ (kernelFermiT\$\\_\\mathrm{d}\\omega 2 \\,/\\, 2\$)",
        linestyle="--",
    )
    ax.legend(; loc="best")
    # ax.set_xlim(0.8, 1.2)
    # ax.set_ylim(nothing, 2)
    ax.set_xlabel("\$\\tau / \\beta\$")
    ax.set_ylabel("\$\\partial^{n}_\\mu g_\\mu(k, \\tau) \$")
    # xloc = 0.3
    # yloc = -3
    xloc = 0.5
    yloc = 0.2
    ax.text(
        xloc,
        yloc,
        # "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\\, \\xi_k = 0\$";
        "\$r_s = 1,\\, \\beta \\hspace{0.1em} \\epsilon_F = $(beta),\\, \\xi_k = 1\$";
        fontsize=14,
    )
    fig.tight_layout()
    # fig.savefig("spectral_deriv_comparison_xi_k=0.pdf")
    fig.savefig("spectral_deriv_comparison_xi_k=1.pdf")

    plt.close("all")
    return
end

main()
