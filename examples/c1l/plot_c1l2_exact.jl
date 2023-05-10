using PyCall

# For saving/loading numpy data
@pyimport numpy as np
@pyimport matplotlib.pyplot as plt
@pyimport mpl_toolkits.axes_grid1.inset_locator as il

function c1l2_over_eTF2_vlambda_vlambda(l)
    m = sqrt(l)
    I1 = (l / (l + 4) + log((l + 4) / l) - 1) / 4
    I2 = (l^2 / (l + 4) - (l + 4) + 2l * log((l + 4) / l)) / 48
    I3 = (π / 2m + 2 / (l + 4) - atan(2 / m) / m) / 3
    # I1 = (l / (l + 4) - log(l / (l + 4)) - 1) / 4
    # I2 = (l^2 / (l + 4) - (l + 4) - 2l * log(l / (l + 4))) / 64
    # I3 = 2(2 / (l + 4) - atan(2 / m) / m) / 3
    return (I1 + I2 + I3)
end

function c1l2_over_eTF2_v_vlambda(l)
    m = sqrt(l)
    return (π / 3m - 1 / 12) + (l / 12 + 1) * log((4 + l) / l) / 4 - (2 / 3m) * atan(2 / m)
end

# Use LaTex fonts for plots
plt.rc("text"; usetex=true)
plt.rc("font"; family="serif")

ls = LinRange(log(1e-5), log(2.0), 500)
lls = exp.(ls)

fig, ax = plt.subplots()
# ax.axhline(0.0; linestyle="--", linewidth=1, color="k")
ax.plot(lls, c1l2_over_eTF2_v_vlambda.(lls); label="\$V V_\\lambda\$")
ax.plot(lls, c1l2_over_eTF2_vlambda_vlambda.(lls); label="\$V_\\lambda V_\\lambda\$")
ax.set_xlim(-0.025, 1.0)
ax.set_ylim(0.0, 3.0)
ax.legend(loc="best")
ax.set_xlabel("\$\\lambda / k^2_F\$")
ax.set_ylabel("\$C^{(1)l}_2(\\lambda) \\,/\\, \\epsilon^2_{\\mathrm{TF}}\$")
plt.tight_layout()
fig.savefig("results/c1l/c1l2_exact_comparison.pdf")
plt.close("all")