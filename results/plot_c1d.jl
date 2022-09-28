using Plots
using PyCall

# Load the vegas result
sosem_vegas = np.load(
    "results/c1d_n=$(settings.n_order)_rs=$(params.rs)_" *
    "beta_ef=$(params.beta)_neval=$(maxeval)_v2.npz"
)
kgrid, means, stdev = sosem

# Compare with quadrature
sosem_exact = np.load("results/soms_rs=2.0_beta_ef=40.0.npz")
kgrid_exact = np.linspace(0.0, 6.0; num=600)
# c1d_exact_dimless = sosem_exact.get("bare_d") / (params.EF / 2.0)
c1d_exact_dimless = sosem_exact.get("bare_d") / params.EF

# Plot the result
fig, ax = plt.subplots()
ax.plot(
    kgrid_exact,
    c1d_exact_dimless,
    "k";
    label=raw"$\widetilde{C}^{(1)d}(\mathbf{k})$ (quad)",
)
# ax.plot(k_kf_grid, means, "o-"; label=raw"$C^{(1)d}(\mathbf{k}) / \epsilon_F$")
# ax.plot(k_kf_grid, means / 2, "o-"; color="C0", label=raw"$C^{(1)d}(\mathbf{k}) / (2\epsilon_F)$ (vegas)")
# ax.fill_between(k_kf_grid, (means - stdevs) / 2, (means + stdevs) / 2; color="C0", alpha=0.4)
ax.plot(
    k_kf_grid,
    means,
    "o-";
    color="C0",
    label=raw"$C^{(1)d}(\mathbf{k}) / \epsilon_F$ (vegas)",
)
ax.fill_between(k_kf_grid, means - stdevs, means + stdevs; color="C0", alpha=0.4)
ax.legend(; loc="best")
ax.set_xlabel(raw"$k / k_F$")
ax.set_xlim(minimum(k_kf_grid), maximum(k_kf_grid))
plt.tight_layout()
fig.savefig(
    "results/c1d_n=$(settings.n_order)_rs=$(params.rs)_" *
    "beta_ef=$(params.beta)_neval=$(maxeval)_dimless_v2.pdf",
)
plt.close("all")