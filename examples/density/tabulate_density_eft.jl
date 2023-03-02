using ElectronLiquid
using DataStructures
using JLD2

# Load the density data
loadparam = ParaMC(; order=3, rs=1.0, beta=40.0, mass2=1.0, isDynamic=false, isFock=false)
param, data = jldopen("data_n.jld2", "a+") do f
    key = "$(UEG.short(loadparam))"
    return f[key]
end
data = sort(data)

# Add overall spin summation (factor of 2) and display
# partitions in units of the non-interacting density n0
n0 = param.kF^3 / (3 * pi^2)
for (k, v) in data
    data[k] = 2 * v / n0
end
println("Partitions (n_loop + 1, n_μ, n_λ):")
for P in keys(data)
    P[3] > 0 && continue  # hide lambda counterterms if present
    println("$((P[1] + 1, P[3], P[2])):\t$(data[P])")
end