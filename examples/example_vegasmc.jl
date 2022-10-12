using JLD2
using MCIntegration
using Measurements

N = 20;

grid = [i / N for i in 1:N];

function integrand(vars, config)
    grid = config.userdata # radius
    x, bin = vars #unpack the variables
    r = grid[bin[1]] # binned variable in [0, 1)
    integrand = Vector(undef, config.N)
    for o in 1:(config.N)
        local region
        if o == 1
            region = x[1]^2 + r^2 < 1 # circle
        else
            region = x[1]^2 + x[2]^2 + r^2 < 1 # sphere   
        end
        integrand[o] = region
    end
    return integrand
    # r1 = x[1]^2 + r^2 < 1 # circle
    # r2 = x[1]^2 + x[2]^2 + r^2 < 1 # sphere
    # return r1, r2
end

function measure(vars, obs, weights, config)
    # obs: prototype of the observables for each integral
    x, bin = vars #unpack the variables
    for o in 1:(config.N)
        obs[o][bin[1]] += weights[o]
    end
    return
end;

dof = [[1, 1]]
obs = [zeros(N)]
# dof = [[1, 1], [2, 1]]
# obs = [zeros(N), zeros(N)]
res = integrate(
    integrand;
    measure=measure, # measurement function
    var=(Continuous(0.0, 1.0), Discrete(1, N)), # a continuous and a discrete variable pool 
    dof=dof,
    # integral-1: one continuous and one discrete variables, integral-2: two continous and one discrete variables
    obs=obs, # prototype of the observables for each integral
    userdata=grid,
    neval=1e5,
    solver=:vegasmc,
)

datadict = Dict{Int,Vector{Measurement}}()
for o in 1:length(dof)
    datadict[o] = measurement.(res.mean[o], res.stdev[o])
end

measurements = [measurement.(res.mean[i], res.stdev[i]) for i in 1:length(dof)]
full_meas = sum(measurements)

# Save to JLD2
@save "datadict.jld2" datadict
@save "example_vegasmc.jld2" full_meas

datadict = nothing
full_meas = nothing

# Load from JLD2
@load "datadict.jld2" datadict
@load "example_vegasmc.jld2" full_meas

full_meas_v2 = sum(datadict[o] for o in 1:length(dof))
@assert full_meas == full_meas_v2

means = [m.val for m in full_meas]
stdevs = [m.err for m in full_meas]
