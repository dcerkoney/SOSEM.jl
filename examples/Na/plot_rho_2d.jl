using DelimitedFiles
using PyCall
using PyPlot

# For style "science"
@pyimport scienceplots

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

const Ecut = 75

function main()
    # Setup plot styles
    style = PyPlot.matplotlib."style"
    style.use(["science", "std-colors"])
    color = [cdict["orange"], cdict["blue"], cdict["cyan"], cdict["magenta"], cdict["red"]]

    # color = [cdict["blue"], cdict["orange"], "green", cdict["red"], "black"]
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")

    # Use LaTex fonts for plots
    rcParams["font.size"] = 16
    rcParams["mathtext.fontset"] = "cm"
    # rcParams["font.family"] = "Times New Roman"

    fig = figure(; figsize=(6, 4))

    data = readdlm("results/Na/Na_rho_x-0-0_Ecut=$Ecut.dat")
    nrow = size(data)[1]
    xs = data[1, :][1:(end - 1)]

    println(data)
    println(nrow)
    println(xs)

    axhline(1.0; linestyle="--", color=cdict["grey"])
    yvals = [0.0, 0.25, 0.5]
    # yvals = [0.0, 0.125, 0.25, 0.375, 0.5]
    ic = 1
    for i in 2:nrow
        yval = data[i, 1]
        if yval ∉ yvals
            continue
        end
        fs = data[i, :][2:end]

        println(yval)
        println(fs)
        plot(xs, fs; color=color[ic], label="\$y / a = $yval\$")
        ic += 1
    end
    # xticks!([0.0, 0.25, 0.5, 0.75, 1.0])
    # yticks!([0.0, 0.25, 0.5, 0.75, 1.0])
    # tight_layout()
    legend(; loc="best")
    xlabel("\$x / a\$")
    ylabel("\$\\rho(x, y, 0) / \\overline{\\rho}\$")
    return savefig("results/Na/Na_rho_x-0-0_Ecut=$Ecut.pdf")
end

main()