using ASEconvert
using Brillouin: Brillouin
import Brillouin.KPaths: KPath, KPathInterpolant, irrfbz_path
import Contour: contours, levels, level, lines, coordinates
using DFTK
using Interpolations
using LazyArtifacts
using Plots
using Plots.PlotMeasures
using Unitful
using UnitfulAtomic

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

function plot_implicit(
    F,
    c=1;
    xlim=(0.0, 1.0),
    ylim=xlim,
    zlim=xlim,
    nlevels=50,         # number of levels in a direction
    # slices=Dict(:z => cdict["blue"]), # which directions and color
    slices=Dict(:x => cdict["blue"], :z => cdict["blue"]), # which directions and color
    # slices=Dict(:x => cdict["blue"], :y => cdict["cyan"], :z => cdict["teal"]), # which directions and color
    # slices=Dict(:x => :blue, :y => :red, :z => :green), # which directions and color
    kwargs...,          # passed to initial `plot` call
)
    _linspace(rng, n=150) = range(rng[1]; stop=rng[2], length=n)

    X1, Y1, Z1 = _linspace(xlim), _linspace(ylim), _linspace(zlim)

    p = Plots.plot(; legend=false, kwargs...)

    if :x ∈ keys(slices)
        for x in _linspace(xlim, nlevels)
            local X1 = [F(x, y, z) for y in Y1, z in Z1]
            cnt = contours(Y1, Z1, X1, [c])
            for line in lines(levels(cnt)[1])
                ys, zs = coordinates(line) # coordinates of this line segment
                plot!(p, x .+ 0 * ys, ys, zs; color=slices[:x])
            end
        end
    end

    if :y ∈ keys(slices)
        for y in _linspace(ylim, nlevels)
            local Y1 = [F(x, y, z) for x in X1, z in Z1]
            cnt = contours(Z1, X1, Y1, [c])
            for line in lines(levels(cnt)[1])
                xs, zs = coordinates(line) # coordinates of this line segment
                plot!(p, xs, y .+ 0 * xs, zs; color=slices[:y])
            end
        end
    end

    if :z ∈ keys(slices)
        for z in _linspace(zlim, nlevels)
            local Z1 = [F(x, y, z) for x in X1, y in Y1]
            cnt = contours(X1, Y1, Z1, [c])
            for line in lines(levels(cnt)[1])
                xs, ys = coordinates(line) # coordinates of this line segment
                plot!(p, xs, ys, z .+ 0 * xs; color=slices[:z])
            end
        end
    end

    return p
end

function run_lda(
    psp_name;
    temperature=0.01,
    tol=1e-5,
    Ecut=50,
    kgrid=[8, 8, 8],
    plots=false,
)
    # # Manually construct Na crystal structure
    # # (data taken from: https://next-gen.materialsproject.org/materials/mp-127)
    # a = 4.21u"angstrom"  # Sodium lattice constant
    # #! format: off
    # lattice = a / 2 * [   # BCC conventional lattice vectors
    #     [-1  1  1];
    #     [ 1 -1  1];
    #     [ 1  1 -1]
    # ]
    # #! format: on
    # atoms     = [ElementPsp(:Na; psp=load_psp(psp_name))]
    # positions = [zeros(3)]  # Primitive basis: one Na atom at (0, 0, 0)
    # model     = model_LDA(lattice, atoms, positions; temperature=temperature)

    # Automatically construct Na crystal structure using ASEconvert
    system_ase = ase.build.bulk("Na")
    system     = pyconvert(AbstractSystem, system_ase)
    system     = attach_psp(system; Na=psp_name)
    model      = model_LDA(system; temperature=temperature)

    # Run LDA
    basis  = PlaneWaveBasis(model; Ecut=Ecut, kgrid=kgrid)
    scfres = self_consistent_field(basis; tol=tol)
    if plots
        bandplot = plot_bandstructure(scfres)
        dosplot  = plot_dos(scfres)
        return (; scfres, model, basis, bandplot, dosplot)
    end
    return (; scfres, model, basis)
end

function run_lda_comparison(psp_names, labels=string.(eachindex(psp_names)))
    # Run DFTK for each pseudopotential
    lda_results = [run_lda(psp_name; plots=true) for psp_name in psp_names]
    energies = [res.scfres.energies for res in lda_results]
    bandplots = [res.bandplot for res in lda_results]
    dosplots = [res.dosplot for res in lda_results]

    println("Na Energies:\n")
    for (ens, label) in zip(energies, labels)
        println("$label:\n$ens")
    end

    # plot(bandplots...; titles=labels, size=(6, 4))
    plot(bandplots...; titles=labels, size=(800, 400))
    savefig("results/Na/Na_bands_comparison.pdf")

    # plot(dosplots...; titles=labels, size=(6, 4))
    plot(dosplots...; titles=labels, size=(800, 400))
    savefig("results/Na/Na_dos_comparison.pdf")

    return lda_results
end

function plot_rho(scfres, Ecut, rho_avg)

    # Get basis vectors along Cartesian axes x, y, z
    rvecs = collect(r_vectors(scfres.basis))
    xvecs = rvecs[:, 1, 1]  # slice along the x axis
    yvecs = rvecs[1, :, 1]  # slice along the y axis
    zvecs = rvecs[1, 1, :]  # slice along the z axis
    xs = [xvec[1] for xvec in xvecs]
    ys = [yvec[2] for yvec in yvecs]
    zs = [zvec[3] for zvec in zvecs]

    # Build linear interpolator for ρ 
    rho_xyz = scfres.ρ[:, :, :, 1]  # ρ for spin up (wlog)
    interp_rho = linear_interpolation((xs, ys, zs), rho_xyz; extrapolation_bc=Line())

    default(;
        fontfamily="Computer Modern",
        framestyle=:box,
        titlefontsize=16,
        tickfontsize=16,
        labelfontsize=16,
        legendfontsize=16,
    )

    # 3D plot of ρ in the unit cell
    interp_rho_xyz(x, y, z) = interp_rho(x % 1, y % 1, z % 1) / rho_avg
    plot_implicit(
        interp_rho_xyz;
        c=1.0,
        title="\$\\rho(\\mathbf{r}) = \\overline{\\rho}\$",
        size=(600, 400),
    )
    xlabel!("\$x / a\$")
    ylabel!("\$y / a\$")
    zlabel!("\$z / a\$")
    xticks!([0.0, 0.5, 1.0])
    yticks!([0.0, 0.5, 1.0])
    zticks!([0.0, 0.5, 1.0])
    savefig("results/Na/Na_rho_x-y-z_Ecut=$Ecut.pdf")

    push!(xs, 1.0)
    push!(ys, 1.0)
    push!(zs, 1.0)

    # Contour plot of ρ in the x-y plane (z = 0)
    interp_rho_xy0(x, y) = interp_rho(x % 1, y % 1, 0) / rho_avg
    contourf(
        xs,
        ys,
        interp_rho_xy0.(xs', ys);
        levels=10,
        color=:turbo,
        linewidth=0.5,
        title="\$\\rho(x, y, 0) / \\overline{\\rho}\$",
        right_margin=20px,
        size=(600, 400),
    )
    contour!(
        xs,
        ys,
        interp_rho_xy0.(xs', ys);
        levels=[1.0],
        color=cdict["grey"],
        linewidth=2,
        size=(600, 400),
    )
    # xticks!([0.0, 0.25, 0.5, 0.75, 1.0])
    # yticks!([0.0, 0.25, 0.5, 0.75, 1.0])
    xlabel!("\$x / a\$")
    ylabel!("\$y / a\$")
    savefig("results/Na/Na_rho_x-y-0_Ecut=$Ecut.pdf")

    # Contour plot of ρ in the x-y-1/2 plane (z = 1/2)
    interp_rho_xyzhalf(x, y) = interp_rho(x % 1, y % 1, 0.5) / rho_avg
    contourf(
        xs,
        ys,
        interp_rho_xyzhalf.(xs', ys);
        levels=10,
        color=:turbo,
        linewidth=0.5,
        title="\$\\rho(x, y, a / 2) / \\overline{\\rho}\$",
        right_margin=20px,
        size=(600, 400),
    )
    contour!(
        xs,
        ys,
        interp_rho_xyzhalf.(xs', ys);
        levels=[1.0],
        color=cdict["grey"],
        linewidth=2,
        size=(600, 400),
    )
    # xticks!([0.0, 0.25, 0.5, 0.75, 1.0])
    # yticks!([0.0, 0.25, 0.5, 0.75, 1.0])
    xlabel!("\$x / a\$")
    ylabel!("\$y / a\$")
    savefig("results/Na/Na_rho_x-y-0.5_Ecut=$Ecut.pdf")

    return
end

function plot_band_data_custom(
    kpath::KPathInterpolant,
    band_data;
    εF=nothing,
    unit=u"eV",
    kwargs...,
)
    eshift = something(εF, 0.0)
    data = DFTK.data_for_plotting(kpath, band_data)

    # Constant to convert from AU to the desired unit
    to_unit = ustrip(auconvert(unit, 1.0))

    # Plot all bands, spins and errors
    p = Plots.plot(; xlabel="Wave vector \$\\mathbf{k}\$", size=(600, 400))

    # Mark the Fermi energy (xi = 0)
    if !isnothing(εF)
        # Plots.hline!(p, [0.0]; label="εF", color=:green, lw=1.5)
        Plots.hline!(p, [0.0]; color=:gray, lw=1.5, label=nothing)
    end

    margs = length(kpath) < 70 ? (; markersize=2, markershape=:circle) : (;)
    for σ in 1:(data.n_spin), iband in 1:(data.n_bands), branch in data.kbranches
        yerror = nothing
        if hasproperty(data, :λerror)
            yerror = data.λerror[:, iband, σ][branch] .* to_unit
        end
        energies = (data.λ[:, iband, σ][branch] .- eshift) .* to_unit
        Plots.plot!(
            p,
            data.kdistances[branch],
            energies;
            label="",
            yerror,
            color=(cdict["blue"], cdict["red"])[σ],
            lw=1.5,
            # color=(:blue, :red)[σ],
            # color=(:1, :2)[σ],
            margs...,
            kwargs...,
        )
    end

    # Delimiter for branches
    for branch in data.kbranches[1:(end - 1)]
        Plots.vline!(p, [data.kdistances[last(branch)]]; color=:black, label="")
    end

    # X-range: 0 to last kdistance value
    Plots.xlims!(p, (0, data.kdistances[end]))
    Plots.xticks!(p, data.ticks.distances, data.ticks.labels)

    ylims = to_unit .* (DFTK.default_band_εrange(band_data.λ; εF) .- eshift)
    Plots.ylims!(p, round.(ylims, sigdigits=2)...)
    if isnothing(εF)
        Plots.ylabel!(p, "Eigenvalues \$\\epsilon_{\\mathbf{k}n}\$ \$($(string(unit)))\$")
    else
        Plots.ylabel!(p, "Eigenvalues \$\\xi_{\\mathbf{k}n}\$ \$($(string(unit)))\$")
    end

    return p
end

function plot_bandstructure_custom(
    basis::PlaneWaveBasis,
    Ecut,
    kpath::KPath=irrfbz_path(basis.model);
    εF=nothing,
    kline_density=40u"bohr",
    unit=u"eV",
    kwargs_plot=(;),
    kwargs...,
)
    # Use LaTeX fonts for the band plot
    default(;
        fontfamily="Computer Modern",
        framestyle=:box,
        titlefontsize=16,
        tickfontsize=16,
        labelfontsize=16,
        legendfontsize=16,
    )

    # Band structure calculation along high-symmetry path
    kinter = Brillouin.interpolate(kpath; density=austrip(kline_density))
    println("Computing bands along kpath:")
    sortlabels = map(bl -> last.(sort(collect(pairs(bl)))), kinter.labels)
    println("       ", join(join.(sortlabels, " -> "), "  and  "))
    band_data = compute_bands(basis, kinter; kwargs...)
    plot_band_data_custom(kinter, band_data; εF, unit, kwargs_plot...)
    ylims!(-5, 9)
    savefig("results/Na/Na_bands_Ecut=$Ecut.pdf")
    return
end
function plot_bandstructure_custom(
    scfres::NamedTuple,
    Ecut,
    kpath::KPath=irrfbz_path(scfres.basis.model);
    n_bands=DFTK.default_n_bands_bandstructure(scfres),
    kwargs...,
)
    return plot_bandstructure_custom(
        scfres.basis,
        Ecut,
        kpath;
        n_bands,
        scfres.ρ,
        scfres.εF,
        kwargs...,
    )
end

function get_rs(rho)
    return (3 / (4π * rho))^(1 / 3)
end

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    println("Available pseudopotentials for Na:")
    for pseudopotential in list_psp(:Na)
        println(pseudopotential)
    end

    # Ecuts = [75]
    Ecuts = [25, 50, 75]
    kgrid = [9, 9, 9]

    # Plot LDA bands for hardest HGH pseudopotential (n_valence = 9)
    for Ecut in Ecuts
        # lda_results = run_lda("hgh/lda/na-q9.hgh"; Ecut=15, kgrid=[4, 4, 4])
        # lda_results = run_lda("hgh/lda/na-q1.hgh"; Ecut=15, kgrid=[4, 4, 4])

        lda_results = run_lda("hgh/lda/na-q1.hgh"; Ecut=Ecut, kgrid=kgrid)
        # lda_results = run_lda("hgh/lda/na-q9.hgh"; Ecut=Ecut, kgrid=kgrid)
        scfres = lda_results.scfres

        # Simple real-space integration of ρ(r) on r-vector mesh: sum(ρ) / N³ ~ <ρ(r)> = ∫ρ(r) dr / Ω
        rho_avg = sum(scfres.ρ) / length(scfres.ρ)

        println("Average density (semi-core + valence) ̄ρ = $rho_avg")
        println("UEG rₛ(̄ρ) = $(get_rs(rho_avg))")
        println("Cartesian forces: $(compute_forces_cart(scfres))")

        # plot_bandstructure_custom(scfres, Ecut)
        # plot_rho(scfres, Ecut, rho_avg)
    end
    return

    # # Example using PseudoLibrary
    # psp_hgh_q9 = load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf/Si.upf");

    # NOTE: pseudopotential format must be hgh or upf
    pseudopotential_names = ["hgh/lda/na-q1.hgh", "hgh/lda/na-q9.hgh"]
    pseudopotential_labels =
        ["HGH, \$n_{\\mathrm{valence}} = 1\$", "HGH, \$n_{\\mathrm{valence}} = 9"]

    # TODO: converge parameters!
    run_lda_comparison(pseudopotential_names, pseudopotential_labels)
    return
end

main()