using ASEconvert
using DFTK
using Interpolations
using LazyArtifacts
using Unitful
using UnitfulAtomic

function run_lda(
    psp_name;
    temperature=0.01,
    tol=1e-5,
    Ecut=50,
    kgrid=[8, 8, 8],
    plots=false,
)
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

function main()
    # Change to project directory
    if haskey(ENV, "SOSEM_CEPH")
        cd(ENV["SOSEM_CEPH"])
    elseif haskey(ENV, "SOSEM_HOME")
        cd(ENV["SOSEM_HOME"])
    end

    Ecut = 75
    kgrid = [9, 9, 9]

    lda_results = run_lda("hgh/lda/na-q1.hgh"; Ecut=Ecut, kgrid=kgrid)
    scfres = lda_results.scfres
    rho = scfres.ρ[:, :, :, 1]  # ρ for spin up (wlog)

    # Get basis vectors along Cartesian axes x, y, z
    rvecs = collect(r_vectors(scfres.basis))
    xvecs = rvecs[:, 1, 1]  # slice along the x axis
    yvecs = rvecs[1, :, 1]  # slice along the y axis
    zvecs = rvecs[1, 1, :]  # slice along the z axis
    xs = [xvec[1] for xvec in xvecs]
    ys = [yvec[2] for yvec in yvecs]
    zs = [zvec[3] for zvec in zvecs]

    xyz = vec([[x, y, z] for x in xs, y in ys, z in zs])
    rho_xyz =
        vec([rho[i, j, k] for i in eachindex(xs), j in eachindex(ys), k in eachindex(zs)])

    # Simple real-space integration of ρ(r) on r-vector mesh: sum(ρ) / N³ ~ <ρ(r)> = ∫ρ(r) dr / Ω
    rho_avg = sum(scfres.ρ) / length(scfres.ρ)

    return rho, rho_xyz, xyz, rho_avg
end

main()