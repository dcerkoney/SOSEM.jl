using SOSEM: DiagGen

function main()
    settings = DiagGen.Settings(n_order=4, plot_trees=true, verbosity=2)
    # Build diagram and expression trees for all sigma_2 diagrams at order n
    sigma2, sigma2_compiled = DiagGen.build_c1bL_with_ct(settings)
    # Check the trees
    DiagGen.checktrees(sigma2, sigma2_compiled, settings)
    return sigma2, sigma2_compiled
end

sigma2, sigma2_compiled = main()