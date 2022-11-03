function test_invalid_expansion(
    n_order::Int,
    observable::DiagGen.Observables,
    expansion_orders::Vector{Int},
)
    # Test settings with filter for Hartree-Fock insertions
    settings = DiagGen.Settings(;
        n_order=n_order,
        observable=observable,
        filter=[NoHartree, NoFock],
    )
    cfg = DiagGen.Config(settings, n_order)
    # Check if this is an invalid SOSEM diagrammatic expansion
    @test DiagGen._is_invalid_expansion(cfg, expansion_orders)
end

@testset verbose = false "Diagram generation" begin
    @testset "Filter invalid expansions" begin
        # Invalid Hartree-Fock insertion (n_gi == 1)
        test_invalid_expansion(3, DiagGen.c1c, [1, 0, 0])
        test_invalid_expansion(5, DiagGen.c1c, [2, 0, 1])
        # Invalid allocation of expansion order(s) to 
        # dashed Green's function line (n_di != 0)
        test_invalid_expansion(4, DiagGen.c1d, [2, 0, 0])
        test_invalid_expansion(6, DiagGen.c1d, [2, 2, 0])
        test_invalid_expansion(6, DiagGen.c1d, [2, 0, 2])
        # Invalid allocation of expanion orders for observable 
        # with 3-point vertex insertion (n_gamma == 0)
        test_invalid_expansion(4, DiagGen.c1bL, [0, 2, 0, 0])
        test_invalid_expansion(6, DiagGen.c1bL, [0, 2, 2, 0])
    end
end
