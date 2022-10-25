"""
Verify the lowest-order μ renormalization using the 
exact Hartree-Fock value: δμ₁ = -Σₓ(k = kF, ikₙ = 0).
"""
function test_mu_renorm()
    return false
end

@testset "Counterterms" begin
    @test_broken test_mu_renorm()
end
