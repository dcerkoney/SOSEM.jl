function test_chemicalpotential_renormalization()
    # Create some test data
    function test_data(n_min=0)
        δμ = [π, π^(1 / 5), π^(1 / 7), π^(1 / 9)]
        min_order = n_min
        max_order = n_min + 4
        data = Dict(
            # 1st order
            (n_min, 0, 0) => 1.0,
            (n_min + 1, 0, 0) => 1.0,
            (n_min, 1, 0) => 1.0,
            # 2nd order
            (n_min + 2, 0, 0) => 1.0,
            (n_min + 1, 1, 0) => 1.0,
            (n_min + 0, 2, 0) => 1.0,
            (n_min + 0, 1, 0) => 1.0,
            # 3rd order
            (n_min + 3, 0, 0) => 1.0,
            (n_min + 2, 1, 0) => 1.0,
            (n_min + 1, 2, 0) => 1.0,
            (n_min + 0, 3, 0) => 1.0,
            (n_min + 1, 1, 0) => 1.0,
            (n_min + 0, 2, 0) => 1.0,
            (n_min + 0, 1, 0) => 1.0,
            # 4th order
            (n_min + 4, 0, 0) => 1.0,
            (n_min + 3, 1, 0) => 1.0,
            (n_min + 2, 2, 0) => 1.0,
            (n_min + 1, 3, 0) => 1.0,
            (n_min + 0, 4, 0) => 1.0,
            (n_min + 2, 1, 0) => 1.0,
            (n_min + 1, 2, 0) => 1.0,
            (n_min + 0, 3, 0) => 1.0,
            (n_min + 0, 2, 0) => 1.0,
            (n_min + 1, 1, 0) => 1.0,
            (n_min + 0, 1, 0) => 1.0,
        )
        return δμ, min_order, max_order, data
    end

    # Call the `chemicalpotential_renormalization` function
    function renorm_data(n_min=0)
        δμ, min_order, max_order, data = test_data(n_min)
        return UEG_MC.chemicalpotential_renormalization(
            data,
            δμ;
            n_min=n_min,
            min_order=min_order,
            max_order=max_order,
        )
    end

    for n_min in 0:2
        @testset "n_min = $n_min" begin
            res = renorm_data(n_min)
            δμ = [π, π^(1 / 5), π^(1 / 7), π^(1 / 9)]
            #! format: off
            @test res[n_min    ] == 1
            @test res[n_min + 1] == 1 + δμ[1]
            @test res[n_min + 2] == 1 + δμ[1] + δμ[1]^2 + δμ[2]
            @test res[n_min + 3] == 1 + δμ[1] + δμ[1]^2 + δμ[1]^3 +
                                    δμ[2] + 2δμ[1] * δμ[2] + δμ[3]
            @test res[n_min + 4] == 1 + δμ[1] + δμ[1]^2 + δμ[1]^3 + δμ[1]^4 +
                                    δμ[2] + δμ[3] + δμ[4] + 2δμ[1] * δμ[2] +
                                    3δμ[1]^2 * δμ[2] + (δμ[2]^2 + 2δμ[1] * δμ[3])
            #! format: on
        end
    end
end

@testset "Renormalization" begin
    test_chemicalpotential_renormalization()
end
