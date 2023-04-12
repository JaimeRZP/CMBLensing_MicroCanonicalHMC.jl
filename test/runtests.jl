using Test
using MicroCanonicalHMC
using LinearAlgebra
using Random

@testset "All tests" begin

    @testset "Settings" begin
        spl = MCHMC(10_000, 0.1; nchains=10, integrator="MN",
                    eps=0.1, L=0.1, sigma=[1.0], gamma=2.0, sigma_xi=2.0,
                    init_eps=1.0, init_L=1.0, init_sigma=[1.0])

        sett = spl.settings
        hp = spl.hyperparameters
        dy = spl.hamiltonian_dynamics
        
        @test sett.nchains == 10
        @test sett.integrator == "MN"
        @test sett.TEV == 0.1
        @test sett.nadapt == 10_000
        @test sett.init_eps == 1.0
        @test sett.init_L == 1.0
        @test sett.init_sigma == [1.0]
        
        @test hp.eps == 0.1
        @test hp.L == 0.1
        @test hp.sigma == [1.0]
        @test hp.gamma == 2.0
        @test hp.sigma_xi == 2.0
        
        @test dy == MicroCanonicalHMC.Minimal_norm
    end

    @testset "Partially_refresh_momentum" begin
        d = 10
        rng = MersenneTwister(0)
        u = MicroCanonicalHMC.Random_unit_vector(rng, d)
        @test length(u) == d
        @test isapprox(norm(u),  1.0, rtol=0.0000001)

        p = MicroCanonicalHMC.Partially_refresh_momentum(rng, 0.1, d, u)
        @test length(p) == d
        @test isapprox(norm(p),  1.0, rtol=0.0000001)
    end
end
