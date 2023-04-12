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

    @testset "Init" begin
        d = 10
        m = zeros(d)
        s = Diagonal(ones(d))
        target = GaussianTarget(m, s)
        spl = MCHMC(0, 0.001)
        init = MicroCanonicalHMC.Init(spl, target; init_x=m)
        @test init.x == m
        @test init.g == m
        @test init.dE == init.Feps == 0.0
        @test init.Weps ==  1.0e-5
    end

    @testset "Step" begin
        d = 10
        m = zeros(d)
        s = Diagonal(ones(d))
        target = GaussianTarget(m, s)
        aspl = spl = MCHMC(0, 0.001; eps=0.1, L=0.1, sigma=ones(d))
        tune_sigma, tune_eps, tune_L = tune_what(spl, target)
        tune_sigma, tune_eps, tune_L = tune_what(aspl, target)
        @test tune_sigma == tune_eps == tune_L == false
        init = MicroCanonicalHMC.Init(spl, target; init_x=m)
        step = MicroCanonicalHMC.Step(spl, target; init_x=m)
        astep = MicroCanonicalHMC.Step(aspl, target; init_x=m, adaptive=true)
        @test spl.hyperparameters.eps == 0.1
        @test aspl.hyperparameters.eps != 0.1
        @test step.x == astep.x 
    end
end
