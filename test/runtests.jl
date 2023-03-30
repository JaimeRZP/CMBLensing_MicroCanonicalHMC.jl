using Test
using MicroCanonicalHMC

@testset "All tests" begin

    @testset "Settings" begin
        spl = MCHMC(0.1, 0.1, 10; integrator="MN",
                    loss_wanted=20, varE_wanted=0.01, tune_eps_nsteps=10, tune_L_nsteps=10,
                    init_eps=1.0, init_L=1.0, init_sigma=[1.0],
                    sigma=[1.0], gamma=2.0, sigma_xi=2.0)

        sett = spl.settings
        hp = spl.hyperparameters
        dy = spl.hamiltonian_dynamics
        
        @test sett.nchains == 10
        @test sett.integrator == "MN"
        @test sett.loss_wanted == 20
        @test sett.varE_wanted == 0.01
        @test sett.tune_eps_nsteps == 10
        @test sett.tune_L_nsteps == 10
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
end
