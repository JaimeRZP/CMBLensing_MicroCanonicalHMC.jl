@testset "Sample" begin
    ##############
    ### Neal's ###
    ##############
    d = 21
    @model function funnel()
        θ ~ Truncated(Normal(0, 3), -3, 3)
        z ~ MvNormal(zeros(d-1), exp(θ)*I)
        x ~ MvNormal(z, I)
    end

    Random.seed!(1)
    (;x) = rand(funnel() | (θ=0,))
    funnel_model = funnel() | (;x)

    target = TuringTarget(funnel_model; d=d, compute_MAP=false)

    spl = MCHMC(10_000, 0.01)
    samples_mchmc = Sample(spl, target, 100_000; dialog=false)

    theta_mchmc = [sample[1] for sample in samples_mchmc]
    x10_mchmc = [sample[10+1] for sample in samples_mchmc]
    E = [sample[end-1] for sample in samples_mchmc];
    VarE = std(E)^2/d

    @test VarE ≈ 0.01 atol=0.03
    @test mm1 ≈ 0.07 atol=0.01
    @test m1 ≈ 0.01 atol=0.01
    @test s1 ≈ 0.72 atol=0.1
    @test mm2 ≈ -0.82 atol=0.1
    @test m2 ≈ -0.86 atol=0.1
    @test s2 ≈ 0.76 atol=0.1

    ##################
    ### Rosembrock ### 
    ##################
    rng = MersenneTwister(1234)
    target = RosenbrockTarget(1.0, 10.0; d=2)
    spl = MCHMC(10_000, 0.01; L=sqrt(2), sigma=ones(target.d), adaptive=true)
    samples = Sample(rng, spl, target, 200_000; dialog=true)
    d1 = [sample[1] for sample in samples]
    d2 = [sample[2] for sample in samples]
    mm1, m1, s1, = (median(d1), mean(d1), std(d1))
    mm2, m2, s2, = (median(d2), mean(d2), std(d2))
    @test mm1 ≈ 1.00 atol=0.1
    @test m1 ≈  1.00 atol=0.1
    @test s1 ≈  1.00 atol=0.3
    @test mm2 ≈ 1.13 atol=0.1
    @test m2 ≈  1.97 atol=0.1
    @test s2 ≈  2.40 atol=0.5
end