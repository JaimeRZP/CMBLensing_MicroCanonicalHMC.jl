@testset "Sample" begin
    ################
    ### Gaussian ###
    ################
    d = 20
    k = 100
    m = Vector(LinRange(1, 100, d))
    e = 10 .^ LinRange(log10(1/sqrt(k)), log10(sqrt(k)), d)
    cov_matt = Diagonal(e);
    target = GaussianTarget(m, cov_matt)

    spl = MCHMC(10_000, 0.01; init_eps=sqrt(d))
    samples_mchmc = Sample(spl, target, 100_000; dialog=true);
    samples_mchmc_adaptive = Sample(spl, target, 100_000;
        adaptive=true, dialog=true);

     _samples_mchmc = mapreduce(permutedims, vcat, samples_mchmc)
    s1 = std(_samples_mchmc, dims=1)[1:end-3]
    m1 = mean(_samples_mchmc, dims=1)[1:end-3]

    _samples_mchmc_adaptive = mapreduce(permutedims, vcat, samples_mchmc_adaptive)
    s2 = std(_samples_mchmc_adaptive, dims=1)[1:end-3]
    m2 = mean(_samples_mchmc_adaptive, dims=1)[1:end-3]

    @test mean((m1 .- m)./sqrt.(e)) ≈ 0.0 atol=0.2
    @test mean(s1 ./sqrt.(e) .-1) ≈ 0.0 atol=0.2
    @test mean((m2 .- m)./sqrt.(e)) ≈ 0.0 atol=0.2
    @test mean(s2./sqrt.(e) .-1) ≈ 0.0 atol=0.2

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

    spl = MCHMC(1_000, 0.01)
    samples_mchmc = Sample(spl, target, 10_000; dialog=false)

    theta_mchmc = [sample[1] for sample in samples_mchmc]
    x10_mchmc = [sample[10+1] for sample in samples_mchmc]
    mm1, m1, s1 = (median(theta_mchmc), mean(theta_mcjmc), std(theta_mchmc))
    mm2, m2, s2 = (median(x10_mchmc), mean(x10_mcjmc), std(x10_mchmc))
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
    @test mm1 ≈ 1.00 atol=0.2
    @test m1 ≈  1.00 atol=0.2
    @test s1 ≈  1.00 atol=0.3
    @test mm2 ≈ 1.13 atol=0.2
    @test m2 ≈  1.97 atol=0.2
    @test s2 ≈  2.40 atol=0.5
end