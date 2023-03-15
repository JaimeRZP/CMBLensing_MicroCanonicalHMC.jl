function tune_what(sampler::Sampler, target::Target)
    tune_sigma, tune_eps, tune_L = false, false, false

    if sampler.hyperparameters.sigma == [0.0]
        @info "Tuning sigma ⏳"
        tune_sigma = true
        if sampler.settings.init_sigma == nothing
            init_sigma = ones(target.d)
        else
            init_sigma = sampler.settings.init_sigma
        end
        sampler.hyperparameters.sigma = init_sigma
    else
        @info "Using given sigma ✅"
    end

    if sampler.hyperparameters.eps == 0.0
        @info "Tuning eps ⏳"
        tune_eps = true
        if sampler.settings.init_eps == nothing
            init_eps = 0.5#*sqt(d)
        else
            init_eps = sampler.settings.init_eps
        end
        sampler.hyperparameters.eps = init_eps
    else
        @info "Using given eps ✅"
    end

    if sampler.hyperparameters.L == 0.0
        @info "Tuning L ⏳"
        tune_L = true
        if sampler.settings.init_sigma == nothing
            init_L = sqrt(target.d)
        else
            init_L = sampler.settings.init_L
        end
        sampler.hyperparameters.L = init_L
    else
        @info "Using given L ✅"
    end

    tune_nu!(sampler, target)

    return tune_sigma, tune_eps, tune_L
end

function ess_corr(samples)
    _samples = zeros(length(samples), length(samples[1]), 1)
    _samples[:, :, 1] = mapreduce(permutedims, vcat, samples)
    _samples = permutedims(_samples, (1,3,2))
    ess, rhat = MCMCDiagnosticTools.ess_rhat(_samples)

    neff = ess ./ length(samples)
    return 1.0 / mean(1 ./ neff)
end

function tune_L!(sampler::Sampler, target::Target, init; kwargs...)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    eps = sampler.hyperparameters.eps
    sampler.hyperparameters.L = sqrt(target.d)
    steps = 10 .^ (LinRange(2, log10(2500), sett.tune_maxiter))
    steps = Int.(round.(steps))
    samples = []
    for s in steps
        for i in 1:s
            init, sample = Step(sampler, target, init; monitor_energy=true)
            push!(samples, sample)
        end
        ESS = ess_corr(samples)
        if dialog
            println(string("samples: ", length(samples), "--> ESS: ", ESS))
        end
        if length(samples) > 10.0 / ESS
            sampler.hyperparameters.L = 0.4 * eps / ESS # = 0.4 * correlation length
            @info string("Found L: ", sampler.hyperparameters.L, " ✅")
            break
        end
    end
end

function tune_sigma!(sampler::Sampler, target::Target; kwargs...)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    MAP_t = target.MAP_t
    Hess = target.hess_nlogp(MAP_t)
    mass_mat = pinv(Hess)
    sigma = sqrt.(diag(mass_mat))
    sampler.hyperparameters.sigma = sigma
    @info string("Found sigma: ", sampler.hyperparameters.sigma, " ✅")
end

function tune_eps!(sampler::Sampler, target::Target, init; α=1, kwargs...)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    eps = sampler.hyperparameters.eps
    L = sqrt(target.d) #sampler.hyperparameters.L
    varE_wanted = sett.varE_wanted
    #x, u, g, time = init

    samples = []
    for i in 1:sett.tune_samples
        init, sample = Step(sampler, target, init;
                            monitor_energy=true)
        push!(samples, sample)
    end

    samples = mapreduce(permutedims, vcat, samples)
    E = samples[:, end-1]
    varE = std(E) / target.d #variance per dimension
    if dialog
        println("eps: ", eps, " --> VarE/d: ", varE)
    end
    no_divergences = isfinite(varE)

    ### update the hyperparameters ###
    if no_divergences
        success = (abs(varE-varE_wanted)/varE_wanted) < 0.05
        if !success
            new_log_eps = log(sampler.hyperparameters.eps)-α*(varE-varE_wanted)
            new_log_eps = max(log(0.00005), new_log_eps)
            sampler.hyperparameters.eps = exp(new_log_eps)
        else
            @info string("Found eps: ", sampler.hyperparameters.eps, " ✅")
        end
    else
        success = false
        sampler.hyperparameters.eps = 0.5 * eps
    end

    return success
end

function eval_nu(eps, L, d)
    nu = sqrt((exp(2 * eps / L) - 1.0) / d)
    return nu
end

function tune_nu!(sampler::Sampler, target::Target)
    eps = sampler.hyperparameters.eps
    L = sampler.hyperparameters.L
    d = target.d
    sampler.hyperparameters.nu = eval_nu(eps, L, d)
end

function tune_hyperparameters(sampler::Sampler, target::Target, init; kwargs...)
    sett = sampler.settings
    ### debugging tool ###
    dialog = get(kwargs, :dialog, false)

    tune_sigma, tune_eps, tune_L = tune_what(sampler, target)

    if tune_sigma
        tune_sigma!(sampler, target; kwargs...)
    end

    if tune_eps
        for i in 1:sett.tune_maxiter
            α = exp.(-(i .- 1)/20)
            if tune_eps!(sampler, target, init; α=α, kwargs...)
                break
            end
        end
    end

    if tune_L
        tune_L!(sampler, target, init; kwargs...)
    end

    tune_nu!(sampler, target)
end
