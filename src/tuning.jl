
function ess_corr(target::Target, samples)
    param_names = target.vsyms
    internal_names = [:E, :logp]
    names = [param_names; internal_names]
    samples = MCMCChains.concretize(samples)
    chain = MCMCChains.Chains(samples, names, (internals = internal_names,))
    stats = summarize(chain)
    esss = stats[:, :ess]  # effective sample size in each dimension

    ### my part (combine all dimensions): ###
    neff = esss ./ length(samples)
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
        ESS = ess_corr(target, samples)
        if dialog
            println(string("samples: ", length(samples), "--> ESS: ", ESS))
        end
        if length(samples) > 10.0 / ESS
            @info string("Found L: ", sampler.hyperparameters.L, " ✅")
            sampler.hyperparameters.L = 0.4 * eps / ESS # = 0.4 * correlation length
            break
        end
    end
end

function tune_eps!(sampler::Sampler, target::Target, init; kwargs...)
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
    varE = std(E)^2 / target.d #variance per dimension
    if dialog
        println("eps: ", eps, " --> VarE: ", varE)
    end
    no_divergences = isfinite(varE)

    ### update the hyperparameters ###
    if no_divergences
        success = varE < varE_wanted #we are done
        if !success
            #eps_new = eps*(varE_wanted/varE)^0.25 #assume var[E] ~ eps^4
            sampler.hyperparameters.eps = 0.5 * eps
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

    # Init guess
    if sampler.hyperparameters.eps == 0.0
        @info "Tuning eps ⏳"
        sampler.hyperparameters.eps = 0.5
        for i in 1:sett.tune_maxiter
            if tune_eps!(sampler, target, init; kwargs...)
                break
            end
        end
    end
    if sampler.hyperparameters.L == 0.0
        @info "Tuning L ⏳"
        tune_L!(sampler, target, init; kwargs...)
    else
        @info "Using given hyperparameters"
        @info string("Found eps: ", sampler.hyperparameters.eps, " ✅")
        @info string("Found L: ", sampler.hyperparameters.L, " ✅")
    end
    tune_nu!(sampler, target)
end
