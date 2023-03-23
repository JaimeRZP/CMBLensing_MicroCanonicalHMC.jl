function tune_what(sampler::EnsembleSampler, target::ParallelTarget)
    tune_sigma, tune_eps, tune_L = false, false, false

    if sampler.hyperparameters.sigma == [0.0]
        @info "Tuning sigma ⏳"
        tune_sigma = true
        if sampler.settings.init_sigma == nothing
            init_sigma = ones(target.target.d)
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
            init_eps = 0.5
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
            init_L = sqrt(target.target.d)
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

function Summarize(samples::AbstractMatrix)
    samples = convert(Vector{Any},[samples[i,:] for i in 1:size(samples,1)])
    _samples = zeros(length(samples), length(samples[1]), 1)
    _samples[:, :, 1] = mapreduce(permutedims, vcat, samples)
    _samples = permutedims(_samples, (1,3,2))
    ess, rhat = MCMCDiagnosticTools.ess_rhat(_samples)
    return ess, rhat
end

function tune_L!(sampler::EnsembleSampler, target::ParallelTarget, init; kwargs...)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    eps = sampler.hyperparameters.eps
    
    nchains = sampler.settings.nchains
    d = target.target.d  
    
    steps = 10 .^ (LinRange(2, log10(2500), sett.tune_maxiter))
    steps = Int.(round.(steps))
    chains = zeros(sum(steps), nchains, d+2)
    l = 0
    for s in steps
        l += s 
        for i in 1:s
            state, sample = Step(sampler, target, state; kwargs...)
            X, E, L = sample
            chains[i, :, :] = [X E L]
        end
        neffs = [Neff(chains[:, i, :]) for i in 1:nchains]
        neff = mean(neffs)
        if dialog
            println(string("samples: ", length(samples), "--> 1/<1/ess>: ", neff))
        end
        if l > 10.0 / neff
            sampler.hyperparameters.L = 0.4 * eps / neff # = 0.4 * correlation length
            @info string("Found L: ", sampler.hyperparameters.L, " ✅")
            break
        end
    end
end

function Virial_loss(x::AbstractMatrix, g::AbstractMatrix)
"""loss^2 = (1/d) sum_i (virial_i - 1)^2"""

    #should be all close to 1 if we have reached the typical set
    v = mean(x .* g, dims=1)  # mean over params
    return sqrt.(mean((v .- 1.0).^2))
end

function Step_burnin(sampler::EnsembleSampler, target::ParallelTarget,
                     init; kwargs...)
    dialog = get(kwargs, :dialog, false)
    x, u, l, g, dE = init
    loss = Virial_loss(x, g)
    step = Dynamics(sampler, target, init)
    xx, uu, ll, gg, dEE = step
    lloss = Virial_loss(xx, gg)
    if dialog
        println("Virial loss: ", lloss, " --> Relative improvement: ", abs(lloss/loss - 1))
    end

    no_divergences = isfinite(loss)
    if no_divergences
        if (lloss <= sampler.settings.loss_wanted) || (abs(lloss/loss - 1) < 0.01)
            return true, step, (target.inv_transform(xx), dE .+ dEE, -ll)
        else
            return false, step, (target.inv_transform(xx), dE .+ dEE, -ll)
        end
    else
        @warn "Divergences encountered during burn-in. Reducing eps!"
        sampler.hyperparameters.eps *= 0.5
        return false, init, (target.inv_transform(x), dE, -l)
    end
end

function Init_burnin(sampler::EnsembleSampler, target::ParallelTarget,
                     init; kwargs...)
    dialog = get(kwargs, :dialog, false)
    x, _, l, g, dE = init
    v = mean(x .* g, dims=1)
    loss = mean((1 .- v).^2)
    sng = -2.0 .* (v .< 1.0) .+ 1.0
    u = -g ./ sqrt.(sum(g.^2, dims=2))
    u .*= sng

    if dialog
        println("Initial Virial loss: ", loss)
    end

    return  loss, (x, u, l, g, dE), (target.inv_transform(x), dE, -l)
end

function dual_averaging(sampler::EnsembleSampler, target::ParallelTarget, state; α=1, kwargs...)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    varE_wanted = sett.varE_wanted
    d = target.target.d

    chains = zeros(sett.num_energy_points, sett.nchains, d+2)
    for i in 1:sett.num_energy_points
        state, sample = Step(sampler, target, state; kwargs...)
        X, E, L = sample
        chains[i, :, :] = [X E L]
    end
    Es = vec(mean(chains[:, :, 3], dims=2))
    m_E = mean(Es)
    s_E = std(Es)
    Es = deleteat!(Es, abs.(Es .- m_E)/s_E.>2)
    varE = std(Es)^2 / d

    if dialog
        println("eps: ", sampler.hyperparameters.eps, " --> VarE: ", varE)
    end

    no_divergences = isfinite(varE)
    ### update the hyperparameters ###
    if no_divergences
        success = (abs(varE-varE_wanted)/varE_wanted) < 0.1
        if !success
            new_log_eps = log(sampler.hyperparameters.eps)-α*(varE-varE_wanted)
            sampler.hyperparameters.eps = exp(new_log_eps)
        end
    else
        success = false
        sampler.hyperparameters.eps *= 0.5
    end

    return success
end

function tune_nu!(sampler::EnsembleSampler, target::ParallelTarget)
    eps = sampler.hyperparameters.eps
    L = sampler.hyperparameters.L
    d = target.target.d
    sampler.hyperparameters.nu = eval_nu(eps, L, d)
end

function tune_hyperparameters(sampler::EnsembleSampler, target::ParallelTarget, init;                                 burn_in::Int=0, kwargs...)
    ### debugging tool ###
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    d = target.target.d

    tune_sigma, tune_eps, tune_L = tune_what(sampler, target)
        
     
    if tune_sigma
        xs, _, _, _, _ = init
        sigma = vec(std(xs, dims=1))
        sampler.hyperparameters.sigma = sigma
        if dialog
            println(string("Initial sigma: ", sigma))
        end
    end    
    
    if burn_in > 0
        @info "Starting burn in ⏳" 
        loss, init, sample = Init_burnin(sampler, target, init; kwargs...)    
        for i  in 1:burn_in
            finished, init, sample = Step_burnin(sampler, target, init; kwargs...)
            if tune_sigma
                x, _, _, _, _ = init
                xs = [xs; x]
                sigma = vec(std(xs, dims=1))
                sampler.hyperparameters.sigma = sigma
                if dialog
                    println(string("Sigma --> ", sigma))
                end
            end
            if finished
                @info string("Virial loss condition met during burn-in at step: ", i)
                break
            end
            if i == burn_in
                @warn "Maximum number of steps reached during burn-in"
            end            
        end
        if tune_sigma
            @info string("Found sigma: ", sampler.hyperparameters.sigma, " ✅")
        end            
    end                               

    if tune_eps
        tuning_method = get(kwargs, :tuning_method, "AdaptiveStep")
        if dialog
            println(string("Using eps tuning method ", tuning_method))        
        end            
        if tuning_method=="DualAveraging"    
            for i in 1:sett.tune_maxiter
                success = dual_averaging(sampler, target, init;
                                         α=exp.(-(i .- 1)/20), kwargs...)
                if success
                    break
                end
            end                
        end
    end

    if tune_L
        tune_L!(sampler, target, init; kwargs...)
    end

    tune_nu!(sampler, target)
                    
    return init, sample
end
