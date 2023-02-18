function tune_L!(sampler::EnsembleSampler, target::ParallelTarget, init; kwargs...)
    @info "Not Implemented using L = sqrt(target.d)"
    sampler.hyperparameters.L = sqrt(target.d)
end

function Virial_loss(x::AbstractMatrix, g::AbstractMatrix)
"""loss^2 = (1/d) sum_i (virial_i - 1)^2"""

    #should be all close to 1 if we have reached the typical set
    v = mean(x .* g, dims=1)  # mean over params
    return sqrt.(mean((v .- 1.0).^2))
end

function Step_burnin(sampler::EnsembleSampler, target::ParallelTarget,
                     loss, init; kwargs...)
    dialog = get(kwargs, :dialog, false)
    step = Dynamics(sampler, target, init)
    xx, uu, ll, gg, dEE = step
    lloss = Virial_loss(xx, gg)

    sigma = std(xx, dims=2)
    if dialog
        println("Virial loss: ", lloss, " --> sigma: ", sigma)
    end

    if (all(isfinite(xx)))
        #Update the preconditioner
        sampler.hyperparameters.sigma = sigma
        if (abs(lloss/loss - 1) < 0.05)
            return true, step
        else
            return false, step
        end
    else
        return false, init
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

    return  loss, (x, u, l, g, dE)
end

function tune_sigma(sampler::EnsembleSampler, target::ParallelTarget, init;
                     burnin=10, kwargs...)
    ### debugging tool ###
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    loss, step = Init_burnin(sampler, target, init; kwargs...)

    for i in 1:burnin
        finished, step = Step_burnin(sampler, target, loss, step; kwargs...)
        if finished
            @info "Virial loss condition met during burn-in"
            break
        end
        if i == burnin
            @info "Maximum number of steps reached during burn-in"
        end
    end
    return step
end

function tune_eps!(sampler::EnsembleSampler, target::ParallelTarget, state; kwargs...)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    varE_wanted = sett.varE_wanted
    d = target.target.d

    state, (X, E, L) = Step(sampler, target, state; kwargs...)
    varE = std(E)^2 / target.d #variance per dimension
    if dialog
        println("eps: ", eps, " --> VarE: ", varE)
    end
    no_divergences = isfinite(varE)
    ### update the hyperparameters ###
    if no_divergences
        success = varE < varE_wanted #we are done
        if !success
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

function tune_nu!(sampler::EnsembleSampler, target::ParallelTarget)
    eps = sampler.hyperparameters.eps
    L = sampler.hyperparameters.L
    d = target.target.d
    sampler.hyperparameters.nu = eval_nu(eps, L, d)
end

function Burnin(sampler::EnsembleSampler, target::ParallelTarget, init; kwargs...)
    ### debugging tool ###
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings

    if sampler.hyperparameters.L == 0.0
        @info "Tuning L ⏳"
        tune_L!(sampler, target, init; kwargs...)
    end
    if sampler.hyperparameters.sigma == [0.0]
        @info "Tuning sigma ⏳"
        x, _, _, _, _ = init
        sampler.hyperparameters.sigma = std(x, dims=2)
        init = tune_sigma(sampler, target, init; kwargs...)
        @info string("Found sigma: ", sampler.hyperparameters.sigma, " ✅")
    end
    if sampler.hyperparameters.eps == 0.0
        @info "Tuning eps ⏳"
        sampler.hyperparameters.eps = sqrt(target.target.d)
        for i in 1:sett.tune_maxiter
            if tune_eps!(sampler, target, init; kwargs...)
                break
            end
        end
    else
        @info "Using given hyperparameters"
        @info string("Found eps: ", sampler.hyperparameters.eps, " ✅")
        @info string("Found L: ", sampler.hyperparameters.L, " ✅")
    end
    tune_nu!(sampler, target)

    x, u, l, g, dE = init
    sample = (target.inv_transform(x), dE, -l)

    return init, sample
end
