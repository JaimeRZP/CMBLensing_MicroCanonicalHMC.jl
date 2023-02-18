function Virial_loss(x::AbstractMatrix, g::AbstractMatrix)
"""loss^2 = (1/d) sum_i (virial_i - 1)^2"""

    #should be all close to 1 if we have reached the typical set
    v = mean(x .* g, dims=1)  # mean over params
    return sqrt.(mean((v .- 1.0).^2))
end

function Step_burnin(sampler::EnsembleSampler, target::ParallelTarget,
                     state; kwargs...)
    steps, loss, fail_count, never_rejected, x, u, l, g, dE = state
    # Update diagonal conditioner
    sampler.hyperparameters.sigma = std(x, dims=2) #std over particles

    step = Dynamics(sampler, target, state)
    xx, uu, ll, gg, dEE = step
    lloss = Virial_loss(xx, gg)

    #will we accept the step?
    # if reject --> reduce eps
    # if accept
    #   if never_rejected --> increase eps
    if (lloss < loss) && (all(isfinite(xx)))
        accept = true
        fail_count = 0
        if never_rejected
           sampler.hyperparameter.eps *= 2.0
        end
        return steps+1, lloss, fail_count, never_rejected, xx, uu, ll, gg, dEE
    else
        accept = false
        never_rejected *= accept
        fail_count += 1
        sampler.hyperparameter.eps *= 0.5
        return steps+1, loss, fail_count, never_rejected, x, u, l, g, dE
    end
end

function Init_burnin(sampler::EnsembleSampler, target::ParallelTarget, init)
    x, _, l, g, dE = init
    v = mean(x .* g, dims=1)
    loss = mean((1 .- v).^2)
    sng = -2.0 .* (v .< 1.0) .+ 1.0
    u = -g ./ sqrt.(sum(g.^2, dims=2))
    u .*= sng
    steps = 0
    fail_count = 0
    never_rejected = true
    init = (steps, loss, fail_count, never_rejected,
            x, u, l, g, dE)
    return init
end

function tune_L!(sampler::EnsembleSampler, target::ParallelTarget, init; kwargs...)
    @info "Not Implemented using L = sqrt(target.d)"
    sampler.hyperparameters.L = sqrt(target.d)
end

function tune_nu!(sampler::EnsembleSampler, target::ParallelTarget)
    eps = sampler.hyperparameters.eps
    L = sampler.hyperparameters.L
    d = target.target.d
    sampler.hyperparameters.nu = eval_nu(eps, L, d)
end

function Burnin(sampler::EnsembleSampler, target::ParallelTarget, init, burnin)
    ### debugging tool ###
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    init = Init_burnin(sampler, target, init)

    if sampler.hyperparameters.L == 0.0
        @info "Tuning L ⏳"
        tune_L!(sampler, target, init; kwargs...)
    end

    for i in 1:burnin
        step_burnin = Step_burnin(sampler, target, init)
        steps, loss, fail_count, never_rejected, x, u, l, g, dE = step_burnin

        if loss < sett.loss_wanted
            @info "Virial loss condition met during burn-in"
            break
        end

        if fail_count >= sett.tune_max_iter
            @info "Maximum number of failed proposals reached during burn-in"
            break
        end

        if i == burnin
            @info "Maximum number of steps reached during burn-in"
        end
    end

    ### determine the epsilon for sampling ###
    #the epsilon before the row of failures, we will take this as a baseline
    eps = sampler.hyperparameters.eps * (1.0/2.0)^fail_count
    #some range around the wanted energy error
    energy_range = sett.varE_wanted .* 10 .^ (range(-2, stop=2, length=sett.num_energy_points))
    #assume Var[E] ~ eps^6 and use the already computed point to set out the grid for extrapolation
    varE =  mean(dE .^ 2)/target.target.d
    epsilon = eps .* (energy_range ./ varE) .^ (1.0/6.0)

    tune_nu!(sampler, target)
    return nothing
end

################
### Old code ###
################

function tune_L!(sampler::EnsembleSampler, target::ParallelTarget, init; kwargs...)
    @info "Not Implemented using L = sqrt(target.d)"
    sampler.hyperparameters.L = sqrt(target.d)
end

function tune_eps!(sampler::EnsembleSampler, target::ParallelTarget, init; kwargs...)
    @info "Not Implemented using eps = sqrt(target.d)"
    sampler.hyperparameters.eps = sqrt(target.d)
    return true
end

function tune_nu!(sampler::EnsembleSampler, target::ParallelTarget)
    eps = sampler.hyperparameters.eps
    L = sampler.hyperparameters.L
    d = target.target.d
    sampler.hyperparameters.nu = eval_nu(eps, L, d)
end

function tune_hyperparameters(sampler::EnsembleSampler, target::ParallelTarget, init; kwargs...)
    ### debugging tool ###
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    init = Init_burnin(sampler, target, init)

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
