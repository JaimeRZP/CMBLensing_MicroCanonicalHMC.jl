#=
function accept_reject_step(old_state, new_state)
    loss, x, u, l, g, varE = old_state
    loss_new, xx, uu, ll, gg, varE_new = new_state

    no_nans = all(isfinite(xx))
    tru = (loss_new < loss) * no_nans
    fals = (1 - tru)
    Loss = loss_new * tru + loss * false

    replace!(xx, NaN=>0.0)
    replace!(uu, NaN=>0.0)
    replace!(ll, NaN=>0.0)
    replace!(gg, NaN=>0.0)
    replace!(varE_new, NaN=>0.0)

    X = xx * tru + x * fals
    U = uu * tru + u * fals
    L = ll * tru + l * fals
    G = gg * tru + g * fals
    Var = varE_new * tru + varE * fals

    return tru, Loss, X, U, L, G, Var
end
=#

function Virial_loss(x::AbstractMatrix, g::AbstractMatrix)
"""loss^2 = (1/d) sum_i (virial_i - 1)^2"""

    #should be all close to 1 if we have reached the typical set
    v = mean(x .* g, dims=1)  # mean over params
    return sqrt.(mean((v .- 1.0).^2))
end

function Step_burnin(sampler::EnsembleSampler, target::ParallelTarget,
                     state; kwargs...)
    steps, loss, fail_count, never_rejected, x, u, l, g = state
    # Update diagonal conditioner
    sampler.hyperparameters.sigma = std(x, dims=2) #std over particles

    step = Dynamics(sampler, target, state)
    xx, uu, ll, gg, kinetic_change, time = step
    EE = dEnergy(l ,ll, E, kinetic_change)

    lloss = Virial_loss(xx, gg)
    varEE = mean((EE).^2)/target.target.d

    #will we accept the step?
    accept, loss, x, u, l, g, varE = accept_reject_step(loss, x, u, l, g, varE, loss_new, xx, uu, ll, gg, varE_new)
    #Ls.append(loss)
    #X.append(x)
    never_rejected *= accept #True if no step has been rejected so far
    fail_count = (fail_count + 1) * (1-accept) #how many rejected steps did we have in a row

                    #reduce eps if rejected    #increase eps if never rejected        #keep the same
    eps = eps * ((1-accept) * self.reduce + accept * (never_rejected * self.increase + (1-never_rejected) * 1.0))
    #epss.append(eps)

    return steps + 1, loss, fail_count, never_rejected, x, u, l, g, key, L, eps, sigma, varE
end

function Init_burnin(sampler::EnsembleSampler, target::ParallelTarget,
                     init)
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
            x, u, l, g)
    return init
end

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
