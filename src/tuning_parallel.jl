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

function Virial_loss(xs, gs)
"""loss^2 = (1/d) sum_i (virial_i - 1)^2"""

    virials = mean(xs*gs) #should be all close to 1 if we have reached the typical set
    return sqrt.(mean((virials .- 1.0).^2))
end

function Init_burnin(sampler::Sampler, target::Target; kwargs...)
    sett = sampler.settings
    kwargs = Dict(kwargs)
    ### initial conditions ###
    if :initial_x ∈ keys(kwargs)
        x0 = target.transform(kwargs[:initial_x])
        get_x0::Function = _ -> x0
    else
        get_x0::Function = _ -> target.prior_draw(sett.key)
    end

    xs = Vector{Any, nchains}
    ls = Vector{Any, nchains}
    gs = Vector{Any, nchains}
    virials = Vector{Any, nchains}
    @Threads.Threads() :static for i in nchains
        x = get_x0()
        l, g = target.nlogp_grad_nlogp(x)
        virial = x .* g

        xs[i] = x
        ls[i] = l
        gs[i] = g
        virials[i] = virial
    end

    loss = sqrt.(mean((virials .- 1.0).^2))
    sng = -2.0 .* (virials .< 1.0) .+ 1.0
    us = -.gs ./ sqrt.(sum((gs.^2))

    return loss, xs, us, ls, gs
end

function Step_Burnin(sampler::Sampler, target::Target, state; kwargs...)
    return nothing
end

function Burnin_parallel(sampler::Sampler, target::Target; kwargs...)
    return nothing
end

function tune_L_parallel!(sampler::Sampler, target::Target, init; kwargs...)
    @Info "Not Implemented using L = sqrt(target.d)"
    sampler.hyperparameters.L = sqrt(target.d)
end

function tune_eps_parallel!(sampler::Sampler, target::Target, init; kwargs...)
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

function tune_nu!(spl::Sampler, trg::Target)
    spl.hyperparameters.nu = eval_nu(spl.hyperparameters.eps, spl.hyperparameters.L, trg.d)
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
        if dialog
            println("Using given hyperparameters")
        end
    end
    tune_nu!(sampler, target)
end
