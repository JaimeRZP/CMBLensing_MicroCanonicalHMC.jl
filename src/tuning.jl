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
            init_L = sqrt(target.d)
        else
            init_L = sampler.settings.init_L
        end
        sampler.hyperparameters.L = init_L
    else
        @info "Using given L ✅"
    end

    tune_nu!(sampler, target)
    @info string("Initial nu ", sampler.hyperparameters.nu)
    
    return tune_sigma, tune_eps, tune_L
end

function Summarize(samples::AbstractVector)
    _samples = zeros(length(samples), 1, length(samples[1]))
    _samples[:, 1, :] = mapreduce(permutedims, vcat, samples)
    ess, rhat = MCMCDiagnosticTools.ess_rhat(_samples)
    return ess, rhat
end

function Neff(samples, l::Int)
    ess, rhat = Summarize(samples)
    neff = ess ./ l
    return 1.0 / mean(1 ./ neff)
end

function tune_L!(sampler::Sampler, target::Target, init; kwargs...)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    eps = sampler.hyperparameters.eps
    
    steps = 10 .^ (LinRange(2, log10(2500), sett.tune_maxiter))
    steps = Int.(round.(steps))
    samples = []
    l = 0
    for s in steps
        l += s
        for i in 1:s
            init, sample = Step(sampler, target, init; monitor_energy=true)
            push!(samples, sample)
        end
        neff = Neff(samples, l)
        if dialog
            println(string("samples: ", l, "--> 1/<1/ess>: ", neff))
        end
        if l > (10.0/neff)
            sampler.hyperparameters.L = 0.4 * eps / neff # = 0.4 * correlation length
            @info string("Found L: ", sampler.hyperparameters.L, " ✅")
            break
        end
    end
end

function get_Hessian_precond(sampler::Sampler, target::Target; kwargs...)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    MAP_t = target.MAP_t
    Hess = target.hess_nlogp(MAP_t)
    return pinv(Hess)
end

function Virial_loss(x::AbstractVector, g::AbstractVector)
"""loss^2 = (1/d) sum_i (virial_i - 1)^2"""

    #should be all close to 1 if we have reached the typical set
    v = mean(x .* g)  # mean over params
    return sqrt.((v .- 1.0).^2)
end

function Step_burnin(sampler::Sampler, target::Target,
                     init; kwargs...)
    dialog = get(kwargs, :dialog, false)
    step = Dynamics(sampler, target, init)
    xx, uu, ll, gg, dEE = step
    lloss = Virial_loss(xx, gg)    
    return lloss, step, [target.inv_transform(xx); dEE; -ll]
end

function Init_burnin(sampler::Sampler, target::Target,
                     init; kwargs...)
    dialog = get(kwargs, :dialog, false)
    x, _, l, g, dE = init
    v = mean(x .* g, dims=1)
    loss = mean((1 .- v).^2)
    sng = -2.0 .* (v .< 1.0) .+ 1.0
    u = -g ./ sqrt.(sum(g.^2))
    u .*= sng

    if dialog
        println("Initial Virial loss: ", loss)
    end

    return  loss, (x, u, l, g, dE), [target.inv_transform(x); dE; -l]
end    

function dual_averaging(sampler::Sampler, target::Target, init; α=1, kwargs...)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    eps = sampler.hyperparameters.eps
    varE_wanted = sett.varE_wanted

    samples = []
    for i in 1:sett.tune_samples
        init, sample = Step(sampler, target, init)
        push!(samples, sample)
    end

    samples = mapreduce(permutedims, vcat, samples)
    E = samples[:, end-1]
    varE = std(E)^2 / target.d #variance per dimension
    if dialog
        println("eps: ", eps, " --> VarE/d: ", varE)
    end
    no_divergences = isfinite(varE)

    ### update the hyperparameters ###
    if no_divergences
        success = (abs(varE-varE_wanted)/varE_wanted) < 0.05
        if !success
            new_log_eps = log(sampler.hyperparameters.eps)-α*(varE/varE_wanted-1)
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

function adaptive_step(sampler::Sampler, target::Target, init;
                        sigma_xi::Float64=1.0,
                        gamma::Float64=(50-1)/(50+1), # (neff-1)/(neff+1) 
                        kwargs...)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    eps = sampler.hyperparameters.eps
    varE_wanted = sett.varE_wanted
    d = target.d
    
    step, yA, yB, max_eps = init    
    step, _ = Step(sampler, target, step)
    xx, uu, ll, gg, dEE = step
    varE = dEE^2/d
    if dialog
        println("eps: ", eps, " --> VarE/d: ", varE)
    end    
    
    no_divergences = isfinite(varE)
    if no_divergences    
        success = (abs(varE-varE_wanted)/varE_wanted) < 0.05 
        if !success
            y = yA/yB    
            xi = -log(varE/varE_wanted + 1e-8) / 6   
            w = exp(-0.5*(xi/sigma_xi)^2)
            # Kalman update the linear combinations
            yA = gamma * yA + w * (xi + y) 
            yB = gamma * yB + w
            new_log_eps = yA/yB

            if exp(new_log_eps) > max_eps
                sampler.hyperparameters.eps = max_eps
            else
                sampler.hyperparameters.eps = exp(new_log_eps)        
            end
        else
            @info string("Found eps: ", sampler.hyperparameters.eps, " ✅")
        end
    else
        success = false    
        max_eps = sampler.hyperparameters.eps
        sampler.hyperparameters.eps = 0.5 * eps    
    end
        
    return success, (step, yA, yB, max_eps)    
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

function tune_hyperparameters(sampler::Sampler, target::Target, init;
                              burn_in::Int=0, kwargs...)
    ### debugging tool ###
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings    

    tune_sigma, tune_eps, tune_L = tune_what(sampler, target)
    
    if burn_in > 0   
        @info "Starting burn in ⏳" 
        loss, init, sample = Init_burnin(sampler, target, init; kwargs...)
        samples = []        
        for i in 1:burn_in
            lloss, step, sample = Step_burnin(sampler, target, init; kwargs...)        
            if lloss < loss
                if dialog
                    println("Virial loss: ", lloss, " --> Relative improvement: ", abs(lloss/loss - 1))
                end    
                push!(samples, sample)        
                if (lloss <= sampler.settings.loss_wanted) || (abs(lloss/loss - 1) < 0.01)
                    @info string("Virial loss condition met during burn-in at step: ", i)
                    break
                end
                loss = lloss
                init = step
            else      
                x, u, l, g, dE = init         
                uu = Partially_refresh_momentum(sampler, target, u)           
                init = (x, uu, l, g, dE)          
            end        
            if i == burn_in
                @warn "Maximum number of steps reached during burn-in"
            end         
        end
        if tune_sigma
            sigma = std(samples)[1:end-2]
            sigma ./= sqrt(sum(sigma.^2))        
            sampler.hyperparameters.sigma = sigma
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
        if tuning_method=="AdaptiveStep"
            yB = 0.1
            yA = yB * log(sampler.hyperparameters.eps)    
            tuning_init = (init, yA, yB, Inf)    
            for i in 1:sett.tune_maxiter
               success, tuning_init = adaptive_step(sampler, target, tuning_init; kwargs...)
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
    @info string("Final nu ", sampler.hyperparameters.nu)        
     
    return init, sample
end
