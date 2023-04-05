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
    end

    tune_nu!(sampler, target)
    
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

function Virial_loss(x::AbstractVector, g::AbstractVector)
"""loss^2 = (1/d) sum_i (virial_i - 1)^2"""

    #should be all close to 1 if we have reached the typical set
    v = mean(x .* g)  # mean over params
    return sqrt.((v .- 1.0).^2)
end

function Step_burnin(sampler::Sampler, target::Target, init::State; kwargs...)
    dialog = get(kwargs, :dialog, false)
    step = Step(sampler, target, init)
    lloss = Virial_loss(step.x, step.g)    
    return lloss, step
end

function Init_burnin(sampler::Sampler, target::Target,
                     init::State; kwargs...)
    dialog = get(kwargs, :dialog, false)
    x = init.x
    g = init.g
    v = mean(x .* g, dims=1)
    loss = mean((1 .- v).^2)

    if dialog
        println("Initial Virial loss: ", loss)
    end

    return  loss, init
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

function tune_hyperparameters(sampler::Sampler, target::Target, state::State;
                              progress=true, kwargs...)
    ### debugging tool ###
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings  
    
    # Tuning
    tune_sigma, tune_eps, tune_L = tune_what(sampler, target)
    nadapt = sampler.settings.nadapt
    
    # Tune eps
    if tune_eps
        for i in 1:nadapt
            state = Step(sampler, target, state; gamma=1.0, adaptive=true, kwargs...)
        end
    end
    # Tune L
    if tune_L || tune_sigma    
        xs = state.x[:]      
        @showprogress "MCHMC (tuning): " (progress ? 1 : Inf) for i in 2:nadapt
            state = Step(sampler, target, state; gamma=(50-1)/(50+1), adaptive=true, kwargs...)   
            xs = [xs state.x[:]]
            if mod(i, Int(nadapt/5))==0
                if dialog
                    println(string("Burn in step: ", i))
                    println(string("eps --->" , sampler.hyperparameters.eps))
                end            
                sigma = vec(std(xs, dims=1))
                if tune_sigma
                    sampler.hyperparameters.sigma = sigma
                end
                if tune_L
                    sampler.hyperparameters.L = mean(sigma) * sampler.hyperparameters.eps
                    if dialog
                        println(string("L   --->" , sampler.hyperparameters.L))
                        println(" ")        
                    end    
                end
            end
        end            
    end    
    
    @info string("eps: ", sampler.hyperparameters.eps)
    @info string("L: ", sampler.hyperparameters.L) 
    @info string("nu: ", sampler.hyperparameters.nu)
    @info string("sigma: ", sampler.hyperparameters.sigma)
    @info string("adaptive: ", sampler.settings.adaptive)         
    # ====

    return state
end
