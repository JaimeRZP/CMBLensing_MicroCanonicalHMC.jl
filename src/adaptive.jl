struct AdaptiveState{T} <: AbstractState
    x::Vector{T}
    u::Vector{T}
    l::T
    g::Vector{T}
    dE::T
    Feps::T
    Weps::T 
end    

function AdaptiveInit(state::State, eps::Float64)
    Weps = 1e-5
    Feps = Weps * eps^(1/6)    
    return AdaptiveState(state.x, state.u, state.l, state.g, state.dE, Weps, Feps)
end     
  
function AdaptiveInit(sampler::Sampler, target::Target; kwargs...)
    state = Init(sampler, target; kwargs...)   
    return AdaptiveInit(state, sampler.hyperparameters.eps)
end   

function AdaptiveStep(sampler::Sampler, target::Target, state::AdaptiveState; kwargs...)
    """One step of the Langevin-like dynamics."""
    dialog = get(kwargs, :dialog, false)    
    sett = sampler.settings    
    eps = sampler.hyperparameters.eps  
    sigma_xi = sampler.hyperparameters.sigma_xi
    gamma = sampler.hyperparameters.gamma    
    varE_wanted = sett.varE_wanted
    d = target.d    
        
    step = Step(sampler, target, state; kwargs...)
    varE = step.dE^2/d
     
    if dialog
        println("eps: ", eps, " --> VarE/d: ", varE)
    end             
        
    xi = varE/varE_wanted + 1e-8   
    w = exp(-0.5*(log(xi)/(6.0 * sigma_xi))^2)
    # Kalman update the linear combinations
    Feps = gamma * state.Feps + w * (xi/eps^6)  
    Weps = gamma * state.Weps + w
    new_eps = (Feps/Weps)^(-1/6)

    sampler.hyperparameters.eps = new_eps
    
    return AdaptiveState(step.x, step.u, step.l, step.g, step.dE, Feps, Weps)
end  

function AdaptiveSample(sampler::Sampler, target::Target, num_steps::Int;
                burn_in::Int=0, fol_name=".", file_name="samples", progress=true, kwargs...)
    """Args:
           num_steps: number of integration steps to take.
           x_initial: initial condition for x (an array of shape (target dimension, )).
                      It can also be 'prior' in which case it is drawn from the prior distribution (self.Target.prior_draw).
           random_key: jax radnom seed, e.g. jax.random.PRNGKey(42).
        Returns:
            samples (shape = (num_steps, self.Target.d))
    """
    
    io = open(joinpath(fol_name, "VarNames.txt"), "w") do io
        println(io, string(target.vsyms))
    end        
       
    state = AdaptiveInit(sampler, target; kwargs...)
        
    # ======
    # Start tuning
        
    # Tune eps   
    sampler.hyperparameters.gamma = 1.0    
    for i in 1:Int(round(num_steps/20))
        state = AdaptiveStep(sampler, target, state; kwargs...)
    end
  
    
    # Tune L
    xs = []
    push!=(xs, state.x)     
    for i in 1:Int(round(num_steps/20))
                state = AdaptiveStep(sampler, target, state; kwargs...)   
                push!(xs, state.x)                
    end            
    sigma = mean(std(xs, dims=1))
    sampler.hyperparameters.L = sigma * sampler.hyperparameters.eps
    
    @info string("Found eps: ", sampler.hyperparameters.eps)
    @info string("Found L: ", sampler.hyperparameters.L)     
    # Finish tuning
    # =====    
        
        
    samples = []
    sample = [target.inv_transform(state.x); state.dE; -state.l]        
    push!(samples, sample)
            
    io = open(joinpath(fol_name, string(file_name, ".txt")), "w") do io
        println(io, sample)
        for i in 1:num_steps-1
            try    
                state = AdaptiveStep(sampler, target, state; kwargs...)
                sample = [target.inv_transform(state.x); state.dE; -state.l]    
                push!(samples, sample)
                println(io, sample)
            catch
                @warn "Divergence encountered after tuning"
            end        
        end
    end
            
    io = open(joinpath(fol_name, string(file_name, "_summary.txt")), "w") do io
        ess, rhat = Summarize(samples)
        println(io, ess)
        println(io, rhat)
    end         
                
    return samples
end