mutable struct Hyperparameters
    eps::Float64
    L::Float64
    nu::Float64
    lambda_c::Float64
    sigma::AbstractVector
    gamma::Float64
    sigma_xi::Float64
end

Hyperparameters(;kwargs...) = begin
   eps = get(kwargs, :eps, 0.0)
   L = get(kwargs, :L, 0.0)
   nu = get(kwargs, :nu, 0.0)
   lambda_c = get(kwargs, :lambda_c, 0.1931833275037836)
   sigma = get(kwargs, :sigma, [0.0])
   gamma = get(kwargs, :gamma, (50-1)/(50+1)) #(neff-1)/(neff+1) 
   sigma_xi = get(kwargs, :sigma_xi, 1.0)
   Hyperparameters(eps, L, nu, lambda_c, sigma, gamma, sigma_xi)
end

mutable struct Settings
    nchains::Int
    integrator::String
    loss_wanted::Float64
    varE_wanted::Float64
    tune_eps_nsteps::Int
    tune_L_nsteps::Int
    init_eps
    init_L
    init_sigma
end

Settings(;kwargs...) = begin
    kwargs = Dict(kwargs)
    
    nchains = get(kwargs, :nchains, 1)
    integrator = get(kwargs, :integrator, "LF")
    loss_wanted = get(kwargs, :loss_wanted, 1.0)
    varE_wanted = get(kwargs, :varE_wanted, 0.001)
    tune_eps_nsteps = get(kwargs, :tune_eps_nsteps, 100)
    tune_L_nsteps = get(kwargs, :tune_L_nsteps, 2500)
    init_eps = get(kwargs, :init_eps, nothing)
    init_L = get(kwargs, :init_L, nothing)
    init_sigma = get(kwargs, :init_sigma, nothing)
    Settings(nchains, integrator,
             loss_wanted, varE_wanted, tune_eps_nsteps, tune_L_nsteps,
             init_eps, init_L, init_sigma)
end

struct Sampler <: AbstractMCMC.AbstractSampler
   settings::Settings
   hyperparameters::Hyperparameters
   hamiltonian_dynamics::Function
end

function MCHMC(eps, L; kwargs...)

   sett = Settings(;kwargs...)
   hyperparameters = Hyperparameters(;eps=eps, L=L, kwargs...)

   if sett.integrator == "LF"  # leapfrog
       hamiltonian_dynamics = Leapfrog
       grad_evals_per_step = 1.0
   elseif sett.integrator == "MN"  # minimal norm integrator
       hamiltonian_dynamics = Minimal_norm
       grad_evals_per_step = 2.0
   else
       println(string("integrator = ", integrator, "is not a valid option."))
   end

   return Sampler(sett, hyperparameters, hamiltonian_dynamics)
end

function MCHMC(; kwargs...)
    return MCHMC(0.0, 0.0; kwargs...)
end

function Random_unit_vector(d)
    """Generates a random (isotropic) unit vector."""    
    u = randn(d)
    u ./= sqrt(sum(u.^2))
    return u
end

function Partially_refresh_momentum(sampler::Sampler, target::Target, u::AbstractVector)
    """Adds a small noise to u and normalizes."""
    return Partially_refresh_momentum(sampler.hyperparameters.nu,
                                      target.d,
                                      u)
end

function Partially_refresh_momentum(nu, d, u::AbstractVector)
    z = nu .* Random_unit_vector(d)
    uu = (u .+ z) ./ sqrt(sum((u .+ z).^2))
    return uu
end

function Update_momentum(d::Number, eff_eps::Number,
                         g::AbstractVector, u::AbstractVector)
    """The momentum updating map of the ESH dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""    
    g_norm = sqrt(sum(g.^2))
    e = - g ./ g_norm
    delta = eff_eps * g_norm / (d-1)
    ue = dot(u, e)    
        
    #=
    sh = sinh(delta)
    ch = cosh(delta)
    th = tanh(delta)
    uu = (u .+ e .* (sh + ue * (ch - 1))) / (ch + ue * sh)
    uu ./= sqrt(sum(uu.^2))
    delta_r = log(ch) + log1p(ue * th)
    =#

    zeta = exp(-delta)
    uu = e .* ((1-zeta) * (1 + zeta + ue * (1-zeta))) + (2 * zeta) .* u
    uu ./= sqrt(sum(uu.^2))
            
    delta_r = delta - log(2) + log(1 + ue + (1-ue) * zeta^2)  
    return uu, delta_r
end

function Energy(target::Target,
                x::AbstractVector, xx::AbstractVector,
                E::Number, kinetic_change::Number)
    l = target.nlogp(x)
    ll = target.nlogp(xx)
    dE = kinetic_change + ll - l
    EE = E + dE
    return -nllogp, EE
end
    
struct State{T} <: AbstractState
    x::Vector{T}
    u::Vector{T}
    l::T
    g::Vector{T}
    dE::T    
end 
    
 
struct AdaptiveState{T} <: AbstractState
    x::Vector{T}
    u::Vector{T}
    l::T
    g::Vector{T}
    dE::T
    Feps::T
    Weps::T 
end     

function Init(sampler::Sampler, target::Target; kwargs...)
    sett = sampler.settings
    kwargs = Dict(kwargs)
    d = target.d
    ### initial conditions ###
    if :initial_x ∈ keys(kwargs)
        x = target.transform(kwargs[:initial_x])
    else
        x = target.prior_draw()
    end
    l, g = target.nlogp_grad_nlogp(x)
    g .*= d/(d-1)
    u = Random_unit_vector(d) #random initial direction

    return State(x, u, l, g, 0.0)
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

function Step(sampler::Sampler, target::Target, state::AbstractState; kwargs...)
    """One step of the Langevin-like dynamics."""
    # Hamiltonian step
    xx, uu, ll, gg, kinetic_change = sampler.hamiltonian_dynamics(sampler, target, state)
    # add noise to the momentum direction
    uuu = Partially_refresh_momentum(sampler, target, uu)   
    dEE = kinetic_change + ll - state.l
    return State(xx, uuu, ll, gg, dEE)
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
    
function Sample(sampler::Sampler, target::Target, num_steps::Int;
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

    state = Init(sampler, target; kwargs...)
    state = tune_hyperparameters(sampler, target, state;
                                 burn_in=burn_in, kwargs...)
            
    samples = []
    sample = [target.inv_transform(state.x); state.dE; -state.l]        
    push!(samples, sample)
            
    io = open(joinpath(fol_name, string(file_name, ".txt")), "w") do io
        println(io, sample)
        for i in 1:num_steps-1
            try    
                state = Step(sampler, target, state; kwargs...)
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