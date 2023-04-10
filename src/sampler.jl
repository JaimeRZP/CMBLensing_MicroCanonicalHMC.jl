mutable struct Hyperparameters
    eps::Float64
    L::Float64
    nu::Float64
    lambda_c::Float64
    sigma
    gamma::Float64
    sigma_xi::Float64
end

Hyperparameters(;kwargs...) = begin
   eps = get(kwargs, :eps, 0.0)
   L = get(kwargs, :L, 0.0)
   nu = get(kwargs, :nu, 0.0)
   sigma = get(kwargs, :sigma, [0.0])
   lambda_c = get(kwargs, :lambda_c, 0.1931833275037836) 
   gamma = get(kwargs, :gamma, (50-1)/(50+1)) #(neff-1)/(neff+1) 
   sigma_xi = get(kwargs, :sigma_xi, 1.5)
   Hyperparameters(eps, L, nu, lambda_c, sigma, gamma, sigma_xi)
end

mutable struct Settings
    nadapt::Int
    TEV::Float64
    nchains::Int
    adaptive::Bool
    integrator::String
    init_eps
    init_L
    init_sigma
end

Settings(;kwargs...) = begin
    kwargs = Dict(kwargs)
    nadapt = get(kwargs, :nadapt, 1000)
    TEV = get(kwargs, :TEV, 0.001)
    adaptive = get(kwargs, :adaptive, false)
    nchains = get(kwargs, :nchains, 1)
    integrator = get(kwargs, :integrator, "LF")
    init_eps = get(kwargs, :init_eps, nothing)
    init_L = get(kwargs, :init_L, nothing)
    init_sigma = get(kwargs, :init_sigma, nothing)
    Settings(nadapt, TEV,  nchains, adaptive, integrator,
             init_eps, init_L, init_sigma)
end

struct Sampler <: AbstractMCMC.AbstractSampler
   settings::Settings
   hyperparameters::Hyperparameters
   hamiltonian_dynamics::Function
end

function MCHMC(nadapt, TEV; kwargs...)
   """the MCHMC (q = 0 Hamiltonian) sampler"""
   sett = Settings(;nadapt=nadapt, TEV=TEV, kwargs...)
   hyperparameters = Hyperparameters(;kwargs...)

   ### integrator ###
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

function Random_unit_vector(rng::MersenneTwister, d::Int)
    """Generates a random (isotropic) unit vector."""    
    u = randn(rng, d)
    u ./= sqrt(sum(u.^2))
    return u
end

function Partially_refresh_momentum(sampler::Sampler, target::Target, u::AbstractVector)
    """Adds a small noise to u and normalizes."""
    return Partially_refresh_momentum(target.rng,
                                      sampler.hyperparameters.nu,
                                      target.d,
                                      u)
end

function Partially_refresh_momentum(rng::MersenneTwister, nu::Float64, d::Int, u::AbstractVector)
    z = nu .* Random_unit_vector(rng, d)
    uu = (u .+ z) ./ sqrt(sum((u .+ z).^2))
    return uu
end

function Update_momentum(d::Number, eff_eps::Number,
                         g::AbstractVector, u::AbstractVector)
    """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
    similar to the implementation: https://github.com/gregversteeg/esh_dynamics
    There are no exponentials e^delta, which prevents overflows when the gradient norm is large."""   
    g_norm = sqrt(sum(g.^2))
    e = - g ./ g_norm
    delta = eff_eps * g_norm / (d-1)
    ue = dot(u, e)    
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
    
struct State
    x
    u
    l
    g
    dE::Float64
    Feps::Float64
    Weps::Float64     
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
    u = Random_unit_vector(sampler, target)
    Weps = 1e-5
    Feps = Weps * sampler.hyperparameters.eps^(1/6) 
    return State(x, u, l, g, 0.0, Feps, Weps)
end 

function Step(sampler::Sampler, target::Target, state::State; kwargs...)
    """One step of the Langevin-like dynamics."""
    dialog = get(kwargs, :dialog, false)    
        
    eps = sampler.hyperparameters.eps  
    sigma_xi = sampler.hyperparameters.sigma_xi
    gamma = get(kwargs, :gamma, sampler.hyperparameters.gamma)
        
    TEV = sampler.settings.TEV
    adaptive = get(kwargs, :adaptive, sampler.settings.adaptive)
        
    d = target.d   
        
    # Hamiltonian step
    xx, uu, ll, gg, kinetic_change = sampler.hamiltonian_dynamics(sampler, target, state)
    # add noise to the momentum direction
    uuu = Partially_refresh_momentum(sampler, target, uu)   
    dEE = kinetic_change + ll - state.l
    
    if adaptive    
        varE = dEE^2/d             
        # 1e-8 is added to avoid divergences in log xi  
        xi = varE/TEV + 1e-8
        # the weight which reduces the impact of stepsizes which 
        # are much larger on much smaller than the desired one.   
        w = exp(-0.5*(log(xi)/(6.0 * sigma_xi))^2)
        # Kalman update the linear combinations
        Feps = gamma * state.Feps + w * (xi/eps^6)  
        Weps = gamma * state.Weps + w
        new_eps = (Feps/Weps)^(-1/6)

        sampler.hyperparameters.eps = new_eps
        tune_nu!(sampler, target)
    else
        Feps = state.Feps
        Weps = state.Weps    
    end
        
    return State(xx, uuu, ll, gg, dEE, Feps, Weps)   
end
    
function _make_sample(sampler::Sampler, target::Target, state::State)
    return [target.inv_transform(state.x)[:]; state.g[:]; sampler.hyperparameters.eps; state.dE; -state.l]
end        
    

"""
    $(TYPEDSIGNATURES)

Sample from the target distribution using the provided sampler.

Keyword arguments:
* `file_name` — if provided, save chain to disk (in HDF5 format)
* `file_chunk` — write to disk only once every `file_chunk` steps
  (default: 10)
 
Returns: a vector of samples
"""        
function Sample(sampler::Sampler, target::Target, num_steps::Int;
                file_name=nothing, file_chunk=10, progress=true, kwargs...)

    state = Init(sampler, target; kwargs...)
    state = tune_hyperparameters(sampler, target, state; progress, kwargs...)
            
    sample = _make_sample(sampler, target, state)      
    samples = similar(sample, (length(sample), num_steps))

    pbar = Progress(num_steps, (progress ? 0 : Inf), "MCHMC: ")

    write_chain(file_name, size(samples)..., eltype(sample), file_chunk) do chain_file
        for i in 1:num_steps
            state = Step(sampler, target, state; kwargs...)
            samples[:,i] = sample = _make_sample(sampler, target, state)
            if chain_file !== nothing
                push!(chain_file, sample)
            end
            ProgressMeter.next!(pbar, showvalues = [
                ("ϵ", sampler.hyperparameters.eps),
                ("dE/d", state.dE / target.d)
            ])
        end
    end

    ProgressMeter.finish!(pbar)

    return samples
end