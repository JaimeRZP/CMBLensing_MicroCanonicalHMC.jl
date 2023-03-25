
struct EnsembleSampler <: AbstractMCMC.AbstractSampler
   settings::Settings
   hyperparameters::Hyperparameters
   hamiltonian_dynamics::Function
end

function MCHMC(eps, L, nchains; kwargs...)

   sett = Settings(;nchains=nchains, kwargs...)
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

   return EnsembleSampler(sett, hyperparameters, hamiltonian_dynamics)
end

function MCHMC(nchains; kwargs...)
    return MCHMC(0.0, 0.0, nchains; kwargs...)
end

function Random_unit_vector(sampler::EnsembleSampler, target::ParallelTarget;)
    """Generates a random (isotropic) unit vector."""
    return Random_unit_matrix(sampler.settings.nchains, target.target.d)
end

function Random_unit_matrix(nchains, d)
    u = randn(nchains, d)
    u ./= sqrt.(sum(u.^2, dims=2))
    return u
end

function Partially_refresh_momentum(sampler::EnsembleSampler, target::ParallelTarget, u)
    """Adds a small noise to u and normalizes."""
    return Partially_refresh_momentum(sampler.hyperparameters.nu,
                                      sampler.settings.nchains,
                                      target.target.d,
                                      u)
end

function Partially_refresh_momentum(nu, nchains, d, u)
    z = nu .* Random_unit_matrix(nchains, d)
    uu = (u .+ z) ./ sqrt.(sum((u .+ z).^2, dims=2))
    return uu
end

function Update_momentum(target::ParallelTarget, eff_eps::Number,
                         g::AbstractMatrix, u::AbstractMatrix)
    # TO DO: type inputs
    # Have to figure out where and when to define target
    """The momentum updating map of the ESH dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""
    Update_momentum(target.target.d, eff_eps::Number, g ,u)
end

function Update_momentum(d::Number, eff_eps::Number,
                         g::AbstractMatrix ,u::AbstractMatrix)
    g_norm = sqrt.(sum(g .^2, dims=2))[:, 1]
    e = - g ./ g_norm
    delta = eff_eps .* g_norm ./ (d-1)    
    # Matrix dot product along dim=2
    ue = sum(u .* e, dims=2)[:, 1]
    
    #=    
    sh = sinh.(delta)
    ch = cosh.(delta)
    th = tanh.(delta)
    uu = @.((u + e * (sh + ue * (ch - 1))) / (ch + ue * sh))
    uu ./= sqrt.(sum(uu.^2, dims=2))   
    delta_r = log.(ch) .+ log1p.(ue .* th)
    =#    
    
    zeta = exp.(-delta)
    uu =  @.(e * ((1-zeta) * (1 + zeta + ue * (1-zeta))) + (2 * zeta) * u)
    uu ./= sqrt.(sum(uu.^2, dims=2))
    delta_r = delta .- log(2) .+ log.(1 .+ ue .+ (1 .- ue) .* zeta.^2)   
    
    return uu, delta_r
end

function Dynamics(sampler::EnsembleSampler, target::ParallelTarget, state)
    """One step of the Langevin-like dynamics."""
    x, u, l, g, dE = state
    # Hamiltonian step
    xx, uu, ll, gg, kinetic_change = sampler.hamiltonian_dynamics(sampler, target, x, u, l, g)
    # add noise to the momentum direction
    uuu = Partially_refresh_momentum(sampler, target, uu)
    dEE = kinetic_change .+ ll .- l
    return xx, uuu, ll, gg, dEE
end

function Init(sampler::EnsembleSampler, target::ParallelTarget; kwargs...)
    sett = sampler.settings
    kwargs = Dict(kwargs)
    d = target.target.d
    ### initial conditions ###
    if :initial_x ∈ keys(kwargs)
        @info "Using provided initial point"    
        x = target.transform(kwargs[:initial_x])
    else
        x = target.prior_draw()
    end
    l, g = target.nlogp_grad_nlogp(x)
    g .*= d/(d-1)
    u = Random_unit_vector(sampler, target) #random initial direction

    dE = zeros(sett.nchains)
    sample = (target.inv_transform(x), dE, -l)
    state = (x, u, l, g, dE)
    return state, sample
end

function Step(sampler::EnsembleSampler, target::ParallelTarget, state; kwargs...)
    """Tracks transform(x) as a function of number of iterations"""
    step = Dynamics(sampler, target, state)
    xx, uu, ll, gg, dEE = step

    return step, (target.inv_transform(xx), dEE, -ll)
end


function Sample(sampler::EnsembleSampler, target::Target, num_steps::Int;
                burn_in::Int=0, fol_name=".", file_name="samples", progress=true, kwargs...)
    """Args:
           num_steps: number of integration steps to take.
           x_initial: initial condition for x (an array of shape (target dimension, )). It can also be 'prior' in which case it is drawn from the prior distribution (self.Target.prior_draw).
           random_key: jax radnom seed, e.g. jax.random.PRNGKey(42).
        Returns:
            samples (shape = (num_steps, self.Target.d))
    """
    
    nchains = sampler.settings.nchains
    target = ParallelTarget(target, nchains)

    io = open(joinpath(fol_name, "VarNames.txt"), "w") do io
        println(io, string(target.target.vsyms))
    end       

    state, sample = Init(sampler, target; kwargs...)
    state, sample = tune_hyperparameters(sampler, target, state, sample;
                                         burn_in=burn_in, kwargs...)

    chains = zeros(num_steps, nchains, target.target.d+2)
    X, E, L = sample
    chains[1, :, :] = [X E L]
            
    io = open(joinpath(fol_name, string(file_name, ".txt")), "w") do io
        println(io, [X E L])
        for i in 1:num_steps
            try    
                state, sample = Step(sampler, target, state; kwargs...)
                X, E, L = sample
                chains[i, :, :] = [X E L]    
                println(io, [X E L])
            catch
                @warn "Divergence encountered after tuning"
            end        
        end
    end
    

    io = open(joinpath(fol_name, string(file_name, "_summary.txt")), "w") do io
        ess, rhat = Summarize(chains)
        println(io, ess)
        println(io, rhat)
    end
    
    return unroll_chains(chains)
end

function unroll_chains(chains)
    nsteps, nchains, nparams = axes(chains)
    chain = []
    for i in nchains
        A = chains[:, i, :]
        chain_i = Vector{eltype(A)}[eachrow(A)...]
        chain = cat(chain, chain_i; dims=1)
    end
    return chain
end