
mutable struct Settings
    key::MersenneTwister
    eps::Float64
    L::Float64
    nu::Float64
    lambda_c::Float64
    tune_varE_wanted::Float64
    tune_burn_in::Int
    tune_samples::Int
    tune_maxiter::Int
    integrator::String
end

function Settings(; kwargs...)
    kwargs = Dict(kwargs)
    seed = get(kwargs, :seed, 0)
    key = MersenneTwister(seed)
    eps = get(kwargs, :eps, 0.0)
    L = get(kwargs, :L, 0.0)
    nu = get(kwargs, :nu, 0.0)
    lambda_c = get(kwargs, :lambda_c, 0.1931833275037836)
    tune_varE_wanted = get(kwargs, :tune_varE_wanted, 0.0005)
    tune_burn_in = get(kwargs, :tune_burn_in, 2000)
    tune_samples = get(kwargs, :tune_samples, 1000)
    tune_maxiter = get(kwargs, :tune_maxiter, 10)
    integrator = get(kwargs, :integrator, "LF")
    sett = Settings(key,
                    eps, L, nu, lambda_c,
                    tune_varE_wanted, tune_burn_in, tune_samples, tune_maxiter,
                    integrator)
end

struct Sampler
    settings::Settings
    hamiltonian_dynamics::Function
end

function Sampler(;kwargs...)

    sett = Settings(;kwargs...)

    if sett.integrator == "LF"  # leapfrog
        hamiltonian_dynamics = Leapfrog
        grad_evals_per_step = 1.0
    elseif sett.integrator == "MN"  # minimal norm integrator
        hamiltonian_dynamics = Minimal_norm
        grad_evals_per_step = 2.0
    else
        println(string("integrator = ", integrator, "is not a valid option."))
    end

    return Sampler(sett, hamiltonian_dynamics)
end

function Random_unit_vector(sampler::Sampler, target::Target;
                            normalize=true)
    """Generates a random (isotropic) unit vector."""
    key = sampler.settings.key
    u = randn(key, target.d)
    if normalize
        u ./=  sqrt.(sum(u.^2))
    end
    return  u

end

function Partially_refresh_momentum(sampler::Sampler, target::Target, u)
    """Adds a small noise to u and normalizes."""
    sett = sampler.settings
    key = sett.key

    z = sett.nu .* Random_unit_vector(sampler, target; normalize=false)
    uu = (u .+ z) / sqrt.(sum((u .+ z).^2))
    return uu
end

function Update_momentum(sampler::Sampler, target::Target, eff_eps::Float64, g, u, r)
    # TO DO: type inputs
    # Have to figure out where and when to define target
    """The momentum updating map of the ESH dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""

    g_norm = sqrt.(sum(g .^2 ))
    e = - g ./ g_norm
    ue = dot(u, e)
    sh = sinh.(eff_eps * g_norm ./ target.d)
    ch = cosh.(eff_eps * g_norm ./ target.d)
    th = tanh.(eff_eps * g_norm ./ target.d)
    delta_r = @.(log(ch) + log1p(ue * th))

    uu = @.((u + e * (sh + ue * (ch - 1))) / (ch + ue * sh))
    rr = r .+ delta_r
    return uu, rr
end

function Dynamics(sampler::Sampler, target::Target, state)
    """One step of the Langevin-like dynamics."""

    x, u, g, r, time = state

    # Hamiltonian step
    xx, gg, uu, rr = sampler.hamiltonian_dynamics(sampler, target, x, g, u, r)

    # add noise to the momentum direction
    uuu = Partially_refresh_momentum(sampler, target, uu)

    # why not this??
    # time += eps

    return xx, uuu, gg, rr, time
end

function Get_initial_conditions(sampler::Sampler, target::Target; kwargs...)
    sett = sampler.settings
    kwargs = Dict(kwargs)
    ### initial conditions ###
    x = get(kwargs, :initial_x, target.prior_draw(sett.key))
    g = target.grad_nlogp(x) .* target.d ./ (target.d - 1)
    u = Random_unit_vector(sampler, target) #random initial direction
    #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p
    r = 0.5 * target.d - target.nlogp(x) / (target.d-1) # initialize r such that all the chains have the same energy = d / 2
    return (x, u, g, r, 0.0)
end

function _set_hyperparameters(init, sampler::Sampler, target::Target; kwargs...)
    eps = sampler.settings.eps
    L = sampler.settings.L
    if [eps, L] == [0.0, 0.0]
        println("Self-tuning hyperparameters")
        eps, L = tune_hyperparameters(init, sampler, target; kwargs...)
    end
    nu = sqrt((exp(2 * eps / L) - 1.0) / target.d)

    sampler.settings.eps = eps
    sampler.settings.L = L
    sampler.settings.nu = nu
end

function Energy(target::Target, x, r)
    return target.d * r + target.nlogp(x)
end

function Step(sampler::Sampler, target::Target, state; kwargs...)
    """Tracks transform(x) as a function of number of iterations"""
    step = Dynamics(sampler, target, state)
    x, u, g, r, time = step
    if get(kwargs, :monitor_energy, false)
        energy = Energy(target, x, r)
    else
        energy = nothing
    end
    return step, (target.inv_transform(x), energy)
end

function Sample(sampler::Sampler, target::Target; kwargs...)
    """Args:
           num_steps: number of integration steps to take.
           x_initial: initial condition for x (an array of shape (target dimension, )). It can also be 'prior' in which case it is drawn from the prior distribution (self.Target.prior_draw).
           random_key: jax radnom seed, e.g. jax.random.PRNGKey(42).
        Returns:
            samples (shape = (num_steps, self.Target.d))
    """

    init = Get_initial_conditions(sampler, target; kwargs...)
    x, u, g, r, time = init
    energy = Energy(target, x, r)

    _set_hyperparameters(init, sampler, target; kwargs...)

    samples = DataFrame(Ω=typeof(x), Energy=typeof(energy))
    push! (samples, (target.inv_transform(x), energy))
    for i in 2:kwargs[:num_steps]+1
        init, sample = Step(sampler, target, init; kwargs...)
        push! (samples, sample)
    end

    return samples
end