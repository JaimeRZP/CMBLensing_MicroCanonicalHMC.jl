
struct Settings
    key::MersenneTwister
    lambda_c::Float64
    eps::Float64
    L::Float64
    integrator::String
end

function Settings(eps, L; kwargs...)
    seed = get(kwargs, :seed, 0)
    key = MersenneTwister(seed)
    lambda_c = get(kwargs, :lambda_c, 0.1931833275037836)
    integrator = get(kwargs, :integrator, "LF")
    sett = Settings(key, lambda_c, eps, L, integrator)
end

struct Hyperparameters
    L::Float64
    eps::Float64
    nu::Float64
end

function Hyperparameters(sett::Settings, target::Target)
    nu = sqrt((exp(2 * sett.eps / sett.L) - 1.0) / target.d)
    return Hyperparameters(sett.L, sett.eps, nu)
end

struct Sampler
    #TO DO: what types are these?
    settings::Settings
    target::Target
    hamiltonian_dynamics
    hyperparameters::Hyperparameters
end

function Sampler(sett::Settings, target::Target)

    if sett.integrator == "LF"  # leapfrog
        hamiltonian_dynamics = Leapfrog
        grad_evals_per_step = 1.0
    elseif sett.integrator == "MN"  # minimal norm integrator
        hamiltonian_dynamics = Minimal_norm
        grad_evals_per_step = 2.0
    else
        println(string("integrator = ", integrator, "is not a valid option."))
    end

    hyperparams = Hyperparameters(sett, target)

    return Sampler(sett, target, hamiltonian_dynamics, hyperparams)
end

function Sampler(target::Target)
    eps=5.0
    L=sqrt.(target.d)
    sett = Settings(eps, L)
    return Sampler(sett, target)
end

function Random_unit_vector(sampler::Sampler)
    """Generates a random (isotropic) unit vector."""
    key = sampler.settings.key
    u = randn(key, sampler.target.d)
    u ./=  sqrt.(sum(u.^2))
    return  u

end

function Partially_refresh_momentum(sampler::Sampler, u)
    """Adds a small noise to u and normalizes."""
    hyperparams = sampler.hyperparameters
    target = sampler.target
    key = sampler.settings.key

    z = hyperparams.nu .* randn(key, sampler.target.d)
    uu = @.((u + z) / sqrt(sum((u + z)^2)))
    return uu
end

function Update_momentum(sampler::Sampler, eff_eps, g, u)
    # TO DO: type inputs
    # Have to figure out where and when to define target
    """The momentum updating map of the ESH dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""

    g_norm = sqrt.(sum(g.^2))
    e = - g ./ g_norm
    ue = dot(u, e)
    sh = sinh.(eff_eps .* g_norm ./ sampler.target.d)
    ch = cosh.(eff_eps .* g_norm ./ sampler.target.d)

    return @.(u + e * (sh + ue * (ch - 1)) / (ch + ue * sh))
end

function Dynamics(sampler::Sampler, state)
    """One step of the Langevin-like dynamics."""

    x, u, g, time = state

    # Hamiltonian step
    xx, gg, uu = sampler.hamiltonian_dynamics(sampler, x, g, u)

    # add noise to the momentum direction
    uuu = Partially_refresh_momentum(sampler, uu)

    return xx, uuu, gg, 0.0
end

function Get_initial_conditions(sampler::Sampler; kwargs...)
    kwargs = Dict(kwargs)
    target = sampler.target
    sett = sampler.settings
    ### initial conditions ###
    x = get(kwargs, :initial_x, target.prior_draw(sett.key))
    g = target.grad_nlogp(x) .* target.d ./ (target.d - 1)
    u = Random_unit_vector(sampler) #random initial direction
    #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

    return [x, u, g, 0.0]
end

function Step(sampler::Sampler, state)
    """Tracks transform(x) as a function of number of iterations"""

    x, u, g, time = Dynamics(sampler, state)

    return [x, u, g, time], sampler.target.transform(x)
end

function Sample(sampler::Sampler; kwargs...)
    """Args:
               num_steps: number of integration steps to take.
               x_initial: initial condition for x (an array of shape (target dimension, )). It can also be 'prior' in which case it is drawn from the prior distribution (self.Target.prior_draw).
               random_key: jax radnom seed, e.g. jax.random.PRNGKey(42).
            Returns:
                samples (shape = (num_steps, self.Target.d))
    """
    init = Get_initial_conditions(sampler; kwargs...)
    x, u, g, time = init
    samples = zeros(eltype(x), length(x), kwargs[:num_steps])
    samples = Vector{eltype(samples)}[eachcol(samples)...]
    for i in 1:kwargs[:num_steps]
        init, sample = Step(sampler, init)
        samples[i] = sample
    end

    return samples
end