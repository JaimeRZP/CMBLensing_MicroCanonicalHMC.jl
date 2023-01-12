
struct Settings
    lambda_c::Float64
    eps::Float64
    L::Float64
    integrator::String
end

struct Hyperparameters
    #TO DO: what types are these?
    L
    eps
    nu
end

function Hyperparameters(sett::Settings, target)
    nu = @.(sqrt((exp(2 * sett.eps / sett.L) - 1.0) / target.d))
    return Hyperparameters(sett.L, sett.eps, nu)
end

struct Sampler
    #TO DO: what types are these?
    key::MersenneTwister
    settings::Settings
    target::Target
    hamiltonian_dynamics::Integrator
    hyperparameters::Hyperparameters
end

function Sampler(sett::Settings, target::Target)

    if integrator == "LF"  # leapfrog
        hamiltonian_dynamics = Leapfrog
        grad_evals_per_step = 1.0
    elseif integrator == "MN"  # minimal norm integrator
        hamiltonian_dynamics = Minimal_norm
        grad_evals_per_step = 2.0
    else
        println(string("integrator = ", integrator, "is not a valid option."))
    end

    hyperparams = Hyperparameters(sett, target)

   return Sampler(sett, target, hamiltonian_dynamics, hyperparams)
end

function Random_unit_vector(sampler::Sampler)
    """Generates a random (isotropic) unit vector."""
    key = sampler.settings.key
    u = randn(key, sampler.target.d)
    return  @.(u / sqrt(sum(square(u))))

end

function Partially_refresh_momentum(sampler::Sampler, u)
    """Adds a small noise to u and normalizes."""
    hyperparams = sampler.hyperparameters
    target = sampler.target
    key = sampler.settings.key

    z = hyperparams.nu .* randn(key, sampler.target.d)
    uu = @.((u + z) / sqrt(sum(square(u + z))))
    return uu
end

function Update_momentum(sett::Settings, g, u)
    # TO DO: type inputs
    # Have to figure out where and when to define target
    """The momentum updating map of the ESH dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""
    eps = sett.eps

    g_norm = sqrt.(sum(g.^2))
    e = .- g ./ g_norm
    ue = dot(u, e)
    sh = sinh.(eps .* g_norm ./ target.d)
    ch = cosh.(eps .* g_norm ./ target.d)

    return @.(u + e * (sh + ue * (ch - 1))) / (ch + ue * sh)
end

function Dynamics(sampler::Sampler, state)
    """One step of the Langevin-like dynamics."""

    x, u, g, time = state

    # Hamiltonian step
    xx, gg, uu = sampler.hamiltonian_dynamics(x, g, u)

    # add noise to the momentum direction
    uuu = Partially_refresh_momentum(uu)

    return xx, uuu, gg, 0.0
end

function Get_initial_conditions(sampler::Sampler, kwargs...)
    kwargs = Dict(kwargs)
    target = Sampler.target
    sett = sampler.settings
    ### initial conditions ###
    x = get(kwargs, initial_x, target.prior_draw(sett.key))

    g = @.(target.grad_nlogp(x) * target.d / (target.d - 1))

    u = Random_unit_vector(sett.key) #random initial direction
    #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

    return [x, u, g, 0.0]
end

function Step(sampler::Sampler, state)
    """Tracks transform(x) as a function of number of iterations"""

    x, u, g, time = Dynamics(sampler, state)

    return [x, u, g, time], sampler.target.transform(x)
end

function Sample(sampler::Sampler, kwargs...)
    """Args:
               num_steps: number of integration steps to take.
               x_initial: initial condition for x (an array of shape (target dimension, )). It can also be 'prior' in which case it is drawn from the prior distribution (self.Target.prior_draw).
               random_key: jax radnom seed, e.g. jax.random.PRNGKey(42).
            Returns:
                samples (shape = (num_steps, self.Target.d))
    """
    init = Get_initial_conditions(sampler; kwargs...)
    samples = zeros(Vector{Vector{eltype(x)}}, kwargs[:num_steps])
    for i in 1:kwargs[:num_steps]
        init, sample = Step(init)
        samples[i] = sample
    end

    return samples
end