
struct Settings
    lambda_c::Float64
    eps::Float64
    L::Float64
    integrator::String
end

struct Target
    #TO DO: what types are these?
    prior_draw
    transform
    grap_nlogp
    d
end

struct Hyperparameters
    #TO DO: what types are these?
    L
    eps
    nu
end

function Hyperparameters(sett::Settings, target::Target)
    nu = @.(sqrt((exp(2 * sett.eps / sett.L) - 1.0) / target.d))
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

    if integrator == "LF"  # leapfrog
        hamiltonian_dynamics = Leapfrog
        grad_evals_per_step = 1.0
    elseif integrator == "MN"  # minimal norm integrator
        hamiltonian_dynamics = Minimal_norm
        grad_evals_per_step = 2.0
    else
        println('integrator = ' + integrator + 'is not a valid option.')
    end

    hyperparams = Hyperparameters(sett, target)

   return Sampler(sett, target, hamiltonian_dynamics, hyperparams)
end

function Random_unit_vector(sampler::Sampler, key):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(key)
        u = rand(subkey, shape = (self.Target.d, ), dtype = 'float64')
        u /=. @.(sqrt(sum(square(u))))
    return u, key
end

function Partially_refresh_momentum(sampler::Sampler, u, key)
    """Adds a small noise to u and normalizes."""
    hyperparams = sampler.hyperparameters
    target = sampler.target

    #key, subkey = jax.random.split(key)
    #z = hyperparams.nu * randn(subkey, shape = (target.d, ), dtype = 'float64')
    uu = @.((u + z) / sqrt(sum(square(u + z))))
    return uu, key
end

function Update_momentum(sett:Settings g, u)
    # TO DO: type inputs
    # Have to figure out where and when to define target
    """The momentum updating map of the ESH dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""
    eps = sett.eps

    g_norm = sqrt.(sum(square.(g)))
    e = -. g /. g_norm
    ue = dot(u, e)
    sh = sinh(eps * g_norm / target.d)
    ch = cosh(eps * g_norm / target.d)

    return @.(u + e * (sh + ue * (ch - 1))) / (ch + ue * sh)
end

function Dynamics(sampler::Sampler, state)
    """One step of the Langevin-like dynamics."""

    x, u, g, key, time = state

    # Hamiltonian step
    xx, gg, uu = sampler.hamiltonian_dynamics(x, g, u)

    # add noise to the momentum direction
    uuu, key = Partially_refresh_momentum(uu, key)

    return xx, uuu, gg, key, 0.0
end

function Get_initial_conditions(sampler::Sampler, kwargs...)
    kwargs = Dict(kwargs)
    target = Sampler.target
    ### random key ###
    key = get(kwargs, random_key, jax.random.PRNGKey(0))
    key, prior_key = jax.random.split(key)
    ### initial conditions ###
    x = get(kwargs, initial_x, target.prior_draw(prior_key))

    g = @.(target.grad_nlogp(x) * target.d / (target.d - 1))

    u, key = Random_unit_vector(key) #random initial direction
    #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

    return x, u, g, key

function Step(sampler::Sampler, state)
    """Tracks transform(x) as a function of number of iterations"""

    x, u, g, key, time = Dynamics(sampler, state)

    return [x, u, g, key, time, sampler.target.transform(x)]
end

function Sample(sampler::Sampler, kwargs...)
    """Args:
               num_steps: number of integration steps to take.
               x_initial: initial condition for x (an array of shape (target dimension, )). It can also be 'prior' in which case it is drawn from the prior distribution (self.Target.prior_draw).
               random_key: jax radnom seed, e.g. jax.random.PRNGKey(42).
            Returns:
                samples (shape = (num_steps, self.Target.d))
    """
    x, u, g, key = Get_initial_conditions(kwargs[:x_initial], kwargs[:x_initial])
    chain = [Step(x, u, g, key, 0.0) for i in 1:kwargs[:num_steps]]
    return chain
end