
struct mutable Settings
    key::MersenneTwister
    eps::Float64
    L::Float64
    nu::Float64
    lambda_c::Float64
    tune_varE_wanted::Float64
    tune_burn_in::Float64
    tune_samples::Float64
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
    tune_burn_in = get(kwargs, :tube_burn_in, 2000)
    tune_samples = get(kwargs, :tue_samples, 1000)
    integrator = get(kwargs, :integrator, "LF")
    sett = Settings(key,
                    eps, L, nu, lambda_c,
                    tune_varE_wanted, turne_burn_in, tune_samples,
                    integrator)
end

struct Sampler
    settings::Settings
    hamiltonian_dynamics::Function
end

function Sampler(;kwargs...)

    sett = settings(kwargs)

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

function Random_unit_vector(sampler::Sampler, target::Target)
    """Generates a random (isotropic) unit vector."""
    key = sampler.settings.key
    u = randn(key, target.d)
    u ./=  sqrt.(sum(u.^2))
    return  u

end

function Partially_refresh_momentum(sampler::Sampler, target::Target, u)
    """Adds a small noise to u and normalizes."""
    sett = sampler.settings
    key = sett.key

    z = sett.nu .* randn(key, target.d)
    uu = (u .+ z) / sqrt.(sum((u .+ z).^2))
    return uu
end

function Update_momentum(sampler::Sampler, target::Target, eff_eps, g, u)
    # TO DO: type inputs
    # Have to figure out where and when to define target
    """The momentum updating map of the ESH dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""

    g_norm = sqrt.(sum(g .^2 ))
    e = - g ./ g_norm
    ue = dot(u, e)
    sh = sinh.(eff_eps * g_norm ./ target.d)
    ch = cosh.(eff_eps * g_norm ./ target.d)

    return @.((u + e * (sh + ue * (ch - 1))) / (ch + ue * sh))
end

function Dynamics(sampler::Sampler, target::Target, state)
    """One step of the Langevin-like dynamics."""

    x, u, g, time = state

    # Hamiltonian step
    xx, gg, uu = sampler.hamiltonian_dynamics(sampler, x, g, u)

    # add noise to the momentum direction
    uuu = Partially_refresh_momentum(sampler, target, uu)

    return xx, uuu, gg, 0.0
end

function Get_initial_conditions(sampler::Sampler, target::Target)
    sett = sampler.settings
    ### initial conditions ###
    x = get(kwargs, :initial_x, target.prior_draw(sett.key))
    g = target.grad_nlogp(x) .* target.d ./ (target.d - 1)
    u = Random_unit_vector(sampler) #random initial direction
    #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

    return [x, u, g, 0.0]
end

function _set_hyperparameters(sampler::Sampler, target::Target)
    satt = sampler.sett
    eps = sett.eps
    L = sett.L
    if [eps, L] == [0.0, 0.0]
        prinln("Self-tuning hyperparameters")
        eps, L = tune_hyperparameters(sett, target)
    end
    nu = sqrt((exp(2 * eps / L) - 1.0) / target.d)

    sett.eps = eps
    sett.L = L
    sett.nu = nu
end

function Step(sampler::Sampler, target::Target, state)
    """Tracks transform(x) as a function of number of iterations"""

    x, u, g, time = Dynamics(sampler, target, state)

    return [x, u, g, time], target.transform(x)
end

function Sample(sampler::Sampler, target::Target; kwargs...)
    """Args:
           num_steps: number of integration steps to take.
           x_initial: initial condition for x (an array of shape (target dimension, )). It can also be 'prior' in which case it is drawn from the prior distribution (self.Target.prior_draw).
           random_key: jax radnom seed, e.g. jax.random.PRNGKey(42).
        Returns:
            samples (shape = (num_steps, self.Target.d))
    """

    init = Get_initial_conditions(sampler, target)
    x, u, g, time = init

    _set_hyperparameters(init, sampler, target; kwargs...)

    samples = zeros(eltype(x), length(x), kwargs[:num_steps])
    samples = Vector{eltype(samples)}[eachcol(samples)...]
    for i in 1:kwargs[:num_steps]
        init, sample = Step(sampler, target, init)
        samples[i] = sample
    end

    return samples
end