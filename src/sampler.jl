mutable struct Hyperparameters{T}
    eps::T
    L::T
    nu::T
    lambda_c::T
    sigma::AbstractVector{T}
    gamma::T
    sigma_xi::T
end

Hyperparameters(; kwargs...) = begin
    eps = get(kwargs, :eps, 0.0)
    L = get(kwargs, :L, 0.0)
    nu = get(kwargs, :nu, 0.0)
    sigma = get(kwargs, :sigma, [0.0])
    lambda_c = get(kwargs, :lambda_c, 0.1931833275037836)
    gamma = get(kwargs, :gamma, (50 - 1) / (50 + 1)) #(neff-1)/(neff+1) 
    sigma_xi = get(kwargs, :sigma_xi, 1.5)
    Hyperparameters(eps, L, nu, lambda_c, sigma, gamma, sigma_xi)
end

mutable struct Settings{T}
    nadapt::Int
    TEV::T
    nchains::Int
    adaptive::Bool
    integrator::String
    init_eps::Union{Nothing,T}
    init_L::Union{Nothing,T}
    init_sigma::Union{Nothing,AbstractVector{T}}
end

Settings(; kwargs...) = begin
    kwargs = Dict(kwargs)
    nadapt = get(kwargs, :nadapt, 1000)
    TEV = get(kwargs, :TEV, 0.001)
    adaptive = get(kwargs, :adaptive, false)
    nchains = get(kwargs, :nchains, 1)
    integrator = get(kwargs, :integrator, "LF")
    init_eps = get(kwargs, :init_eps, nothing)
    init_L = get(kwargs, :init_L, nothing)
    init_sigma = get(kwargs, :init_sigma, nothing)
    Settings(nadapt, TEV, nchains, adaptive, integrator, init_eps, init_L, init_sigma)
end

struct MCHMCSampler <: AbstractMCMC.AbstractSampler
    settings::Settings
    hyperparameters::Hyperparameters
    hamiltonian_dynamics::Function
end

function MCHMC(nadapt, TEV; kwargs...)
    """the MCHMC (q = 0 Hamiltonian) sampler"""
    sett = Settings(; nadapt = nadapt, TEV = TEV, kwargs...)
    hyperparameters = Hyperparameters(; kwargs...)

    ### integrator ###
    if sett.integrator == "LF" # leapfrog
        hamiltonian_dynamics = Leapfrog
        grad_evals_per_step = 1.0
    elseif sett.integrator == "MN" # minimal norm
        hamiltonian_dynamics = Minimal_norm
        grad_evals_per_step = 2.0
    else
        println(string("integrator = ", integrator, "is not a valid option."))
    end

    return MCHMCSampler(sett, hyperparameters, hamiltonian_dynamics)
end

function Random_unit_vector(rng::Random.AbstractRNG, target::Target; _normalize = true)
    return Random_unit_vector(rng, target.d; _normalize=_normalize)
end

function Random_unit_vector(rng::Random.AbstractRNG, d::Int; _normalize = true)
    """Generates a random (isotropic) unit vector."""
    u = randn(rng, d)
    if _normalize
        u = normalize(u)
    end
    return u
end

function Partially_refresh_momentum(
    rng::Random.AbstractRNG,
    sampler::MCHMCSampler,
    target::Target,
    u::AbstractVector,
)
    nu = sampler.hyperparameters.nu
    d = target.d
    return Partially_refresh_momentum(rng, nu, d, u)
end

function Partially_refresh_momentum(
    rng::Random.AbstractRNG,
    nu::Number,
    d::Int,
    u::AbstractVector,
)
    """Adds a small noise to u and normalizes."""
    z = nu .* Random_unit_vector(rng, d; _normalize = false)
    uu = u .+ z
    return normalize(uu)
end

function Update_momentum(d::Number, eff_eps::Number, g::AbstractVector, u::AbstractVector)
    """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
    similar to the implementation: https://github.com/gregversteeg/esh_dynamics
    There are no exponentials e^delta, which prevents overflows when the gradient norm is large."""
    g_norm = norm(g)
    e = -g ./ g_norm
    ue = dot(u, e)
    delta = eff_eps * g_norm / (d - 1)
    zeta = exp(-delta)
    uu = e .* ((1 - zeta) * (1 + zeta + ue * (1 - zeta))) + (2 * zeta) .* u
    delta_r = delta - log(2) + log(1 + ue + (1 - ue) * zeta^2)
    return normalize(uu), delta_r
end

struct MCHMCState{T}
    rng::Random.AbstractRNG
    i::Int
    x::Vector{T}
    u::Vector{T}
    l::T
    g::Vector{T}
    dE::T
    Feps::T
    Weps::T
end

function Transition(sampler::MCHMCSampler, target::Target, state::MCHMCState)
    return [
        target.inv_transform(state.x)[:]
        sampler.hyperparameters.eps
        state.dE
        -state.l
    ]
end

function Step(rng::Random.AbstractRNG, sampler::MCHMCSampler, target::Target; kwargs...)
    sett = sampler.settings
    kwargs = Dict(kwargs)
    d = target.d
    ### initial conditions ###
    if :init_x ∈ keys(kwargs)
        x = target.transform(kwargs[:init_x])
    else
        x = target.prior_draw()
    end
    l, g = target.nlogp_grad_nlogp(x)
    u = Random_unit_vector(rng, target)
    Weps = 1e-5
    Feps = Weps * sampler.hyperparameters.eps^(1 / 6)

    state = MCHMCState(rng, 0, x, u, l, g, 0.0, Feps, Weps)
    state = tune_hyperparameters(rng, sampler, target, state; kwargs...)

    return Step(rng, sampler, target, state; kwargs...)
end

function Step(
    rng::Random.AbstractRNG,
    sampler::MCHMCSampler,
    target::Target,
    state::MCHMCState;
    kwargs...,
)
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
    # Langevin-like noise
    uuu = Partially_refresh_momentum(rng, sampler, target, uu)
    dEE = kinetic_change + ll - state.l

    if adaptive
        varE = dEE^2 / d
        # 1e-8 is added to avoid divergences in log xi        
        xi = varE / TEV + 1e-8
        # the weight which reduces the impact of stepsizes which 
        # are much larger on much smaller than the desired one.        
        w = exp(-0.5 * (log(xi) / (6.0 * sigma_xi))^2)
        # Kalman update the linear combinations
        Feps = gamma * state.Feps + w * (xi / eps^6)
        Weps = gamma * state.Weps + w
        new_eps = (Feps / Weps)^(-1 / 6)

        sampler.hyperparameters.eps = new_eps
        tune_nu!(sampler, target)
    else
        Feps = state.Feps
        Weps = state.Weps
    end

    state = MCHMCState(rng, state.i + 1, xx, uuu, ll, gg, dEE, Weps, Feps)
    transition = Transition(sampler, target, state)
    return transition, state
end

function Sample(sampler::MCHMCSampler, target::Target, nadapt::Int; kwargs...)
    return Sample(Random.GLOBAL_RNG, sampler, target, n; kwargs...)
end

function Sample(
    rng::Random.AbstractRNG,
    sampler::MCHMCSampler,
    target::Target,
    n::Int;
    file_path = "./chain",
    progress = true,
    kwargs...,
)
    """Args:
           n: number of integration steps to take.
           x_initial: initial condition for x (an array of shape (target dimension, )).
                      It can also be 'prior' in which case it is drawn from the prior distribution (self.Target.prior_draw).
           random_key: jax radnom seed, e.g. jax.random.PRNGKey(42).
        Returns:
            samples (shape = (n, self.Target.d))
    """

    io = open(joinpath(file_path, "VarNames.txt"), "w") do io
        println(io, string(target.vsyms))
    end

    transition, state = step(sampler, target; kwargs...)

    chain = []
    push!(chain, transition)

    io = open(string(file_path, ".txt"), "w") do io
        println(io, transition)
        @showprogress "MCHMC: " (progress ? 1 : Inf) for i = 1:n-1
            try
                transition, state = step(sampler, target, state; kwargs...)
                push!(chain, transition)
                println(io, transition)
            catch err
                if err isa InterruptException
                    rethrow(err)
                else
                    @warn "Divergence encountered after tuning"
                end
            end
        end
    end

    io = open(joinpath(fol_name, string(file_name, "_summary.txt")), "w") do io
        ess, rhat = Summarize(chain)
        println(io, ess)
        println(io, rhat)
    end

    return samples
end
