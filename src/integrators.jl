function Leapfrog(sampler::MCHMCSampler, h::Hamiltonian, state::MCHMCState)
    eps = sampler.hyperparameters.eps
    sigma = sampler.hyperparameters.sigma
    return Leapfrog(h, eps, sigma, state.x, state.u, state.l, state.g)
end

function Leapfrog(
    h::Hamiltonian,
    eps::Number,
    sigma::AbstractVector,
    x::AbstractVector,
    u::AbstractVector,
    l::Number,
    g::AbstractVector,
)
    """leapfrog"""
    d = length(x)
    # go to the latent space
    z = x ./ sigma

    #half step in momentum
    uu, dr1 = Update_momentum(d, eps * 0.5, g .* sigma, u)

    #full step in x
    zz = z .+ eps .* uu
    xx = zz .* sigma # rotate back to parameter space
    ll, gg = -1 .* h.∂lπ∂θ(xx)

    #half step in momentum
    uu, dr2 = Update_momentum(d, eps * 0.5, gg .* sigma, uu)
    kinetic_change = (dr1 + dr2) * (d - 1)

    return xx, uu, ll, gg, kinetic_change
end

function Minimal_norm(sampler::MCHMCSampler, h::Hamiltonian, state::MCHMCState)
    """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""
    # V T V T V
    eps = sampler.hyperparameters.eps
    lambda_c = sampler.hyperparameters.lambda_c
    sigma = sampler.hyperparameters.sigma
    return Minimal_norm(h, eps, lambda_c, sigma, state.x, state.u, state.l, state.g)
end

function Minimal_norm(
    h::Hamiltonian,
    eps::Number,
    lambda_c::Number,
    sigma::AbstractVector,
    x::AbstractVector,
    u::AbstractVector,
    l::Number,
    g::AbstractVector,
)
    """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""
    d = length(x)
    # go to the latent space
    z = x ./ sigma

    # V T V T V
    #V (momentum update)
    uu, dr1 = Update_momentum(d, eps * lambda_c, g .* sigma, u)

    #T (postion update)
    zz = z .+ (0.5 * eps) .* uu
    xx = sigma .* zz
    ll, gg = -1 .* h.∂lπ∂θ(xx)

    #V (momentum update)
    uu, dr2 = Update_momentum(d, eps * (1 - 2 * lambda_c), gg .* sigma, uu)

    #T (postion update)
    zz = zz .+ (0.5 * eps) .* uu
    xx = zz .* sigma
    ll, gg = -1 .* h.∂lπ∂θ(xx)

    #V (momentum update)
    uu, dr3 = Update_momentum(d, eps * lambda_c, gg .* sigma, uu)

    #kinetic energy change
    kinetic_change = (dr1 + dr2 + dr3) * (d - 1)

    return xx, uu, ll, gg, kinetic_change
end
