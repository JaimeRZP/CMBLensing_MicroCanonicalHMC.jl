function Leapfrog(sampler::Sampler, target::Target, state::State)
    eps = sampler.hyperparameters.eps
    sigma = sampler.hyperparameters.sigma
    return Leapfrog(target, eps, sigma, state.x, state.u, state.l, state.g)
end

function Leapfrog(target::Target,
                  eps::Number, sigma::AbstractVector,
                  x::AbstractVector, u::AbstractVector,
                  l::Number, g::AbstractVector)
    """leapfrog"""
    d = target.d
    # go to the latent space
    z = x ./ sigma 
    
    #half step in momentum
    uu, dr1 = Update_momentum(d, eps * 0.5, g .* sigma, u)

    #full step in x
    zz = z .+ eps .* uu
    xx = zz .* sigma # rotate back to parameter space
    ll, gg = target.nlogp_grad_nlogp(xx)

    #half step in momentum
    uu, dr2 = Update_momentum(d, eps * 0.5, gg .* sigma, uu)
    kinetic_change = (dr1 + dr2) * (d - 1)

    return xx, uu, ll, gg, kinetic_change
end

function Minimal_norm(sampler::Sampler, target::Target, state::State)
    """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""
    # V T V T V
    eps = sampler.hyperparameters.eps
    lambda_c = sampler.hyperparameters.lambda_c
    sigma = sampler.hyperparameters.sigma
    return Minimal_norm(target, eps, lambda_c, sigma, state.x, state.u, state.l, state.g)
end

function Minimal_norm(target::Target,
                      eps::Number, lambda_c::Number, sigma::AbstractVector,
                      x::AbstractVector, u::AbstractVector,
                      l::Number, g::AbstractVector)
    """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""
    d = target.d
    # go to the latent space
    z = x ./ sigma 
    
    # V T V T V
    #V (momentum update)
    uu, dr1 = Update_momentum(d, eps * lambda_c, g .* sigma, u)

    #T (postion update)
    zz = z .+ (0.5 * eps) .* uu
    xx = sigma .* zz
    ll, gg = target.nlogp_grad_nlogp(xx)
    
    #V (momentum update)
    uu, dr2 = Update_momentum(d, eps * (1 - 2 * lambda_c), gg .* sigma, uu)
    
    #T (postion update)
    zz = zz .+ (0.5 * eps) .* uu
    xx = zz .* sigma
    ll, gg = target.nlogp_grad_nlogp(xx)
    
    #V (momentum update)
    uu, dr3 = Update_momentum(d, eps * lambda_c, gg .* sigma, uu)
    
    #kinetic energy change
    kinetic_change = (dr1 + dr2 + dr3) * (d -1)

    return xx, uu, ll, gg, kinetic_change
end
