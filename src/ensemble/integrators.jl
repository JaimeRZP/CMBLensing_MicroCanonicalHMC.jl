function Leapfrog(sampler::EnsembleSampler, target::ParallelTarget,
                  x::AbstractMatrix, u::AbstractMatrix,
                  l::AbstractVector, g::AbstractMatrix)
    eps = sampler.hyperparameters.eps
    L = sampler.hyperparameters.L
    lambda_c = sampler.hyperparameters.lambda_c
    d = target.target.d

    uu, dr1 = Update_momentum(target, eps * 0.5, g, u)

    #full step in x
    xx = x .+ eps .* uu
    ll, gg = target.nlogp_grad_nlogp(xx)
    gg .*= d/(d-1)

    #half step in momentum
    uu, dr2 = Update_momentum(target, eps * 0.5, gg, uu)

    kinetic_change = (dr1 .+ dr2) .* d

    return xx, uu, ll, gg, kinetic_change
end

function Minimal_norm(sampler::EnsembleSampler, target::ParallelTarget,
                      x::AbstractMatrix, u::AbstractMatrix,
                      l::AbstractVector, g::AbstractMatrix)
    """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""
    # V T V T V
    eps = sampler.hyperparameters.eps
    lambda_c = sampler.hyperparameters.lambda_c
    grad_nlogp = target.grad_nlogp
    d = target.target.d

    uu, dr1 = Update_momentum(d, eps * lambda_c, g, u)

    xx = x .+ eps .* 0.5 .* uu
    ll, gg = target.nlogp_grad_nlogp(xx)
    gg .*= d/(d-1)

    uu, dr2 = Update_momentum(d, eps * (1 - 2 * lambda_c), gg, uu)

    xx = xx .+ eps .* 0.5 .* uu
    ll, gg = target.nlogp_grad_nlogp(xx)
    gg .*= d/(d-1)

    uu, dr3 = Update_momentum(d, eps * lambda_c, gg, uu)

    kinetic_change = (dr1 .+ dr2 .+ dr3) .* (d -1)

    return xx, uu, ll, gg, kinetic_change
end
