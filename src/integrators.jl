function Leapfrog(sampler::Sampler, target::Target, x, g, u, r)
    eps = sampler.hyperparameters.eps
    L = sampler.hyperparameters.L
    lambda_c = sampler.hyperparameters.lambda_c

    uu, rr = Update_momentum(sampler, target, eps * 0.5, g, u, r)

    #full step in x
    xx = x .+ eps .* uu
    gg = target.grad_nlogp(xx) .* target.d ./ (target.d - 1)

    #half step in momentum
    uu, rr = Update_momentum(sampler, target, eps * 0.5, gg, uu, rr)

    return xx, gg, uu, rr
end


function Minimal_norm(sampler::Sampler, target::Target, x, g, u, r)
    """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""
    eps = sampler.hyperparameters.eps
    L = sampler.hyperparameters.L
    lambda_c = sampler.hyperparameters.lambda_c

    # V T V T V
    sett = sampler.settings

    uu, rr = Update_momentum(sampler, target, eps * lambda_c, g, u, r)

    xx = x .+ eps .* 0.5 .* uu
    gg = target.grad_nlogp(xx) .* target.d ./ (target.d - 1)

    uu, rr = Update_momentum(sampler, target, eps * (1 - 2 * lambda_c), gg, uu, rr)

    xx = xx .+ eps .* 0.5 .* uu
    gg = target.grad_nlogp(xx) .* target.d / (target.d - 1)

    uu, rr = Update_momentum(sampler, target, eps * lambda_c, gg, uu, rr)

    return xx, gg, uu, rr
end