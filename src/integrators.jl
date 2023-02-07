function Leapfrog(sampler::Sampler, target::Target, x, g, u)
    eps = sampler.hyperparameters.eps
    L = sampler.hyperparameters.L
    lambda_c = sampler.hyperparameters.lambda_c

    uu, dr1 = Update_momentum(sampler, target, eps * 0.5, g, u)

    #full step in x
    xx = x .+ eps .* uu
    gg = target.grad_nlogp(xx) .* target.d ./ (target.d - 1)

    #half step in momentum
    uu, dr2 = Update_momentum(sampler, target, eps * 0.5, gg, uu)

    kinetic_change = (dr1 + dr2) * target.d

    return xx, gg, uu, kinetic_change
end


function Minimal_norm(sampler::Sampler, target::Target, x, g, u)
    """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""
    eps = sampler.hyperparameters.eps
    L = sampler.hyperparameters.L
    lambda_c = sampler.hyperparameters.lambda_c

    # V T V T V
    sett = sampler.settings

    uu, dr1 = Update_momentum(sampler, target, eps * lambda_c, g, u)

    xx = x .+ eps .* 0.5 .* uu
    gg = target.grad_nlogp(xx) .* target.d ./ (target.d - 1)

    uu, dr2 = Update_momentum(sampler, target, eps * (1 - 2 * lambda_c), gg, uu)

    xx = xx .+ eps .* 0.5 .* uu
    gg = target.grad_nlogp(xx) .* target.d / (target.d - 1)

    uu, dr3 = Update_momentum(sampler, target, eps * lambda_c, gg, uu)

    kinetic_change = (dr1 + dr2 + dr3) * (target.d -1)

    return xx, gg, uu, kinetic_change
end