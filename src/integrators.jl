function Leapfrog(sampler::Sampler, target::Target, x, g, u, r)
    sett = sampler.settings

    uu, rr = Update_momentum(sampler, target, sett.eps * 0.5, g, u, r)

    #full step in x
    xx = x .+ sett.eps .* uu
    gg = target.grad_nlogp(xx) .* target.d ./ (target.d - 1)

    #half step in momentum
    uu, rr = Update_momentum(sampler, target, sett.eps * 0.5, gg, uu, rr)

    return xx, gg, uu, rr
end


function Minimal_norm(sampler::Sampler, target::Target, x, g, u, r)
    """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

    # V T V T V
    sett = sampler.settings

    uu, rr = Update_momentum(sampler, target, sett.eps * sett.lambda_c, g, u, r)

    xx = x .+ sett.eps .* 0.5 .* uu
    gg = target.grad_nlogp(xx) .* target.d ./ (target.d - 1)

    uu, rr = Update_momentum(sampler, target, sett.eps * (1 - 2 * sett.lambda_c), gg, uu, rr)

    xx = xx .+ sett.eps .* 0.5 .* uu
    gg = target.grad_nlogp(xx) .* target.d / (target.d - 1)

    uu, rr = Update_momentum(sampler, target, sett.eps * sett.lambda_c, gg, uu, rr)

    return xx, gg, uu, rr
end