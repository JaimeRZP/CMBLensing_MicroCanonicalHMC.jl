function Leapfrog(sampler::Sampler, x, g, u)
    #TO DO: type the inputs
    sett = sampler.settings
    target = sampler.target

    uu = Update_momentum(sampler, sett.eps * 0.5, g, u)

    #full step in x
    xx = x + sett.eps * uu
    gg = @.(target.grad_nlogp(xx) * target.d / (target.d - 1))

    #half step in momentum
    uu = Update_momentum(sampler, sett.eps * 0.5, gg, uu)

    return xx, gg, uu
end


function Minimal_norm(sampler::Sampler, x, g, u)
    """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

    # V T V T V
    sett = sampler.settings
    target = sampler.target

    uu = Update_momentum(sampler, sett.eps * sett.lambda_c, g, u)

    xx = @.(x + sett.eps * 0.5 * uu)
    gg = @.(target.grad_nlogp(xx) * target.d / (target.d - 1))

    uu = Update_momentum(sampler, sett.eps .* (1 .- 2 .* sett.lambda_c), gg, uu)

    xx = @.(xx + self.eps * 0.5 * uu)
    gg = @.(target.grad_nlogp(xx) * self.Target.d / (self.Target.d - 1))

    uu = Update_momentum(sett.eps * sett.lambda_c, gg, uu)

    return xx, gg, uu
end