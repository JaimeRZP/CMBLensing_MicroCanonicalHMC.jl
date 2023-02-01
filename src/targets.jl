mutable struct TuringTarget <: Target
    d::Int
    nlogp::Function
    grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

function _get_dists(vi)
    mds = values(vi.metadata)
    return [md.dists[1] for md in mds]
end

TuringTarget(model; kwargs...) = begin
    ctxt = model.context
    vi = DynamicPPL.VarInfo(model, ctxt)
    #Turing.link!!(vi, model)
    vsyms = keys(vi)
    d = length(vsyms)

    ℓ = LogDensityProblemsAD.ADgradient(DynamicPPL.LogDensityFunction(vi, model, ctxt))
    ℓπ = Base.Fix1(LogDensityProblems.logdensity, ℓ)
    #ℓπ = (args...) -> LogDensityProblems.logdensity(ℓ, args...)
    #OPT: we probably want to use this in the future.
    #     and merge nlogp and grad_nlogp into one function.
    #∂lπ∂θ(x) = LogDensityProblems.logdensity_and_gradient(ℓ, x)

    #function transform(xs)
    #    dists = _get_dists(vi)
    #    xxs = [invlink(dist, x) for (dist, x) in zip(dists, xs)]
    #    return xxs
    #end

    function transform(x)
        xt = x
        return xt
    end

    function inv_transform(xt)
        x = xt
        return x
    end

    function nlogp(x)
        return -1.0 .* ℓπ(x)
    end

    function grad_nlogp(x)
        return ForwardDiff.gradient(nlogp, x)
        #OPT: we probably want to use this in the future.
        #return ∂lπ∂θ(x)[2]
    end

    function prior_draw(key)
        x = vi[DynamicPPL.SampleFromPrior()]
        xt = transform(x)
        return xt
    end

    TuringTarget(d,
               nlogp,
               grad_nlogp,
               transform,
               inv_transform,
               prior_draw)
end

mutable struct CustomTarget <: Target
    d::Int
    nlogp::Function
    grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

CustomTarget(nlogp, grad_nlogp, priors; kwargs...) = begin
    d = length(priors)

    #function transform(xs)
    #    xxs = [invlink(dist, x) for (dist, x) in zip(priors, xs)]
    #    return xxs
    #end

    function transform(x)
        xt = x
        return xt
    end

    function inv_transform(xt)
        x = xt
        return x
    end

    function prior_draw(key)
        x = [rand(key, dist) for dist in priors]
        xt = transform(x)
        return xt
    end

    CustomTarget(d,
               nlogp,
               grad_nlogp,
               transform,
               inv_transform,
               prior_draw)
end

mutable struct StandardGaussianTarget <: Target
    d::Int
    nlogp::Function
    grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

StandardGaussianTarget(; kwargs...) = begin

    d = kwargs[:d]

    function transform(x)
        xt = x
        return xt
    end

    function inv_transform(xt)
        x = xt
        return x
    end

    function nlogp(x)
        return 0.5 * sum(x.^2)
    end

    function grad_nlogp(x)
        return ForwardDiff.gradient(nlogp, x)
    end

    function prior_draw(key)
        mean = zeros(d)
        variance = ones(d)
        x = 4*rand(key, MvNormal(mean, variance))
        xt = transform(x)
        return xt
    end

    StandardGaussianTarget(kwargs[:d],
                           nlogp,
                           grad_nlogp,
                           transform,
                           inv_transform,
                           prior_draw)
end

mutable struct CMBLensingTarget <: Target
    d::Int
    Λmass
    nlogp::Function
    grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

CMBLensingTarget(prob; kwargs...) = begin
    d = length(prob.Ωstart)
    Λmass = real(prob.Λmass)
    sqrtΛmass = sqrt(Λmass)
    inv_sqrtΛmass = pinv(sqrtΛmass)

    function transform(x)
        xt = CMBLensing.LenseBasis(inv_sqrtΛmass * x)
        return xt
    end

    function inv_transform(xt)
        x = CMBLensing.LenseBasis(sqrtΛmass * xt)
        return x
    end

    function nlogp(xt)
        x = inv_transform(xt)
        return -1.0 .* prob(x)
    end

    function grad_nlogp(xt)
        return CMBLensing.LenseBasis(Zygote.gradient(nlogp, xt)[1])
        #return ForwardDiff.gradient(nlogp, x)
    end

    function prior_draw(key)
        x = prob.Ωstart
        xt = transform(x)
        return CMBLensing.LenseBasis(xt)
        #return prob.Ωstart
    end

    CMBLensingTarget(d,
                     Λmass,
                     nlogp,
                     grad_nlogp,
                     transform,
                     inv_transform,
                     prior_draw)
end