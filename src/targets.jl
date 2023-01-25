struct TuringTarget <: Target
    d::Int
    nlogp::Function
    grad_nlogp::Function
    transform::Function
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

    function nlogp(x)
        return -1.0 .* ℓπ(x)
    end

    function grad_nlogp(x)
        return ForwardDiff.gradient(nlogp, x)
        #OPT: we probably want to use this in the future.
        #return ∂lπ∂θ(x)[2]
    end

    #function transform(xs)
    #    dists = _get_dists(vi)
    #    xxs = [invlink(dist, x) for (dist, x) in zip(dists, xs)]
    #    return xxs
    #end

    function transform(x)
        return x
    end

    function prior_draw(key)
        return vi[DynamicPPL.SampleFromPrior()]
    end

    TuringTarget(d,
               nlogp,
               grad_nlogp,
               transform,
               prior_draw)
end

struct CustomTarget <: Target
    d::Int
    nlogp::Function
    grad_nlogp::Function
    transform::Function
    prior_draw::Function
end

CustomTarget(nlogp, grad_nlogp, priors; kwargs...) = begin
    d = length(priors)

    #function transform(xs)
    #    xxs = [invlink(dist, x) for (dist, x) in zip(priors, xs)]
    #    return xxs
    #end

    function transform(x)
        return x
    end

    function prior_draw(key)
        return [rand(key, dist) for dist in priors]
    end

    CustomTarget(d,
               nlogp,
               grad_nlogp,
               transform,
               prior_draw)
end

struct StandardGaussianTarget <: Target
    d::Int
    nlogp::Function
    grad_nlogp::Function
    transform::Function
    prior_draw::Function
end

StandardGaussianTarget(; kwargs...) = begin

    d = kwargs[:d]

    function nlogp(x)
        return 0.5 * sum(x.^2)
    end

    function grad_nlogp(x)
        return ForwardDiff.gradient(nlogp, x)
    end

    function transform(x)
        return x
    end

    function prior_draw(key)
        mean = zeros(d)
        variance = ones(d)
        return 4*rand(key, MvNormal(mean, variance))
    end

    StandardGaussianTarget(kwargs[:d],
                           nlogp,
                           grad_nlogp,
                           transform,
                           prior_draw)
end

struct CMBLensingTarget <: Target
    d::Int
    nlogp::Function
    grad_nlogp::Function
    transform::Function
    prior_draw::Function
end

CMBLensingTarget(prob; kwargs...) = begin
    d = length(prob.Ωstart)
    inv_Λmass = pinv(prob.Λmass)

    function nlogp(x)
        return prob(x)
    end

    function grad_nlogp(x)
        return LenseBasis(Zygote.gradient(nlogp, x)[1])
        #return ForwardDiff.gradient(nlogp, x)
    end

    function transform(x)
        return LenseBasis(inv_Λmass * x)
    end

    function prior_draw(key)
        return LenseBasis(prob.Λmass * prob.Ωstart)
    end

    CMBLensingTarget(d,
                     nlogp,
                     grad_nlogp,
                     transform,
                     prior_draw)
end