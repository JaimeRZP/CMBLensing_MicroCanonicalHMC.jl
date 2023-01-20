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
    # Hack soon to be depricated
    spl = DynamicPPL.SampleFromPrior()
    ###
    ctxt = model.context
    vi = DynamicPPL.VarInfo(model, context)
    link!!(vi, model)
    vsyms = keys(vi)
    d = length(vsyms)

    ℓ = LogDensityProblemsAD.ADgradient(Turing.LogDensityFunction(vi, model, spl, ctxt))
    ℓπ = Base.Fix1(LogDensityProblems.logdensity, ℓ)
    #OPT: we probably want to use this in the future.
    #     and merge nlogp and grad_nlogp into one function.
    #∂lπ∂θ(x) = LogDensityProblems.logdensity_and_gradient(ℓ, x)

    function nlogp(x)
        return ℓπ(x)
    end

    function grad_nlogp(x)
        return ForwardDiff.gradient(nlogp, x)
        #OPT: we probably want to use this in the future.
        #return ∂lπ∂θ(x)[2]
    end

    function transform(xs)
        dists = _get_dists(vi)
        xxs = [invlink(dist, x) for (dist, x) in zip(dists, xs)]
        return xxs
    end

    function prior_draw(key)
        return vi[spl]
    end

    TuringTarget(kwargs[:d],
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

    d = kwargs[:d]
    inv_Λmass = inv(Λmass)

    function nlogp(x)
        return prob()
    end

    function grad_nlogp(x)
        return Zygote.gradient(prob, x)[1]
    end

    function transform(x)
        return inv_Λmass * x
    end

    function prior_draw(key)
        return prob.Λmass * prob.Ωstart
    end

    StandardGaussianTarget(kwargs[:d],
                           nlogp,
                           grad_nlogp,
                           transform,
                           prior_draw)
end