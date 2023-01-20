struct Target <: Target
    #TO DO: what types are these?
    d::Int
    variance::Vector{Float64}
    nlogp::Function
    grad_nlogp::Function
    transform::Function
    prior_draw::Function
end

function _get_dists(vi)
    mds = values(vi.metadata)
    return [md.dists[1] for md in mds]
end

function TuringTarget(model; kwargs...)
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

    function prior_draw()
        return vi[spl]
    end

    Target(kwargs[:d],
           ones(d),
           nlogp,
           grad_nlogp,
           transform,
           prior_draw)
end

function StandardGaussianTarget(; kwargs...)

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

    Target(kwargs[:d],
           ones(d),
           nlogp,
           grad_nlogp,
           transform,
           prior_draw)
end