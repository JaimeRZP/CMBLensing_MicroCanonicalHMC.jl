mutable struct TuringTarget <: Target
    model::DynamicPPL.Model
    d::Int
    vsyms::Any
    dists::Any
    nlogp::Function
    grad_nlogp::Function
    nlogp_grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

function _get_dists(vi)
    mds = values(vi.metadata)
    return [md.dists[1] for md in mds]
end

function _name_variables(vi, dist_lengths)
    vsyms = keys(vi)
    names = []
    for (vsym, dist_length) in zip(vsyms, dist_lengths)
        if dist_length == 1
            name = [vsym]
            append!(names, name)
        else
            name = [DynamicPPL.VarName(Symbol(vsym, i)) for i = 1:dist_length]
            append!(names, name)
        end
    end
    return names
end

TuringTarget(model; kwargs...) = begin
    ctxt = model.context
    vi = DynamicPPL.VarInfo(model, ctxt)
    vi_t = Turing.link!!(vi, model)
    dists = _get_dists(vi)
    dist_lengths = [length(dist) for dist in dists]
    vsyms = _name_variables(vi, dist_lengths)
    d = length(vsyms)

    ℓ = LogDensityProblemsAD.ADgradient(DynamicPPL.LogDensityFunction(vi_t, model, ctxt))
    ℓπ(x) = LogDensityProblems.logdensity(ℓ, x)
    ∂lπ∂θ(x) = LogDensityProblems.logdensity_and_gradient(ℓ, x)

    function _reshape_params(x::AbstractVector)
        xx = []
        idx = 0
        for dist_length in dist_lengths
            append!(xx, [x[idx+1:idx+dist_length]])
            idx += dist_length
        end
        return xx
    end

    function transform(x)
        x = _reshape_params(x)
        xt = [Bijectors.link(dist, par) for (dist, par) in zip(dists, x)]
        return vcat(xt...)
    end

    function inv_transform(xt)
        xt = _reshape_params(xt)
        x = [Bijectors.invlink(dist, par) for (dist, par) in zip(dists, xt)]
        return vcat(x...)
    end

    function nlogp(xt)
        return -ℓπ(xt)
    end

    function grad_nlogp(xt)
        return ForwardDiff.gradient(nlogp, xt)
    end

    function nlogp_grad_nlogp(xt)
        return -1 .* ∂lπ∂θ(xt)
    end

    function prior_draw()
        ctxt = model.context
        vi = DynamicPPL.VarInfo(model, ctxt)
        vi_t = Turing.link!!(vi, model)
        return vi_t[DynamicPPL.SampleFromPrior()]
    end

    TuringTarget(
        model,
        d,
        vsyms,
        dists,
        nlogp,
        grad_nlogp,
        nlogp_grad_nlogp,
        transform,
        inv_transform,
        prior_draw,
    )
end


mutable struct CustomTarget <: Target
    d::Int
    vsyms::Any
    nlogp::Function
    grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

CustomTarget(nlogp, grad_nlogp, priors; kwargs...) = begin
    d = length(priors)
    vsyms = [DynamicPPL.VarName(Symbol("d_", i)) for i = 1:d]

    function transform(x)
        xt = x
        return xt
    end

    function inv_transform(xt)
        x = xt
        return x
    end

    function prior_draw()
        x = [rand(dist) for dist in priors]
        xt = transform(x)
        return xt
    end

    CustomTarget(d, nlogp, grad_nlogp, transform, inv_transform, prior_draw)
end

mutable struct GaussianTarget <: Target
    d::Int
    vsyms::Any
    nlogp::Function
    grad_nlogp::Function
    nlogp_grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

GaussianTarget(_mean::AbstractVector, _cov::AbstractMatrix) = begin
    d = length(_mean)
    vsyms = [DynamicPPL.VarName(Symbol("d_", i)) for i = 1:d]

    _gaussian = MvNormal(_mean, _cov)
    ℓπ(θ::AbstractVector) = logpdf(_gaussian, θ)
    ∂lπ∂θ(θ::AbstractVector) = gradlogpdf(_gaussian, θ)

    function transform(x)
        xt = x
        return xt
    end

    function inv_transform(xt)
        x = xt
        return x
    end

    function nlogp(x)
        xt = transform(x)
        return -ℓπ(xt)
    end

    function grad_nlogp(x)
        xt = transform(x)
        return -∂lπ∂θ(xt)
    end

    function nlogp_grad_nlogp(x)
        l = nlogp(x)
        g = grad_nlogp(x)
        return l, g
    end

    function prior_draw()
        xt = rand(MvNormal(zeros(d), ones(d)))
        return xt
    end

    GaussianTarget(
        d,
        vsyms,
        nlogp,
        grad_nlogp,
        nlogp_grad_nlogp,
        transform,
        inv_transform,
        prior_draw,
    )
end

mutable struct RosenbrockTarget <: Target
    d::Int
    vsyms::Any
    nlogp::Function
    grad_nlogp::Function
    nlogp_grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

RosenbrockTarget(a, b; kwargs...) = begin
    kwargs = Dict(kwargs)
    d = kwargs[:d]
    vsyms = [DynamicPPL.VarName(Symbol("d_", i)) for i = 1:d]

    function ℓπ(x, y; a = a, b = b)
        m = @.((a - x)^2 + b * (y - x^2)^2)
        return -0.5 * sum(m)
    end

    function transform(x)
        xt = x
        return xt
    end

    function inv_transform(xt)
        x = xt
        return x
    end

    function nlogp(x)
        xt = transform(x)
        xt_1 = xt[1:Int(d / 2)]
        xt_2 = xt[Int(d / 2)+1:end]
        return -ℓπ(xt_1, xt_2)
    end

    function grad_nlogp(x)
        return ForwardDiff.gradient(nlogp, x)
    end

    function nlogp_grad_nlogp(x)
        l = nlogp(x)
        g = grad_nlogp(x)
        return l, g
    end

    function prior_draw()
        xt = rand(MvNormal(zeros(d), ones(d)))
        return xt
    end

    RosenbrockTarget(
        d,
        vsyms,
        nlogp,
        grad_nlogp,
        nlogp_grad_nlogp,
        transform,
        inv_transform,
        prior_draw,
    )
end
