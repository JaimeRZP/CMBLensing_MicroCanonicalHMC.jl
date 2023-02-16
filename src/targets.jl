mutable struct TuringTarget <: Target
    model::DynamicPPL.Model
    d::Int
    vsyms
    dists
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

TuringTarget(model; kwargs...) = begin
    ctxt = model.context
    vi = DynamicPPL.VarInfo(model, ctxt)
    vi_t = Turing.link!!(vi, model)
    dists = _get_dists(vi)
    vsyms = keys(vi)
    d = length(vsyms)

    ℓ = LogDensityProblemsAD.ADgradient(DynamicPPL.LogDensityFunction(vi_t, model, ctxt))
    ℓπ(x) = LogDensityProblems.logdensity(ℓ, x)
    ∂lπ∂θ(x) = LogDensityProblems.logdensity_and_gradient(ℓ, x)


    function transform(x)
        xt = [Bijectors.link(dist, par) for (dist, par) in zip(dists, x)]
        return xt
    end

    function inv_transform(xt)
        x = [Bijectors.invlink(dist, par) for (dist, par) in zip(dists, xt)]
        return x
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

    function prior_draw(key)
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
               prior_draw)
end

mutable struct ParallelTarget <: Target
    target::Target
    nlogp::Function
    grad_nlogp::Function
    nlogp_grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

ParallelTarget(target::Target, nchains) = begin
    d = target.d
    function transform(xs)
        xs_t = Matrix{Real}(undef, nchains, d)
        @inbounds Threads.@threads :static for i in 1:nchains
            xs_t[i, :] .= target.transform(xs[i, :])
        end
        return xs_t
    end

    function inv_transform(xs_t)
        xs = Matrix{Real}(undef, nchains, d)
        @inbounds Threads.@threads :static for i in 1:nchains
            xs[i, :] .= target.inv_transform(xs_t[i, :])
        end
        return xs
    end

    function nlogp(xs_t)
        ls = Vector{Real}(undef, nchains)
        @inbounds Threads.@threads :static for i in 1:nchains
            ls[i] = target.nlogp(xs_t[i, :])
        end
        return ls
    end

    function grad_nlogp(xs_t)
        gs = Matrix{Real}(undef, nchains, d)
        @inbounds Threads.@threads :static for i in 1:nchains
            gs[i, :] .= target.grad_nlogp(xs_t[i, :])
        end
        return gs
    end

    function nlogp_grad_nlogp(xs_t)
        ls = Vector{Real}(undef, nchains)
        gs = Matrix{Real}(undef, nchains, d)
        @inbounds Threads.@threads :static for i in 1:nchains
            ls[i], = target.nlogp(xs_t[i, :])
            gs[i, :] = target.grad_nlogp(xs_t[i, :])
        end
        return ls , gs
    end

    function prior_draw(key)
        xs_t = Matrix{Real}(undef, nchains, d)
        @inbounds Threads.@threads :static for i in 1:nchains
            xs_t[i, :] .= target.prior_draw(key)
        end
        return xs_t
    end

    ParallelTarget(
        target,
        nlogp,
        grad_nlogp,
        nlogp_grad_nlogp,
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
    #    xxs = [Bijectors.invlink(dist, x) for (dist, x) in zip(priors, xs)]
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
