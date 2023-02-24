mutable struct TuringTarget <: Target
    model::DynamicPPL.Model
    d::Int
    vsyms
    dists
    nlogp::Function
    grad_nlogp::Function
    nlogp_grad_nlogp::Function
    hess_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
    MAP::AbstractVector
    MAP_t::AbstractVector
end

function _get_dists(vi)
    mds = values(vi.metadata)
    return [md.dists[1] for md in mds]
end

TuringTarget(model; compute_MAP=true, kwargs...) = begin
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

    function hess_nlogp(xt)
        return ForwardDiff.hessian(nlogp, xt)
    end

    function prior_draw(key)
        ctxt = model.context
        vi = DynamicPPL.VarInfo(model, ctxt)
        vi_t = Turing.link!!(vi, model)
        return vi_t[DynamicPPL.SampleFromPrior()]
    end

    if compute_MAP
        MAP_t = Optim.minimizer(optimize(nlogp, prior_draw(0.0), Newton(); autodiff = :forward))
        MAP = inv_transform(MAP_t)
    else
        MAP = MAP_t = zeros(d)
    end

    TuringTarget(
               model,
               d,
               vsyms,
               dists,
               nlogp,
               grad_nlogp,
               nlogp_grad_nlogp,
               hess_nlogp,
               transform,
               inv_transform,
               prior_draw,
               MAP,
               MAP_t)
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

mutable struct RosenbrockTarget <: Target
    d::Int
    nlogp::Function
    grad_nlogp::Function
    nlogp_grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

struct Rosenbrock{Tμ,Ta,Tb}
    μ::Tμ
    a::Ta
    b::Tb
end

RosenbrockTarget(Tμ, Ta, Tb; kwargs...) = begin
    kwargs = Dict(kwargs)
    D = HybridRosenbrock(Tμ, Ta, Tb)
    d = kwargs[:d]

    block = [1.0, D.μ; D.μ 1.0]
    cov = BlockDiagonal([block for _ in 1:(d/2)])

    function _mean(θ::AbstractVector)
        i = 1:(d/2)
        u = ones(length(θ))
        u[2 .* i] .= θ[2 * i] ./ D.a
        u[2 .* (i .+ 1)] .= D.a .* θ[2 * i ] .- D.b .* (θ[2 * i] ^ 2 + D.a ^ 2)
        return u
    end

    ℓπ(θ::AbstractVector) = logpdf(MvNormal(_mean(θ), cov), θ)

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
        return ForwardDiff.gradient(nlogp, x)
    end

    function nlogp_grad_nlogp(x)
        l = nlogp(x)
        g = grad_nlogp(x)
        return l, g
    end

    function prior_draw(key)
        xt = MvNormal(zeros(d), ones(d))
        return xt
    end

    RosenbrockTarget(d,
    nlogp,
    grad_nlogp,
    nlogp_grad_nlogp,
    transform,
    inv_transform,
    prior_draw)
end
