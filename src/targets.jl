struct StandardGaussianTarget <: Target
    #TO DO: what types are these?
    d::Int
    variance::Vector{Float64}
    nlogp
    grad_nlogp
    transform
    prior_draw
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

    StandardGaussianTarget(kwargs[:d],
                           ones(d),
                           nlogp,
                           grad_nlogp,
                           transform,
                           prior_draw)
end