abstract type Target end

struct StandardGaussianTarget <: Target
    #TO DO: what types are these?
    d::Int
    variance::Vector{Float64}
    prior_draw
    transform
    grap_nlogp
end

function StandardGaussianTarget(;...kwargs)

    function nlogp(x)
        return 0.5 * sum(square.(x))
    end

    function grad_nlogp(x)
        return x
    end

    function transform(x)
        return x
    end

    function prior_draw(key)
        mean = zeros(d)
        variance = ones(d)
        return rand(MvNormal(mean, variance), 1)
    end

    StandardGaussianTarget(kwargs[:d],
                           ones(d),
                           nlogp,
                           grad_nlogp,
                           transform,
                           prior_draw)
end