mutable struct CustomTarget <: Target
    rng::MersenneTwister                                                    
    d::Int
    vsyms                                                            
    nlogp::Function
    grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

CustomTarget(nlogp, grad_nlogp, priors; rng=0, kwargs...) = begin
    d = length(priors)
    vsyms = [DynamicPPL.VarName(Symbol("d_", i,)) for i in 1:d]
    rng = MersenneTwister(rng)                                                             

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

    CustomTarget(
               rng,
               d,
               nlogp,
               grad_nlogp,
               transform,
               inv_transform,
               prior_draw)
end

mutable struct GaussianTarget <: Target
    rng::MersenneTwister                                                                            
    d::Int
    vsyms                                                                                    
    nlogp::Function
    grad_nlogp::Function
    nlogp_grad_nlogp::Function
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

GaussianTarget(_mean::AbstractVector, _cov::AbstractMatrix; rng=0) = begin
    d = length(_mean)
    vsyms = [DynamicPPL.VarName(Symbol("d_", i,)) for i in 1:d]                                            
    rng = MersenneTwister(rng)                                                                                    
                                                                                        
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
    rng,
    d,
    vsyms,                                                                                                                    
    nlogp,
    grad_nlogp,
    nlogp_grad_nlogp,
    transform,
    inv_transform,
    prior_draw)
end
                                                        
mutable struct CMBLensingTarget <: Target
    rng::MersenneTwister 
    d::Int
    Λmass
    nlogp::Function
    grad_nlogp::Function
    nlogp_grad_nlogp::Function 
    transform::Function
    inv_transform::Function
    prior_draw::Function
end

CMBLensingTarget(prob; rng=0, kwargs...) = begin
    Ωstart = prob.Ωstart
    d = length(Ωstart)
    Λmass = real(prob.Λmass)
    sqrtΛmass = sqrt(Λmass)
    inv_sqrtΛmass = pinv(sqrtΛmass)
    rng = MersenneTwister(rng)


    function transform(x)
        xt = CMBLensing.LenseBasis(sqrtΛmass * x)
        return xt
    end

    function inv_transform(xt)
        x = CMBLensing.LenseBasis(inv_sqrtΛmass * xt)
        return x
    end

    function nlogp(xt)
        x = inv_transform(xt)
        return -1.0 .* prob(x)
    end

    function grad_nlogp(xt)
        return CMBLensing.LenseBasis(Zygote.gradient(nlogp, xt)[1])
    end
    
    function nlogp_grad_nlogp(xt)
        return nlogp(xt), grad_nlogp(xt)
    end

    function prior_draw()
        xt = transform(Ωstart)
        return CMBLensing.LenseBasis(xt)
    end

    CMBLensingTarget(rng,
                     d,
                     Λmass,
                     nlogp,
                     grad_nlogp,
                     nlogp_grad_nlogp,
                     transform,
                     inv_transform,
                     prior_draw)
end