function Random_unit_vector(target::CMBLensingTarget;
                            normalize=true)
    """Generates a random (isotropic) unit vector."""
    u = simulate(target.rng, Diagonal(one(LenseBasis(diag(target.Λmass)))))
    u.θ.r = 0.0
    u.θ.Aϕ = 0.0
    if normalize
        u ./=  sqrt.(sum(u.^2))
    end
    return u
end

function Partially_refresh_momentum(sampler::Sampler, target::CMBLensingTarget, u::AbstractVector)
    """Adds a small noise to u and normalizes."""
    nu = sampler.hyperparameters.nu
    z = nu .* Random_unit_vector(target; normalize=false)
    uu = (u .+ z) 
    uu.θ.r = 0.0
    uu.θ.Aϕ = 0.0
    uu ./= sqrt(sum((uu).^2))
    return uu
end