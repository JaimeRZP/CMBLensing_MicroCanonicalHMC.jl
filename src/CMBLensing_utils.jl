function Random_unit_vector(sampler::Sampler, target::CMBLensingTarget)
    """Generates a random (isotropic) unit vector."""
    rng = target.rng
    u = simulate(rng, Diagonal(one(LenseBasis(diag(target.Λmass)))))
    u ./=  sqrt.(sum(u.^2))
    return u
end

function Partially_refresh_momentum(sampler::Sampler, target::CMBLensingTarget, u::AbstractVector)
    """Adds a small noise to u and normalizes."""
    nu = sampler.hyperparameters.nu
    z = nu .* Random_unit_vector(sampler, target)
    uu = (u .+ z) ./ sqrt(sum((u .+ z).^2))
    return uu
end