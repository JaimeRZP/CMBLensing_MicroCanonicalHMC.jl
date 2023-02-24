function Random_unit_vector(sampler::Sampler, target::CMBLensingTarget;
                            normalize=true)
    """Generates a random (isotropic) unit vector."""
    u = simulate(Diagonal(one(LenseBasis(diag(target.Λmass)))))
    if normalize
        u ./=  sqrt.(sum(u.^2))
    end
    return u
end

function Partially_refresh_momentum(sampler::Sampler, target::CMBLensingTarget, u::AbstractVector)
    """Adds a small noise to u and normalizes."""
    nu = sampler.hyperparameters.nu
    z = nu .* Random_unit_vector(sampler, target; normalize=false)
    uu = (u .+ z) ./ sqrt(sum((u .+ z).^2))
    return uu
end