function Random_unit_vector(sampler::Sampler, target::CMBLensingTarget;
                            normalize=true)
    """Generates a random (isotropic) unit vector."""
    u = simulate(Diagonal(one(LenseBasis(diag(target.Λmass)))))
    if normalize
        u ./=  sqrt.(sum(u.^2))
    end
    return u
end