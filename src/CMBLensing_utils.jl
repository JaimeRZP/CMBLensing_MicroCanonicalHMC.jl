function Random_unit_vector(sampler::Sampler, target::CMBLensingTarget)
    """Generates a random (isotropic) unit vector."""
    return  simulate(Diagonal(one(LenseBasis(diag(target.Λmass)))))

end