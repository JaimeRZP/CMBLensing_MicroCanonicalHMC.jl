function AbstractMCMC.step(sampler::MCHMCSampler, model::AbstractMCMC.LogDensityModel; kwargs...)
    logdensity = model.logdensity
    h = Hamiltonian(logdensity)
    return Step(sampler, h; kwargs...)
end

function AbstractMCMC.step(sampler::MCHMCSampler, model::AbstractMCMC.LogDensityModel, state::MCHMCState; kwargs...)
    logdensity = model.logdensity
    h = Hamiltonian(logdensity)
    return Step(sampler, h, state; kwargs...)
end