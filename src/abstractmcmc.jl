function AbstractMCMC.step(sampler::MCHMCSampler, model::AbstractMCMC.LogDensityModel; kwargs...)
    h = Hamiltonian(model)
    return Step(sampler::Sampler, target::Target; kwargs...)
end

function AbstractMCMC.step(sampler::MCHMCSampler, model::AbstractMCMC.LogDensityModel, state::MCHMCState; kwargs...)
    return Step(sampler::Sampler, target::Target, state; kwargs...)
end