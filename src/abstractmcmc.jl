function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    sampler::MCHMCSampler,
    target::Target;
    kwargs...
)
    return Step(rng, sampler, target; kwargs...)
end 

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    sampler::MCHMCSampler,
    target::Target,
    state::MCHMCState;
    kwargs...
)
    return Step(rng, sampler, target, state; kwargs...)
end               