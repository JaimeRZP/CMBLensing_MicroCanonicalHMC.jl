chainsstack(c::AbstractVector{MCMCChains.Chains}) = reduce(chainscat, c)

function AbstractMCMC.sample(model::DynamicPPL.Model,
                             sampler::AbstractMCMC.AbstractSampler,
                             ::MCMCThreads,
                             N::Integer,
                             nchains::Integer;
                             progress=PROGRESS[],
                             progressname="Sampling",
                             resume_from=nothing,
                             kwargs...)

    if resume_from === nothing
        target = TuringTarget(model)
        init = Init(sampler, target; kwargs...)
        state, sample = init
        # We will have to parallelize this later
        tune_hyperparameters(sampler, target, state; kwargs...)

        if nchains < Threads.nthreads()
            @info string("number of chains: ",
                         nchains,
                         " smaller than number of threads: ",
                         Threads.nthreads(),  ".",
                         " Increase the number of chains to make full use of your threads.")
        end
        if nchains > Threads.nthreads()
            @info string("number of chains: ",
                         nchains,
                         " requesteed larger than number of threads: ",
                         Threads.nthreads(),  ".",
                         " Setting number of chains to number of threads.")
            nchains = Threads.nthreads()
        end

        interval = 1:nchains
        chains = Vector{MCMCChains.Chains}(undef, nchains)
        targets = [deepcopy(target) for _ in interval]
        samplers = [deepcopy(sampler) for _ in interval]
        inits = [Init(sampler, target) for _ in interval]

    else
        @info "Starting from previous run"
        nchains = length(resume_from)
        interval = 1:nchains
        chains = Vector{MCMCChains.Chains}(undef, nchains)
        targets = [chain.info[:target] for chain in resume_from]
        samplers = [chain.info[:sampler] for chain in resume_from]
        inits = [chain.info[:init] for chain in resume_from]
    end

    @AbstractMCMC.ifwithprogresslogger progress name = progressname begin
        # Create a channel for progress logging.
        if progress
            channel = Channel{Bool}(length(interval))
        end

        Threads.@threads for i in interval
            _sampler = samplers[i]
            _target = targets[i]
            _init = inits[i]
            chains[i] = AbstractMCMC.mcmcsample(
                _target,
                _sampler,
                _init,
                N;
                progress=PROGRESS[],
                progressname=string("chain ", i),
                kwargs...)

            # Update the progress bar.
            progress && put!(channel, true)
        end
    end

    return chains
end
