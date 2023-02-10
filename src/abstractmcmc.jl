
const PROGRESS = Ref(true)

function resume(chain::MCMCChains.Chains, N; kwargs...)
    isempty(chain.info) && error("cannot resume from a chain without state info")

    # Sample a new chain.
    return AbstractMCMC.mcmcsample(
        chain.info[:target],
        chain.info[:sampler],
        N;
        kwargs...)
end

function AbstractMCMC.step(sampler::Sampler, target::Target, state; kwargs...)
    return Step(sampler::Sampler, target::Target, state; kwargs...)
end

function AbstractMCMC.sample(model::DynamicPPL.Model,
                             sampler::AbstractMCMC.AbstractSampler,
                             N::Int;
                             resume_from=nothing,
                             kwargs...)
    # Get target
    target = TuringTarget(model)
    if resume_from === nothing
        return AbstractMCMC.mcmcsample(target, sampler, N; kwargs...)
    else
        return resume(resume_from, N; kwargs...)
    end
end

function AbstractMCMC.mcmcsample(target::AbstractMCMC.AbstractModel,
                                 sampler::AbstractMCMC.AbstractSampler,
                                 N::Integer;
                                 save_state=true,
                                 burn_in = 0,
                                 progress=PROGRESS[],
                                 progressname="Sampling",
                                 callback=nothing,
                                 thinning=1,
                                 kwargs...)

    # Check the number of requested samples.
    N > 0 || error("the number of samples must be ≥ 1")
    Ntotal = thinning * (N - 1) + burn_in + 1

    # Start the timer
    start = time()
    local state

    # Obtain the initial sample and state.
    state, sample = Get_initial_conditions(sampler, target; kwargs...)
    tune_hyperparameters(sampler, target, state; kwargs...)

    @AbstractMCMC.ifwithprogresslogger progress name = progressname begin
        # Determine threshold values for progress logging
        # (one update per 0.5% of progress)
        if progress
            threshold = Ntotal ÷ 200
            next_update = threshold
        end

        # Discard burn in.
        for i in 1:burn_in
            # Update the progress bar.
            if progress && i >= next_update
                AbstractMCMC.ProgressLogging.@logprogress i / Ntotal
                next_update = i + threshold
            end

            # Obtain the next sample and state.
            state, sample = AbstractMCMC.step(sampler, target, state; kwargs...)
        end

        # Run callback.
        callback === nothing || callback(rng, target, sampler, sample, state, 1; kwargs...)

        # Save the sample.
        samples = AbstractMCMC.samples(sample, target, sampler, N; kwargs...)
        samples = AbstractMCMC.save!!(samples, sample, 1, target, sampler, N; kwargs...)

        # Update the progress bar.
        itotal = 1 + burn_in
        if progress && itotal >= next_update
            AbstractMCMC.ProgressLogging.@logprogress itotal / Ntotal
            next_update = itotal + threshold
        end

        # Step through the sampler.
        for i in 2:N
            # Discard thinned samples.
            for _ in 1:(thinning - 1)
                # Obtain the next sample and state.
                state, sample = AbstractMCMC.step(sampler, target, state; kwargs...)

                # Update progress bar.
                if progress && (itotal += 1) >= next_update
                    AbstractMCMC.ProgressLogging.@logprogress itotal / Ntotal
                    next_update = itotal + threshold
                end
            end

            # Obtain the next sample and state.
            state, sample = AbstractMCMC.step(sampler, target, state; kwargs...)

            # Run callback.
            callback === nothing || callback(rng, target, sampler, sample, state, i; kwargs...)

            # Save the sample.
            samples = AbstractMCMC.save!!(samples, sample, 1, target, sampler, N; kwargs...)

            # Update the progress bar.
            if progress && (itotal += 1) >= next_update
                AbstractMCMC.ProgressLogging.@logprogress itotal / Ntotal
                next_update = itotal + threshold
            end
        end
    end

    # Get the sample stop time.
    stop = time()
    duration = stop - start
    stats = AbstractMCMC.SamplingStats(start, stop, duration)

    return AbstractMCMC.bundle_samples(
    samples,
    target,
    sampler,
    state;
    save_state=save_state,
    stats=stats,
    burn_in=burn_in,
    thinning=thinning,
    kwargs...)
end

#=
function AbstractMCMC.mcmcsample(
    target::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    ::MCMCThreads,
    N::Integer,
    nchains::Integer;
    save_state=true,
    burn_in = 0,
    progress=PROGRESS[],
    progressname="Sampling",
    callback=nothing,
    thinning=1,
    kwargs...)
    # Check if actually multiple threads are used.
    if Threads.nthreads() == 1
        @warn "Only a single thread available: MCMC chains are not sampled in parallel"
    end

    # Check if the number of chains is larger than the number of samples
    if nchains > N
        @warn "Number of chains ($nchains) is greater than number of samples per chain ($N)"
    end

    # Copy the random number generator, model, and sample for each thread
    nchunks = min(nchains, Threads.nthreads())
    chunksize = cld(nchains, nchunks)
    interval = 1:nchunks
    targets = [deepcopy(target) for _ in interval]
    samplers = [deepcopy(sampler) for _ in interval]

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    # Ensure that initial parameters are `nothing` or indexable
    _init_params = _first_or_nothing(init_params, nchains)

    # Set up a chains vector.
    chains = Vector{Any}(undef, nchains)

    @ifwithprogresslogger progress name = progressname begin
        # Create a channel for progress logging.
        if progress
            channel = Channel{Bool}(length(interval))
        end

        Distributed.@sync begin
            if progress
                # Update the progress bar.
                Distributed.@async begin
                    # Determine threshold values for progress logging
                    # (one update per 0.5% of progress)
                    threshold = nchains ÷ 200
                    nextprogresschains = threshold

                    progresschains = 0
                    while take!(channel)
                        progresschains += 1
                        if progresschains >= nextprogresschains
                            ProgressLogging.@logprogress progresschains / nchains
                            nextprogresschains = progresschains + threshold
                        end
                    end
                end
            end

            Distributed.@async begin
                try
                    Distributed.@sync for (i, _target, _sampler) in
                                          zip(1:nchunks, rngs, models, samplers)
                        chainidxs = if i == nchunks
                            ((i - 1) * chunksize + 1):nchains
                        else
                            ((i - 1) * chunksize + 1):(i * chunksize)
                        end
                        Threads.@spawn for chainidx in chainidxs
                            # Sample a chain and save it to the vector.
                            chains[chainidx] = AbstractMCMC.mcmcsample(
                                _target,
                                _sampler,
                                N;
                                progress=false,
                                init_params=if _init_params === nothing
                                    nothing
                                else
                                    _init_params[chainidx]
                                end,
                                kwargs...,
                            )

                            # Update the progress bar.
                            progress && put!(channel, true)
                        end
                    end
                finally
                    # Stop updating the progress bar.
                    progress && put!(channel, false)
                end
            end
        end
    end

    # Concatenate the chains together.
    return chainsstack(tighten_eltype(chains))
end
=#

function AbstractMCMC.bundle_samples(
    samples::Vector,
    target::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    state;
    save_state = true,
    stats = missing,
    burn_in = 0,
    thinning = 1,
    kwargs...)

    param_names = target.vsyms
    internal_names = [:E, :logp]
    names = [param_names; internal_names]

    # Set up the info tuple.
    if save_state
        info = (target=target, sampler=sampler, state=state)
    else
        info = NamedTuple()
    end

    # Merge in the timing info, if available
    if !ismissing(stats)
        info = merge(info, (start_time=stats.start, stop_time=stats.stop))
    end

    # Conretize the array before giving it to MCMCChains.
    samples = MCMCChains.concretize(samples)

    # Chain construction.
    chain = MCMCChains.Chains(
        samples,
        names,
        (internals = internal_names,);
        info=info,
        start=burn_in + 1,
        thin=thinning)

    return chain
end