function AbstractMCMC.sample(model::DynamicPPL.Model,
                             sampler::AbstractMCMC.AbstractSampler,
                             ::MCMCThreads,
                             N::Integer,
                             nchains::Integer;
                             resume_from=nothing,
                             kwargs...)
    # Get target
    if resume_from === nothing
        target = TuringTarget(model)
        tune_hyperparameters(sampler, target, state; kwargs...)
    else
        target =  chain.info[:target]
        chain = chain.info[:sampler]
    end

    # Start parallelization
    nchains = Threads.nthreads()
    interval = 1:nchains
    targets = [deepcopy(target) for _ in interval]
    samplers = [deepcopy(sampler) for _ in interval]

    @AbstractMCMC.ifwithprogresslogger progress name = progressname begin
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
                            AbstractMCMC.ProgressLogging.@logprogress progresschains / nchains
                            nextprogresschains = progresschains + threshold
                        end
                    end
                end
            end

            Distributed.@async begin
                try
                    Distributed.@sync for (_target, _sampler) in
                                          zip(targets, samplers)

                        Threads.@spawn for i in interval
                            # Sample a chain and save it to the vector.
                            chains[chainidx] = AbstractMCMC.mcmcsample(
                                _target,
                                _sampler,
                                N;
                                progress=false,
                                kwargs...)

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

    return chains
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

    # Ensure that initial parameters are `nothing` or indexable
    _init_params = _first_or_nothing(init_params, nchains)

    # Set up a chains vector.
    chains = Vector{Any}(undef, nchains)

    @AbstractMCMC.ifwithprogresslogger progress name = progressname begin
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
                            AbstractMCMC.ProgressLogging.@logprogress progresschains / nchains
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