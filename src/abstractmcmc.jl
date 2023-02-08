
const PROGRESS = Ref(true)

function AbstractMCMC.step(sampler::Sampler, target::Target, state; kwargs...)
    return Step(sampler::Sampler, target::Target, state; kwargs...)
end

function AbstractMCMC.sample(model::DynamicPPL.Model,
                             sampler::AbstractMCMC.AbstractSampler,
                             N::Int;
                             kwargs...)
    # Get target
    target = TuringTarget(model)
    return AbstractMCMC.mcmcsample(target, sampler, N; kwargs...)
end

function AbstractMCMC.mcmcsample(target::AbstractMCMC.AbstractModel,
                                 sampler::AbstractMCMC.AbstractSampler,
                                 N::Integer;
                                 progress=PROGRESS[],
                                 progressname="Sampling",
                                 callback=nothing,
                                 discard_initial=0,
                                 thinning=1,
                                 chain_type::Type=AbstractMCMC.AbstractChains,
                                 kwargs...)
    # Check the number of requested samples.
    N > 0 || error("the number of samples must be ≥ 1")
    Ntotal = thinning * (N - 1) + discard_initial + 1

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

        # Discard initial samples.
        for i in 1:discard_initial
            # Update the progress bar.
            if progress && i >= next_update
                AbstractMCMC.ProgressLogging.@logprogress i / Ntotal
                next_update = i + threshold
            end

            # Obtain the next sample and state.
            state, sample = AbstractMCMC.step(sampler, target, state; kwargs...)
        end

        # Run callback.
        # WTF is this?
        #callback === nothing || callback(rng, model, sampler, sample, state, 1; kwargs...)

        # Save the sample.
        samples = AbstractMCMC.samples(sample, target, sampler, N; kwargs...)
        samples = AbstractMCMC.save!!(samples, sample, 1, target, sampler, N; kwargs...)

        # Update the progress bar.
        itotal = 1 + discard_initial
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
            # WTF does this do?
            #callback === nothing ||
            #    callback(rng, model, sampler, sample, state, i; kwargs...)

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

    return samples
    #=
    return AbstractMCMC.bundle_samples(samples,
                                       target,
                                       sampler,
                                       state,
                                       chain_type;
                                       stats=stats,
                                       discard_initial=discard_initial,
                                       thinning=thinning,
                                       kwargs...)
    =#
end

function _init_samples(target::Target, extra_vars)
    df = DataFrame()

    for vsym in target.vsyms
        df[!, Symbol(vsym)] = Any[]
    end

    for var in extra_vars
        df[!, var] = Any[]
    end

    return df
end

function _push_samples(df::DataFrame, sample)
    params, stats = sample
    push!(samples, [params; stats])
end

#=
function AbstractMCMC.bundle_samples(
    vals::Vector,
    model::AbstractModel,
    spl::Union{Sampler{<:InferenceAlgorithm},SampleFromPrior},
    state,
    chain_type::Type{MCMCChains.Chains};
    save_state = false,
    stats = missing,
    discard_initial = 0,
    thinning = 1,
    kwargs...
)
    # Convert transitions to array format.
    # Also retrieve the variable names.

    # Get the values of the extra parameters in each transition.
    extra_params, extra_values = get_transition_extras(ts)

    # Extract names & construct param array.
    nms = [nms; extra_params]
    parray = hcat(vals, extra_values)

    # Get the average or final log evidence, if it exists.
    le = getlogevidence(ts, spl, state)

    # Set up the info tuple.
    if save_state
        info = (model = model, sampler = spl, samplerstate = state)
    else
        info = NamedTuple()
    end

    # Merge in the timing info, if available
    if !ismissing(stats)
        info = merge(info, (start_time=stats.start, stop_time=stats.stop))
    end

    # Conretize the array before giving it to MCMCChains.
    parray = MCMCChains.concretize(parray)

    # Chain construction.
    chain = MCMCChains.Chains(
        parray,
        nms,
        (internals = extra_params,);
        evidence=le,
        info=info,
        start=discard_initial + 1,
        thin=thinning,
    )

    return chain
end
=#