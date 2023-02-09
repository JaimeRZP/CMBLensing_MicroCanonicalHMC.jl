#=
function ess_corr(samples)
    """Taken from: https://blackjax-devs.github.io/blackjax/diagnostics.html
        shape(x) = (num_samples, d)"""
    input_array = samples.Ω
    num_samples = length(input_array)
    mean_across_chain = mean(input_array)
    # Compute autocovariance estimates for every lag for the input array using FFT.
    centered_array = mapreduce(permutedims, vcat, input_array - mean_across_chain)
    #m = next_fast_len(2 * num_samples)
    ifft_ary = fft(centered_array)
    ifft_ary *= conj(ifft_ary)
    autocov_value = ifft(ifft_ary, n=m, axis=1)
    autocov_value ./ num_samples #(jnp.take(autocov_value, jnp.arange(num_samples), axis=1) / num_samples)
    mean_autocov_var = mean(autocov_value)
    #mean_var0 = (jnp.take(mean_autocov_var, jnp.array([0]), axis=1) * num_samples / (num_samples - 1.0))
    #weighted_var = mean_var0 * (num_samples - 1.0) / num_samples
    #weighted_var = jax.lax.cond(num_chains > 1,
    #                            lambda _: weighted_var+ mean_across_chain.var(axis=0, ddof=1, keepdims=True),
    #                            lambda _: weighted_var,
    #                            operand=None)
    # Geyer's initial positive sequence
    num_samples_even = num_samples - mod(num_samples, 2)
    mean_autocov_var_tp1 = mean_auto_cov[1:num_samples_even] #jnp.take(mean_autocov_var, jnp.arange(1, num_samples_even), axis=1)
    rho_hat = jnp.concatenate([jnp.ones_like(mean_var0), 1.0 - (mean_var0 - mean_autocov_var_tp1) / weighted_var,], axis=1,)
    rho_hat = jnp.moveaxis(rho_hat, 1, 0)
    rho_hat_even = rho_hat[0::2]
    rho_hat_odd = rho_hat[1::2]
    mask0 = (rho_hat_even + rho_hat_odd) > 0.0
    carry_cond = jnp.ones_like(mask0[0])
    max_t = jnp.zeros_like(mask0[0], dtype=int)
    def positive_sequence_body_fn(state, mask_t):
        t, carry_cond, max_t = state
        next_mask = carry_cond & mask_t
        next_max_t = jnp.where(next_mask, jnp.ones_like(max_t) * t, max_t)
        return (t + 1, next_mask, next_max_t), next_mask
    (*_, max_t_next), mask = jax.lax.scan(
        positive_sequence_body_fn, (0, carry_cond, max_t), mask0
    )
    indices = jnp.indices(max_t_next.shape)
    indices = tuple([max_t_next + 1] + [indices[i] for i in range(max_t_next.ndim)])
    rho_hat_odd = jnp.where(mask, rho_hat_odd, jnp.zeros_like(rho_hat_odd))
    # improve estimation
    mask_even = mask.at[indices].set(rho_hat_even[indices] > 0)
    rho_hat_even = jnp.where(mask_even, rho_hat_even, jnp.zeros_like(rho_hat_even))
    # Geyer's initial monotone sequence
    def monotone_sequence_body_fn(rho_hat_sum_tm1, rho_hat_sum_t):
        update_mask = rho_hat_sum_t > rho_hat_sum_tm1
        next_rho_hat_sum_t = jnp.where(update_mask, rho_hat_sum_tm1, rho_hat_sum_t)
        return next_rho_hat_sum_t, (update_mask, next_rho_hat_sum_t)
    rho_hat_sum = rho_hat_even + rho_hat_odd
    _, (update_mask, update_value) = jax.lax.scan(
        monotone_sequence_body_fn, rho_hat_sum[0], rho_hat_sum
    )
    rho_hat_even_final = jnp.where(update_mask, update_value / 2.0, rho_hat_even)
    rho_hat_odd_final = jnp.where(update_mask, update_value / 2.0, rho_hat_odd)
    # compute effective sample size
    ess_raw = num_chains * num_samples
    tau_hat = (
        -1.0
        + 2.0 * jnp.sum(rho_hat_even_final + rho_hat_odd_final, axis=0)
        - rho_hat_even_final[indices]
    )
    tau_hat = jnp.maximum(tau_hat, 1 / np.log10(ess_raw))
    ess = ess_raw / tau_hat
    ### my part (combine all dimensions): ###
    neff = ess.squeeze() / num_samples
    return 1.0 / mean(1 / neff)
end
=#

function tune_eps(sampler::Sampler, target::Target, init; kwargs...)
    dialog = get(kwargs, :dialog, false)
    sett = sampler.settings
    eps = sampler.hyperparameters.eps
    L = sampler.hyperparameters.L
    varE_wanted = sett.varE_wanted
    #x, u, g, time = init

    samples = []
    for i in 1:sett.tune_samples
        init, sample = Step(sampler, target, init;
                            monitor_energy=true)
        push!(samples, sample)
    end

    samples = mapreduce(permutedims, vcat, samples)
    E = samples[:, end-1]
    varE = std(E)^2 / target.d #variance per dimension
    if dialog
        println("eps: ", eps, " --> VarE: ", varE)
    end
    no_divergences = isfinite(varE)

    ### update the hyperparameters ###
    if no_divergences
        success = varE < varE_wanted #we are done
        if !success
            #eps_new = eps*(varE_wanted/varE)^0.25 #assume var[E] ~ eps^4
            sampler.hyperparameters.eps = 0.5 * eps
        else
            @info "Found eps ✅"
        end
    else
        success = false
        sampler.hyperparameters.eps = 0.5 * eps
    end

    return success
end

function tune_hyperparameters(sampler::Sampler, target::Target, init; kwargs...)
    sett = sampler.settings

    ### debugging tool ###
    dialog = get(kwargs, :dialog, false)

    # Init guess
    if sampler.hyperparameters.eps == 0.0
        if dialog
            @info "Tuning eps ⏳"
        end
        sampler.hyperparameters.eps = 0.5
        for i in 1:sett.tune_maxiter
            if tune_eps(sampler, target, init; kwargs...)
                break
            end
        end

    #TODO: is this an elseif or an if?
    elseif sampler.hyperparameters.L == 0.0
        if dialog
            @info "Tuning L ⏳"
        end
        sampler.hyperparameters.L = sqrt(target.d)
        steps = 10 .^ (LinRange(2, log10(2500), sett.tune_maxiter))
        steps = Int.(round.(steps))
        samples = []
        for s in steps
            for i in 1:s
                init, sample = Step(sampler, target, init; monitor_energy=true)
                push!(samples, sample)
            end
            ESS = ess_corr(samples)
            if dialog
                println(string("n = ", n[1], "ESS = ", ESS))
            end
            if n[i] > 10.0 / ESS
                break
            end
        end
        L = 0.4 * eps / ESS # = 0.4 * correlation length
        if dialog
            println(string("L / sqrt(d) = ", L / sqrt(target.d),
                           "ESS(correlations) = ", ESS))
            println("-------------")
        end

    else
        if dialog
            println("Using given hyperparameters")
        end
    end

    tune_nu!(sampler, target)
end

function eval_nu(eps, L, d)
    nu = sqrt((exp(2 * eps / L) - 1.0) / d)
    return nu
end

function tune_nu!(spl::Sampler, trg::Target)
    spl.hyperparameters.nu = eval_nu(spl.hyperparameters.eps, spl.hyperparameters.L, trg.d)
end
