#=
function ess_corr(x)
    """Taken from: https://blackjax-devs.github.io/blackjax/diagnostics.html
        shape(x) = (num_samples, d)"""
    input_array = [x, ]

    num_chains = 1 #input_array.shape[0]
    num_samples = input_array.shape[1]

    mean_across_chain = input_array.mean(axis=1, keepdims=True)
    # Compute autocovariance estimates for every lag for the input array using FFT.
    centered_array = input_array - mean_across_chain
    m = next_fast_len(2 * num_samples)
    ifft_ary = jnp.fft.rfft(centered_array, n=m, axis=1)
    ifft_ary *= jnp.conjugate(ifft_ary)
    autocov_value = jnp.fft.irfft(ifft_ary, n=m, axis=1)
    autocov_value = (
        jnp.take(autocov_value, jnp.arange(num_samples), axis=1) / num_samples
    )
    mean_autocov_var = autocov_value.mean(0, keepdims=True)
    mean_var0 = (jnp.take(mean_autocov_var, jnp.array([0]), axis=1) * num_samples / (num_samples - 1.0))
    weighted_var = mean_var0 * (num_samples - 1.0) / num_samples
    weighted_var = jax.lax.cond(num_chains > 1,
                                lambda _: weighted_var+ mean_across_chain.var(axis=0, ddof=1, keepdims=True),
                                lambda _: weighted_var,
                                operand=None)

    # Geyer's initial positive sequence
    num_samples_even = num_samples - num_samples % 2
    mean_autocov_var_tp1 = jnp.take(mean_autocov_var, jnp.arange(1, num_samples_even), axis=1)
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
    return 1.0 / jnp.average(1 / neff)
end
=#

function _tuning(init, sampler::Sampler, target::Target, props)
    sett = sampler.settings
    eps_inappropriate, eps_appropriate, success = props

    # get a small number of samples
    x, u, g, time = init
    E = zeros(eltype(x), length(x), sett.tune_samples)
    E = Vector{eltype(samples)}[eachcol(samples)...]
    for i in 1:sett.tune_samples
        init, sample = Step(sampler, target, init)
        E[i] = sample
    end

    # remove large jumps in the energy
    E -= mean(E)
    E = remove_jumps(E)

    ### compute quantities of interest ###

    # typical size of the posterior
    x1 = mean(X, axis= 0) #first moments
    x2 = mean(square.(X), axis=0) #second moments
    sigma = sqrt.(mean(x2 - square.(x1))) #average variance over the dimensions

    # energy fluctuations
    varE = std(E)^2 / target.d #variance per dimension
    no_divergences = isfinite(varE)

    ### update the hyperparameters ###

    if no_divergences
        L_new = sigma * sqrt(target.d)
        eps_new = sett.eps * (varE_wanted / varE)^0.25 #assume var[E] ~ eps^4
        success = abs(1.0 - varE / varE_wanted) < 0.2 #we are done
    else
        L_new = self.L
        if sett.eps < eps_inappropriate
            eps_inappropriate = sett.eps
        end
    end
        eps_new = Inf #will be lowered later


    #update the known region of appropriate eps

    if not no_divergences # inappropriate epsilon
        if sett.eps < eps_inappropriate #it is the smallest found so far
            eps_inappropriate = sett.eps
        end
    else # appropriate epsilon
        if sett.eps > eps_appropriate #it is the largest found so far
            eps_appropriate = sett.eps
        end
    end

    # if suggested new eps is inappropriate we switch to bisection
    if eps_new > eps_inappropriate
        eps_new = 0.5 * (eps_inappropriate + eps_appropriate)
    end

    self.set_hyperparameters(L_new, eps_new)

    #if dialog
    #    word = 'bisection' if (not no_divergences) else 'update'
    #    print('varE / varE wanted: {} ---'.format(np.round(varE / varE_wanted, 4)) + word + '---> eps: {}, sigma = L / sqrt(d): {}'.format(np.round(eps_new, 3), np.round(L_new / np.sqrt(self.Target.d), 3)))

    return eps_inappropriate, eps_appropriate, success
end

function tune_hyperparameters(init, sampler::Sampler, target::Target; kwargs...)
    sett = sampler.settings

    # targeted energy variance per dimension
    varE_wanted = sett.tune_varE_wanted
    burn_in = sett.tune_burn_in
    samples = sett.tune_samples

    ### debugging tool ###
    dialog = get(kwargs, :dialog, false)

    ### random key ###
    key = sett.key

    init_eps, init_L = sqrt(target.d), 0.6
    for i in 1:sett.tune_burn_in
        init, x0 = Step(sampler, target, init)
    end

    props = (Inf, 0.0, false)

    if dialog
        println("Hyperparameter tuning (first stage)")
    end

    ### first stage: L = sigma sqrt(d)  ###
    ### LEFT HERE
    for i in 1:sett.tune_maxiter
        props = _tuning(init, sampler, target, props)
        if props[end] # success == True
            break
        end
    end

    ### second stage: L = epsilon(best) / ESS(correlations)  ###
    if dialog
        println("Hyperparameter tuning (second stage)")
    end

    n = 10 .^ (LinRange(2, log10(2500), 6))
    n = append!([2.0], n)
    n = Int.(round.(n))

    X = [zeros(Float64, target.d) for i in 1:n[end]]
    X[1] = x0
    for i in 2:length(n)
        init = X[n[i-1]-1]
        for j in n[i-1]:n[i]
            init, X[j] = Step(sampler, target, init; monitor_energy=true)
        end
        ESS = ess_corr(X[:n[i]])
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

    return eps, L
end