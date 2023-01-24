function _tuning_step(props)

        key, eps_inappropriate, eps_appropriate, success = props

        # get a small number of samples
        key_new, subkey = jax.random.split(key)
        X, E = self.sample(samples, x0, subkey, monitor_energy= True)

        # remove large jumps in the energy
        E -= jnp.average(E)
        E = remove_jumps(E)

        ### compute quantities of interest ###

        # typical size of the posterior
        x1 = jnp.average(X, axis= 0) #first moments
        x2 = jnp.average(jnp.square(X), axis=0) #second moments
        sigma = jnp.sqrt(jnp.average(x2 - jnp.square(x1))) #average variance over the dimensions

        # energy fluctuations
        varE = jnp.std(E)**2 / self.Target.d #variance per dimension
        no_divergences = np.isfinite(varE)

        ### update the hyperparameters ###

        if no_divergences:
            L_new = sigma * jnp.sqrt(self.Target.d)
            eps_new = self.eps * jnp.power(varE_wanted / varE, 0.25) #assume var[E] ~ eps^4
            success = jnp.abs(1.0 - varE / varE_wanted) < 0.2 #we are done

        else:
            L_new = self.L

            if self.eps < eps_inappropriate:
                eps_inappropriate = self.eps
            end
        end
            eps_new = jnp.inf #will be lowered later


        #update the known region of appropriate eps

        if not no_divergences: # inappropriate epsilon
            if self.eps < eps_inappropriate: #it is the smallest found so far
                eps_inappropriate = self.eps
            end
        else: # appropriate epsilon
            if self.eps > eps_appropriate: #it is the largest found so far
                eps_appropriate = self.eps
            end
        end

        # if suggested new eps is inappropriate we switch to bisection
        if eps_new > eps_inappropriate:
            eps_new = 0.5 * (eps_inappropriate + eps_appropriate)
        end

        self.set_hyperparameters(L_new, eps_new)

        #if dialog:
        #    word = 'bisection' if (not no_divergences) else 'update'
        #    print('varE / varE wanted: {} ---'.format(np.round(varE / varE_wanted, 4)) + word + '---> eps: {}, sigma = L / sqrt(d): {}'.format(np.round(eps_new, 3), np.round(L_new / np.sqrt(self.Target.d), 3)))

        return key_new, eps_inappropriate, eps_appropriate, success
end

function tune_hyperparameters(sett:Settings, target::Target; kwargs...)
        kwargs = Dict(kwargs)

        # targeted energy variance per dimension
        varE_wanted = get(kwargs, :varE_wanted, 0.0005)
        burn_in = get(kwargs, :burn_in, 2000)
        samples = get(kwargs, :samples, 1000)

        ### debugging tool ###
        dialog = get(kwargs, :dialog, false)

        ### random key ###
        key = sett.key

        init_eps, init_L = np.sqrt(target.d), 0.6
        init_nu = sqrt((exp(2 * init_eps / init_L) - 1.0) / target.d)

        x0 = sample(burn_in, x_initial, random_key=subkey, final_state=true)
        props = (Inf, 0.0, false)

        if dialog:
            print('Hyperparameter tuning (first stage)')
        end

        ### first stage: L = sigma sqrt(d)  ###
        for i in 1:10 # = maxiter
            props = tuning_step(props)
            if props[end]: # success == True
                break
            end
        end

        ### second stage: L = epsilon(best) / ESS(correlations)  ###
        if dialog:
            print('Hyperparameter tuning (second stage)')
        end

        n = np.logspace(2, np.log10(2500), 6).astype(int) # = [100, 190, 362, 689, 1313, 2499]
        n = np.insert(n, [0, ], [1, ])
        X = np.empty((n[-1] + 1, self.Target.d))
        X[0] = x0
        for i in 1:length(n):
            X[n[i-1]:n[i]] = self.sample(n[i] - n[i-1], x_initial= X[n[i-1]-1], random_key= subkey, monitor_energy=True)[0]
            ESS = ess_corr(X[:n[i]])
            if dialog:
                print('n = {0}, ESS = {1}'.format(n[i], ESS))
            if n[i] > 10.0 / ESS:
                break

        L = 0.4 * self.eps / ESS # = 0.4 * correlation length
        self.set_hyperparameters(L, self.eps)

        if dialog:
            print('L / sqrt(d) = {}, ESS(correlations) = {}'.format(L / np.sqrt(self.Target.d), ESS))
            print('-------------')