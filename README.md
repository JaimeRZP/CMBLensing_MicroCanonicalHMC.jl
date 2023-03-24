# MicroCanonicalHMC.jl

[![Build Status](https://github.com/JaimeRZP/MCHMC.jl/workflows/CI/badge.svg)](https://github.com/JaimeRZP/MicroCanonicalHMC.jl/actions?query=workflow%3AMCHMC-CI+branch%3Amaster)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jaimerzp.github.io/MicroCanonicalHMC.jl/dev/)
![size](https://img.shields.io/github/repo-size/jaimerzp/MicroCanonicalHMC.jl)

![](https://raw.githubusercontent.com/JaimeRZP/MicroCanonicalHMC.jl/master/docs/src/assets/mchmc_logo.png)

A Julia implementation of [Micro-Canonical HMC](https://arxiv.org/pdf/2212.08549.pdf). You can checkout the JAX version [here](https://github.com/JakobRobnik/MicroCanonicalHMC). 

## How to use it

### Define the Model
Start by drawing a Neal's funnel model in Turing.jl:

```julia
# The statistical inference frame-work we will use
using Turing
using Random
using PyPlot
using LinearAlgebra
using MicroCanonicalHMC

d = 21
@model function funnel()
    θ ~ Normal(0, 3)
    z ~ MvNormal(zeros(d-1), exp(θ)*I)
    x ~ MvNormal(z, I)
end
```
### Define the Target
Wrap the Turing model inside a MicrocanonicalHMC.jl target:

```julia
target = TuringTarget(funnel_model; d=d, compute_MAP=false)
```


### Define the Sampler

```julia
spl = MCHMC(0.0, 0.0; varE_wanted=0.001, sigma=ones(d))
ensemble_spl = MCHMC(0.0, 0.0, 10; varE_wanted=0.001, sigma=ones(d))
```
The first two entries mean that the step size and the trajectory length will be self-tuned. In the ensemble sampler, the third number represents the number of workers.
`VaE_wanted` sets the hamiltonian error per dimension that will be targeted. Fixing `sigma=ones(d)` avoids tunin the preconditioner.

### Start Sampling

```julia
samples_mchmc = Sample(spl, target, 50_000; burn_in=5_000, dialog=false)
samples_mchmc_ensemble = Sample(ensemble_spl, target, 50_000; burn_in=5_000, dialog=false)
```

### Compare to NUTS

```julia
samples_hmc = sample(funnel_model, NUTS(5_000, 0.95), 50_000; progress=true, save_state=true)
```

![](https://raw.githubusercontent.com/JaimeRZP/MicroCanonicalHMC.jl/master/docs/src/assets/Neal_funnel_comp.png)


## Using MicroCanonicalHMC.jl with AbstractMCMC.jl

```julia
samples_hmc = sample(funnel_model, spl, 50_000; progress=true, save_state=true)
```

Note that we are passing the `Turing` model directly instead of the `Target` object
