module MicroCanonicalHMC

export Settings, Hyperparameters, mchmc, Sample
export Leapfrog, Minimal_norm
export TuringTarget, StandardGaussianTarget, CustomTarget, CMBLensingTarget

using Interpolations, LinearAlgebra, Statistics, Distributions, Random, DataFrames
using DynamicPPL, Turing, LogDensityProblemsAD, LogDensityProblems, ForwardDiff, Zygote
using AbstractMCMC, MCMCChains

abstract type Target <: AbstractMCMC.AbstractModel end

include("sampler.jl")
include("targets.jl")
include("tuning.jl")
include("integrators.jl")
include("abstractmcmc.jl")


end
