module MicroCanonicalHMC

export Settings, Hyperparameters, MCHMC, Sample
export Leapfrog, Minimal_norm
export StandardGaussianTarget, CustomTarget, CMBLensingTarget
export ParallelTarget

using Interpolations, LinearAlgebra, Statistics
using Distributions, Random, ForwardDiff, Distributed
using CMBLensing, Zygote, MCMCDiagnosticTools, AbstractMCMC

abstract type Target end

include("sampler.jl")
include("targets.jl")
include("tuning.jl")
include("integrators.jl")
include("CMBLensing_utils.jl")

include("ensemble/targets.jl")
include("ensemble/sampler.jl")
include("ensemble/integrators.jl")
include("ensemble/tuning.jl")
end
