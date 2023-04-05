module MicroCanonicalHMC


using AbstractMCMC, Adapt, CMBLensing, Distributed, Distributions, 
    DocStringExtensions, ForwardDiff, HDF5, Interpolations, LinearAlgebra, 
    MCMCDiagnosticTools, Markdown, ProgressMeter, Random, Statistics, Zygote

export CMBLensingTarget, CustomTarget, Hyperparameters, Leapfrog, MCHMC, 
    Minimal_norm, ParallelTarget, Sample, Settings, StandardGaussianTarget

abstract type Target end

include("sampler.jl")
include("targets.jl")
include("tuning.jl")
include("integrators.jl")
include("chains.jl")
include("CMBLensing_utils.jl")

include("ensemble/targets.jl")
include("ensemble/sampler.jl")
include("ensemble/integrators.jl")
include("ensemble/tuning.jl")
end
