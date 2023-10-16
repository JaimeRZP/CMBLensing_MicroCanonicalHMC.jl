using Pkg
Pkg.activate("../examples")
Pkg.instantiate()

using Revise, Adapt, CMBLensing, CMBLensingInferenceTestProblem, CUDA,
    JLD2, LaTeXStrings, LinearAlgebra, 
    MCMCChains, MCMCDiagnosticTools, MicroCanonicalHMC, MuseInference, MuseInference, Plots, 
    ProgressMeter, Random, Statistics, Zygote

Nside = 512
T = Float64;
masking = true
global_parameters = true
precond_path = "../chains/pixel_preconditioners/pp_nside_512_t_0.13" #nothing 
println("Nside: ", Nside)
println("Masking: ", masking)
println("Global_parameters: ", global_parameters)
println("Precond: ", precond_path)

prob = load_cmb_lensing_problem(;storage=CuArray, T, Nside,
    masking=masking, global_parameters=global_parameters);
d = length(prob.Ωstart)
to_vec, from_vec = CMBLensingInferenceTestProblem.to_from_vec(prob.Ωstart);
println("Built problem")

#Precond
prob.Λmass.diag.θ.r *= 5.85
prob.Λmass.diag.θ.Aϕ *= 112.09

if precond_path === nothing
    use_precond = false
    precond = one(simulate(Diagonal(one(LenseBasis(diag(prob.Λmass))))));
else
    use_precond = true
    precond = load(precond_path, "dist_mat_precond")
    precond = adapt(CuArray, precond)
    precond = from_vec(precond);
end

#Sampler
ϵ=0.005
Ω = prob.Ωstart
samples_hmc = []
rng = Xoshiro(1)
prob.ncalls[] = 0

iterations = 5
@showprogress for i=1:iterations
    Ω, = state = hmc_step(rng, prob, prob.Ωstart, prob.Λmass; symp_kwargs=[(N=25, ϵ=ϵ)], progress=false, always_accept=(i<10))
    push!(samples_hmc, adapt(Array, state))
end
ncalls_hmc = prob.ncalls[]
println("N_calls: ", ncalls_hmc)

_samples_hmc = zeros(iterations, 3*Nside^2+2)
for i in 1:iterations
    _samples_hmc[i, :]  = samples_hmc[i][1][:]
end

file_name=string("/pscratch/sd/j/jaimerz/chains/test/CMBLensing",
    "_cosmo_", global_parameters,
    "_masking_", masking,
    "_Nside_", Nside,
    "_precond_", use_precond,
    "_ϵ_", ϵ)
@save file_name _samples_hmc
