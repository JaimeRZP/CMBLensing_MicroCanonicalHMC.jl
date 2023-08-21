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
file_path = 
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

#Target
target = CMBLensingTarget(prob);

#Sampler
TEV = 0.0001
spl = MCHMC(500, TEV; adaptive=true, init_eps=30, init_L=500, sigma=precond);
file_name=string("/pscratch/sd/j/jaimerz/chains/MCHMC/CMBLensing",
    "_cosmo_", global_parameters,
    "_masking_", masking,
    "_Nside_", Nside,
    "_precond_", use_precond,
    "_TEV_", TEV)
samples_mchmc = Sample(spl, target, 10_000, dialog=false, progress=true,
                       thinning=20, file_name=file_name);