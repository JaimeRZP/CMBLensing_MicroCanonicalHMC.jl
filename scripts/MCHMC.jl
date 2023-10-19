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
t = 0.15 #nothing
precond_path = string("../chains/pixel_preconditioners/pp_nside_512_t_", t)
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

if t == nothing
    precond = one(simulate(Diagonal(one(LenseBasis(diag(prob.Λmass))))));
else
    precond = load(precond_path, "dist_mat_precond")
    precond = adapt(CuArray, precond)
    precond = from_vec(precond);
end

#Target
target = CMBLensingTarget(prob);

#Sampler
TEV = 0.0001
samples = 100
spl = MCHMC(samples, TEV; adaptive=true, init_eps=30, init_L=500, sigma=precond);
fol_name=string("/pscratch/sd/j/jaimerz/chains/test/MCHMC",
    "_cosmo_", global_parameters,
    "_masking_", masking,
    "_Nside_", Nside,
    "_precond_", t,
    "_TEV_", TEV)

if isdir(fol_name)
    fol_files = readdir(fol_name)
    println("Found existing file ", fol_name)
    if length(fol_files) != 0
        last_chain = last([file for file in fol_files if occursin("chain", file)])
        last_n = parse(Int, last_chain[end])
        last_chain = load(string(fol_name, "/", last_chain), "samples")
        init_params = last_chain[:, end]
        println("Restarting chain")
    else
        println("Starting new chain")
        last_n = 0
        init_params = prob.Ωstart
    end
else
    mkdir(fol_name)
    println(string("Created new folder ", fol_name))
    last_n = 0
end

file_name = string(fol_name, "/chain_", last_n+1, "_", samples)

samples_mchmc = Sample(spl, target, samples, dialog=false, progress=true,
                       thinning=20, file_name=file_name);
