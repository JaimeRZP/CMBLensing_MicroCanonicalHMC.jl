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

#Sampler
muse_prob = CMBLensingMuseProblem(
    prob.ds, 
    MAP_joint_kwargs = (minsteps=3, nsteps=15, αtol=1e-2, gradtol=3e-5, progress=false, history_keys=(:logpdf, :ΔΩ°_norm)),
);

# small hack to allow getting MUSE covariance in terms of transformed θ
CMBLensingMuseInferenceExt = Base.get_extension(CMBLensing,:CMBLensingMuseInferenceExt)
CMBLensingMuseInferenceExt.mergeθ(prob::CMBLensingMuseInferenceExt.CMBLensingMuseProblem, θ) = exp.(θ)

z₀ = zero(MuseInference.sample_x_z(muse_prob, Xoshiro(0), prob.Ωstart.θ).z);
result = MuseResult()
nsims = 200
rng = Xoshiro(0)

prob.ncalls[] = 0
MuseInference.muse!(result,  muse_prob, prob.Ωstart.θ; nsims, rng, z₀, maxsteps=2, θ_rtol=0, progress=true, save_MAPs=false)
MuseInference.get_J!(result, muse_prob; nsims,   rng, z₀, progress=true)
MuseInference.get_H!(result, muse_prob; nsims=4, rng, z₀, progress=true, step=std(result.gs)/100, fdm=central_fdm(2,1,adapt=0))
ncalls_muse = prob.ncalls[];

chain_muse = Chains(permutedims(rand(result.dist,5_000)), [:logr, :logAϕ]);
file_name=string("/pscratch/sd/j/jaimerz/chains/MUSE/CMBLensing",
    "_cosmo_", global_parameters,
    "_masking_", masking,
    "_Nside_", Nside,
    "_precond_", use_precond,
    "_nsims_", nsims)
@save string("/pscratch/sd/j/jaimerz/chains/MUSE/CMBLensing_masking_", masking, "_Nside_", Nside) chain_muse