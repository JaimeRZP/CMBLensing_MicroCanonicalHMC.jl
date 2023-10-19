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

#Sampler
ϵ=0.0025
Ω = prob.Ωstart
samples_hmc = []
rng = Xoshiro(1)
prob.ncalls[] = 0

deriv_precond = DerivBasis(precond)
new_f = sqrt.(real(pinv(Diagonal(EBFourier(deriv_precond.f°)))*prob.Λmass[:f°]*conj.(pinv.(Diagonal(EBFourier(deriv_precond.f°))))))
deriv_precond.θ.r = 5.85
deriv_precond.θ.Aϕ = 112.09
Λmass_new = Diagonal(FieldTuple(f°=diag(new_f), ϕ°=diag(prob.Λmass[:ϕ°]), θ=deriv_precond.θ));

iterations = 10_000
@showprogress for i=1:iterations
    Ω, = state = hmc_step(rng, prob, prob.Ωstart, new_Λmass; symp_kwargs=[(N=25, ϵ=ϵ)], progress=false, always_accept=(i<10))
    push!(samples_hmc, adapt(Array, state))
end
ncalls_hmc = prob.ncalls[]
println("N_calls: ", ncalls_hmc)

_samples_hmc = zeros(iterations, 3*Nside^2+2)
for i in 1:iterations
    _samples_hmc[i, :]  = samples_hmc[i][1][:]
end

fol_name=string("/pscratch/sd/j/jaimerz/chains/HMC/HMC",
    "_cosmo_", global_parameters,
    "_masking_", masking,
    "_Nside_", Nside,
    "_ϵ_", ϵ)

file_name = string(fol_name, "/chain_", last_n+1, "_", samples)

@save file_name _samples_hmc
