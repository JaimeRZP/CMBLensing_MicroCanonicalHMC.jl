struct Hamiltonian
    ℓπ
    ∂lπ∂θ
end 

function Hamiltonian(model::AbstractMCMC.LogDensityModel)
    logdensity = model.logdensity
    ℓπ(x) = -LogDensityProblems.logdensity(ℓ, x)
    ∂lπ∂θ(x) = -LogDensityProblems.logdensity_and_gradient(ℓ, x)
    return Hamiltonian(ℓπ, ∂lπ∂θ)
end
