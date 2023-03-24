# make.jl
using Documenter, GaussianProcess

makedocs(sitename = "MicroCanonicalHMC.jl",
         modules = [MiCroCanonicalHMC],
         pages = ["Home" => "index.md",
                  "API" => "api.md"])
deploydocs(
    repo = "github.com/JaimeRZP/MicroCanonicalHMC.jl"
)
