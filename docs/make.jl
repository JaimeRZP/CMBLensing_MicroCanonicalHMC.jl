# make.jl
using Documenter, MicroCanonicalHMC

makedocs(sitename = "MicroCanonicalHMC.jl",
         modules = [GaussianProcess],
         pages = ["Home" => "index.md",
                  "API" => "api.md"])
deploydocs(
    repo = "github.com/JaimeRZP/MicroCanonicalHCM.jl"
)
