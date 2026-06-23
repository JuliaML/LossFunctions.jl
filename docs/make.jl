using Documenter, LossFunctions

using LossFunctions.Traits

makedocs(
  modules=[LossFunctions, LossFunctions.Traits],
  format=Documenter.HTML(
    prettyurls=get(ENV, "CI", "false") == "true",
    assets=["assets/style.css", "assets/favicon.ico"]
  ),
  sitename="LossFunctions.jl",
  authors="Christof Stocker, Tom Breloff, Alex Williams, Júlio Hoffimann",
  pages=[
    "Home" => "index.md",
    "Introduction" => ["introduction/gettingstarted.md", "introduction/motivation.md"],
    "User's Guide" => ["user/interface.md", "user/aggregate.md"],
    "Available Losses" => ["losses/distance.md", "losses/margin.md", "losses/other.md"],
    "Advanced Topics" => ["advanced/extend.md", "advanced/developer.md"],
    "Indices" => "indices.md",
    "acknowledgements.md",
    "LICENSE.md"
  ]
)

deploydocs(;
  repo="github.com/JuliaML/LossFunctions.jl.git",
  versions=["stable" => "v^", "dev" => "dev"]
)
