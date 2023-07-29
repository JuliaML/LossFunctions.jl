using Documenter, LossFunctions

makedocs(
  modules=[LossFunctions, LossFunctions.Traits],
  authors="Christof Stocker, Tom Breloff, Alex Williams",
  repo="https://github.com/JuliaML/LossFunctions.jl/blob/{commit}{path}#{line}",
  sitename="LossFunctions.jl",
  format=Documenter.HTML(
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://JuliaML.github.io/LossFunctions.jl",
    assets=["assets/style.css", "assets/favicon.ico"]
  ),
  pages=[
    hide("Home" => "index.md"),
    "Introduction" => ["introduction/gettingstarted.md", "introduction/motivation.md"],
    "User's Guide" => ["user/interface.md", "user/aggregate.md"],
    "Available Losses" => ["losses/distance.md", "losses/margin.md", "losses/other.md"],
    "Advances Topics" => ["advanced/extend.md", "advanced/developer.md"],
    hide("Indices" => "indices.md"),
    "acknowledgements.md",
    "LICENSE.md"
  ]
)

deploydocs(repo="github.com/JuliaML/LossFunctions.jl.git", devbranch="master", push_preview=true)
