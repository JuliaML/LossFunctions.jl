using Documenter, LossFunctions

istravis = "TRAVIS" ∈ keys(ENV)

makedocs(
    format = Documenter.HTML(assets=["assets/style.css","assets/favicon.ico"], prettyurls=istravis),
    sitename = "LossFunctions.jl",
    authors = "Christof Stocker, Tom Breloff, Alex Williams",
    pages = [
        hide("Home" => "index.md"),
        "Introduction" => [
            "introduction/gettingstarted.md",
            "introduction/motivation.md",
        ],
        "User's Guide" => [
            "user/interface.md",
            "user/aggregate.md",
        ],
        "Available Losses" => [
            "losses/distance.md",
            "losses/margin.md",
            "losses/other.md",
        ],
        "Advances Topics" => [
            "advanced/extend.md",
            "advanced/developer.md",
        ],
        hide("Indices" => "indices.md"),
        "acknowledgements.md",
        "LICENSE.md",
    ],
)

deploydocs(repo="github.com/JuliaML/LossFunctions.jl.git")
