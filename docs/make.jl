using Documenter, LossFunctions

makedocs(
    modules = [LossFunctions],
    clean = false,
    format = :html,
    assets = [
        joinpath("assets", "favicon.ico"),
        joinpath("assets", "style.css"),
    ],
    sitename = "LossFunctions.jl",
    authors = "Christof Stocker, Tom Breloff, Alex Williams",
    linkcheck = !("skiplinks" in ARGS),
    pages = Any[
        "Home" => "index.md",
        "Introduction" => [
            "Getting Started" => "introduction/gettingstarted.md",
            "Background and Motivation" => "introduction/motivation.md",
        ],
        "User's Guide" => [
            "Working with Losses" => "user/interface.md",
        ],
        hide("Indices" => "indices.md"),
        "LICENSE.md",
    ],
    html_prettyurls = !("local" in ARGS),
)

deploydocs(
    repo = "github.com/JuliaML/LossFunctions.jl.git",
    target = "build",
    julia = "1.0",
    deps = nothing,
    make = nothing,
)
