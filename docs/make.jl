using SOSEM
using Documenter

DocMeta.setdocmeta!(SOSEM, :DocTestSetup, :(using SOSEM); recursive=true)

makedocs(;
    modules=[SOSEM],
    authors="Daniel Cerkoney <dcerkoney@physics.rutgers.edu>",
    repo="https://github.com/dcerkoney/SOSEM.jl/blob/{commit}{path}#{line}",
    sitename="SOSEM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dcerkoney.github.io/SOSEM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dcerkoney/SOSEM.jl",
    devbranch="master",
)
