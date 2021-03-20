using Documenter, MLJSerialization

makedocs(
    modules = [MLJSerialization],
    sitename = "MLJSerialization.jl",
)

deploydocs(
    repo = "github.com/alan-turing-institute/MLJSerialization.jl.git",
)
