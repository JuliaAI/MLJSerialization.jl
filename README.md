# MLJSerialization.jl 

**As of MLJ 0.18, this package is no longer a dependecy of MLJ, with serialization functionality provided by MLJBase 0.20**

| Linux | Coverage |
| :-----------: | :------: |
| [![Build status](https://github.com/JuliaAI/MLJSerialization.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJSerialization.jl/actions)| [![codecov.io](http://codecov.io/github/JuliaAI/MLJSerialization.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaAI/MLJSerialization.jl?branch=dev) |

A package adding model serialization to the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) machine
learning framework.


## Installation

```julia
using Pkg
Pkg.add("MLJ")
Pkg.add("MLJSerialization")
```

## Sample usage

Fit and save a decision tree model:

```julia
Pkg.add("DecisionTree")

using MLJ
using MLJSerialization

X, y = @load_iris

Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()
mach = fit!(machine(tree, X, y))
MLJSerialization.save("my_machine.jlso", mach)
```

Retrieve the saved machine:

```julia
mach2 = machine("my_machine.jlso")

Xnew = selectrows(X, 1:3)
predict_mode(mach2, Xnew)

julia> predict_mode(mach2, Xnew)
3-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "setosa"
 "setosa"
 "setosa"
```

## Documentation

Documentation is provided in the [Saving
machines](https://alan-turing-institute.github.io/MLJ.jl/dev/machines/#Saving-machines-1)
section of the
[MLJManual](https://alan-turing-institute.github.io/MLJ.jl/dev/)


