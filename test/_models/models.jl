module Models

using MLJModelInterface
import MLJBase # needed for UnivariateFinite in ConstantClassifier

const MMI = MLJModelInterface

include("DecisionTree.jl")

end
