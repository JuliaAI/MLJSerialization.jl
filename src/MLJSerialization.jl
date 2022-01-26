module MLJSerialization

using MLJModelInterface
using MLJBase
using Serialization
using MLJTuning
using MLJEnsembles

import MLJBase: machine
import MLJModelInterface: save, restore


export serializable, restore!, save, machine

include("machines.jl")

end
