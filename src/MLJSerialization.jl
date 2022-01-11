module MLJSerialization

using MLJModelInterface
using MLJBase
using Serialization

import MLJBase: machine
import MLJModelInterface: save, restore
import IterationControl

export serializable, restore!, save, machine

# export IterationControl controls:
export Save

include("machines.jl")
include("controls.jl")

end
