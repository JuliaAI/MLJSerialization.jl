using Test

include("_dummy_model.jl")

@testset "machines" begin
    include("machines.jl")
end

@testset "controls" begin
    include("controls.jl")
end
