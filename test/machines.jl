module TestMachines

using MLJSerialization
using MLJBase
using Test

include(joinpath(@__DIR__, "_models", "models.jl"))
using .Models

@testset "serialization" begin

    @test MLJSerialization._filename("mymodel.jlso") == "mymodel"
    @test MLJSerialization._filename("mymodel.gz") == "mymodel"
    @test MLJSerialization._filename("mymodel") == "mymodel"

    model = DecisionTreeRegressor()

    X = (a = Float64[98, 53, 93, 67, 90, 68],
         b = Float64[64, 43, 66, 47, 16, 66],)
    Xnew = (a = Float64[82, 49, 16],
            b = Float64[36, 13, 36],)
    y =  [59.1, 28.6, 96.6, 83.3, 59.1, 48.0]

    mach =machine(model, X, y)
    filename = joinpath(@__DIR__, "machine.jlso")
    io = IOBuffer()
    @test_throws Exception MLJSerialization.save(io, mach; compression=:none)

    fit!(mach)
    report = mach.report
    pred = predict(mach, Xnew)
    MLJSerialization.save(io, mach; compression=:none)
    # Un-comment to update the `machine.jlso` file:
    #MLJSerialization.save(filename, mach)

    # test restoring data from filename:
    m = machine(filename)
    p = predict(m, Xnew)
    @test m.model == model
    @test m.report == report
    @test p ≈ pred
    m = machine(filename, X, y)
    fit!(m)
    p = predict(m, Xnew)
    @test p ≈ pred

    # test restoring data from io:
    seekstart(io)
    m = machine(io)
    p = predict(m, Xnew)
    @test m.model == model
    @test m.report == report
    @test p ≈ pred
    seekstart(io)
    m = machine(io, X, y)
    fit!(m)
    p = predict(m, Xnew)
    @test p ≈ pred

end

@testset "errors for deserialized machines" begin
    filename = joinpath(@__DIR__, "machine.jlso")
    m = machine(filename)
    @test_throws ArgumentError predict(m)
end

end # module

true
