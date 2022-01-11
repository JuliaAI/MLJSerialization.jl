module TestMachines

using MLJSerialization
using MLJBase
using Test
using MLJTuning

include(joinpath(@__DIR__, "_models", "models.jl"))
using .Models
using MLJXGBoostInterface


function check_unchanged_fields(mach₁, mach₂)
    for field in (:state, :frozen, :model, :old_model, :old_upstream_state, :fit_okay)
        @test getfield(mach₁, field) == getfield(mach₂, field)
    end
end

simpledata(;n=100) = (x₁=rand(n),), rand(n)


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



@testset "Test serializable method of simple machines" begin
    X, y = simpledata()
    # Simple C based model with specific save method
    mach = machine(XGBoostRegressor(), X, y)
    fit!(mach, verbosity=0)
    smach = serializable("xgboost_mach.jls", mach)
    @test smach.report == mach.report
    @test smach.fitresult isa Vector
    @test smach.data == () != mach.data
    @test smach.resampled_data == () != mach.resampled_data
    @test smach.cache === nothing === mach.cache
    @test typeof(smach).parameters[2] == typeof(mach).parameters[2]
    @test all(s isa Source for s in smach.args)
    check_unchanged_fields(mach, smach)

    rm("xgboost_mach.xgboost.model")

    # Simple Pure julia model
    mach = machine(DecisionTreeRegressor(), X, y)
    fit!(mach, verbosity=0)
    smach = serializable("decisiontree.jls", mach)
    @test smach.report == mach.report
    @test smach.fitresult == mach.fitresult
    @test smach.data == () != mach.data
    @test smach.resampled_data == () != mach.resampled_data
    @test smach.cache === nothing === mach.cache
    @test all(s isa Source for s in smach.args)
    check_unchanged_fields(mach, smach)
end


@testset "Test TunedModel" begin
    X, y = simpledata()
    base_model = XGBoostRegressor()
    tuned_model = TunedModel(
        model=base_model,
        tuning=Grid(),
        range=[range(base_model, :num_round, values=[9,10,11])],
    )
    mach = machine(tuned_model, X, y)
    fit!(mach, rows=1:50)
    smach = serializable("tuned_model.jls", mach)
    @test smach.fitresult.fitresult isa Vector
    @test smach.data == () != mach.data
    @test smach.resampled_data == () != mach.resampled_data
    # There is a machine in the cache, should I call `serializable` on it?
    for i in 1:length(mach.cache)-1
        @test mach.cache[i] == smach.cache[i]
    end
    @test smach.cache == mach.cache
    @test check_unchanged_fields(mach, smach)
end

@testset "Test serializable of composite machines" begin
    # Composite model with sub model composite itself and some C inside
    model = Stack(
        metalearner = DecisionTreeRegressor(), 
        pipe = Pipeline(X -> coerce(X, :x₁=>Continuous), XGBoostRegressor),
        tree = DecisionTreeRegressor())
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)

    smach = serializable("pipeline_mach.jls", mach)
    # Check data has been wiped out from models at the first level of composition
    for submach in smach.report.machines
        @test submach.data == ()
        @test submach.resampled_data == ()
        @test submach.cache isa Nothing || :data ∉ keys(submach.cache)
    end
    # Check data has been wiped out at the second level too
    # The pipeline is itself a composite
    pipe = smach.report.machines[2]
    @test pipe isa Machine{<:DeterministicPipeline}
    @test pipe.report.machines[1].data == ()
    @test pipe.report.machines[1].resampled_data == ()
    @test pipe.report.machines[1].cache === nothing
    # Checking the fitresult of xgboost
    @test pipe.report.machines[1].fitresult isa Vector
end

end # module

true
