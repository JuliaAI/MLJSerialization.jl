module TestMachines

using Serialization
using MLJSerialization
using MLJBase
using Test
using MLJTuning
using MLJEnsembles

include(joinpath(@__DIR__, "_models", "models.jl"))
using .Models
using MLJXGBoostInterface


function generic_tests(mach₁, mach₂)
    @test mach₂.data == () != mach₁.data
    @test mach₂.args == () != mach₁.args
    @test mach₂.resampled_data == () != mach₁.resampled_data
    for field in (:state, :frozen, :model, :old_model, :old_upstream_state, :fit_okay)
        @test getfield(mach₁, field) == getfield(mach₂, field)
    end
end

simpledata(;n=100) = (x₁=rand(n),), rand(n)

@testset "Test serializable method of simple machines" begin
    X, y = simpledata()
    filename = "xgboost_mach.jls"
    # Simple C based model with specific save method
    mach = machine(XGBoostRegressor(), X, y)
    fit!(mach, verbosity=0)
    smach = serializable(filename, mach)
    @test smach.report == mach.report
    @test smach.fitresult isa Vector
    @test smach.cache === nothing === mach.cache
    @test typeof(smach).parameters[2] == typeof(mach).parameters[2]
    @test all(s isa Source for s in smach.args)
    generic_tests(mach, smach)

    Serialization.serialize(filename, smach)
    smach = Serialization.deserialize(filename)
    restore!(smach, filename)

    @test MLJBase.predict(smach, X) == MLJBase.predict(mach, X)
    @test fitted_params(smach) isa NamedTuple
    @test report(smach) == report(mach)

    rm("xgboost_mach.xgboost.model")
    rm(filename)
    # End to end
    MLJSerialization.save(filename, mach)
    smach = MLJSerialization.machine(filename)
    @test predict(smach, X) == predict(mach, X)

    rm("xgboost_mach.xgboost.model")
    rm(filename)

    # Simple Pure julia model
    filename = "decisiontree.jls"
    mach = machine(DecisionTreeRegressor(), X, y)
    fit!(mach, verbosity=0)
    smach = serializable(filename, mach)
    @test smach.report == mach.report
    @test smach.fitresult == mach.fitresult
    @test smach.cache === nothing === mach.cache
    @test all(s isa Source for s in smach.args)
    generic_tests(mach, smach)

    Serialization.serialize(filename, smach)
    smach = Serialization.deserialize(filename)
    restore!(smach, filename)

    @test MLJBase.predict(smach, X) == MLJBase.predict(mach, X)
    @test fitted_params(smach) isa NamedTuple
    @test report(smach) == report(mach)

    rm(filename)

    # End to end
    MLJSerialization.save(filename, mach)
    smach = MLJSerialization.machine(filename)
    @test predict(smach, X) == predict(mach, X)

    rm(filename)

end


@testset "Test TunedModel" begin
    filename = "tuned_model.jls"
    X, y = simpledata()
    base_model = XGBoostRegressor()
    tuned_model = TunedModel(
        model=base_model,
        tuning=Grid(),
        range=[range(base_model, :num_round, values=[9,10,11])],
    )
    mach = machine(tuned_model, X, y)
    fit!(mach, rows=1:50)
    smach = serializable(filename, mach)
    @test smach.fitresult.fitresult isa Vector
    @test smach.report == mach.report
    # There is a machine in the cache, should I call `serializable` on it?
    for i in 1:length(mach.cache)-1
        @test mach.cache[i] == smach.cache[i]
    end
    generic_tests(mach, smach)

    Serialization.serialize(filename, smach)
    smach = Serialization.deserialize(filename)
    restore!(smach, filename)

    @test MLJBase.predict(smach, X) == MLJBase.predict(mach, X)
    @test fitted_params(smach) isa NamedTuple
    @test report(smach) == report(mach)

    rm("tuned_model.xgboost.model")
    rm(filename)

    # End to end
    MLJSerialization.save(filename, mach)
    smach = MLJSerialization.machine(filename)
    @test predict(smach, X) == predict(mach, X)

    rm("tuned_model.xgboost.model")
    rm(filename)

end

@testset "Test serializable Ensemble machine" begin
    filename = "ensemble_mach.jls"
    X, y = simpledata()
    model = EnsembleModel(model=XGBoostRegressor())
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)
    smach = serializable(filename, mach)
    @test mach.cache == smach.cache
    @test smach.report === mach.report
    generic_tests(mach, smach)
    @test smach.fitresult isa MLJEnsembles.WrappedEnsemble
    @test smach.fitresult.atom == model.model
    for fr in smach.fitresult.ensemble 
        @test fr isa Vector
    end

    Serialization.serialize(filename, smach)
    smach = Serialization.deserialize(filename)
    restore!(smach, filename)

    @test MLJBase.predict(smach, X) == MLJBase.predict(mach, X)
    @test fitted_params(smach) isa NamedTuple
    @test report(smach).measures == report(mach).measures
    @test report(smach).oob_measurements isa Missing
    @test report(mach).oob_measurements isa Missing

    rm("ensemble_mach.xgboost.model")
    rm(filename)

    # End to end
    MLJSerialization.save(filename, mach)
    smach = MLJSerialization.machine(filename)
    @test predict(smach, X) == predict(mach, X)

    rm("ensemble_mach.xgboost.model")
    rm(filename)

end

@testset "Test serializable of composite machines" begin
    # Composite model with some C inside
    filename = "stack_mach.jls"
    X, y = simpledata()
    model = Stack(
        metalearner = DecisionTreeRegressor(), 
        xgboost = XGBoostRegressor(),
        tree = DecisionTreeRegressor())
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)

    smach = serializable(filename, mach)

    generic_tests(mach, smach)
    # Check data has been wiped out from models at the first level of composition
    @test length(machines(glb(smach))) == length(machines(glb(mach)))
    for submach in machines(glb(smach))
        @test submach.data == ()
        @test submach.resampled_data == ()
        @test submach.cache isa Nothing || :data ∉ keys(submach.cache)
    end

    @test smach.fitresult isa MLJBase.CompositeFitresult

    Serialization.serialize(filename, smach)
    smach = Serialization.deserialize(filename)
    restore!(smach, filename)

    @test MLJBase.predict(smach, X) == MLJBase.predict(mach, X)
    @test keys(fitted_params(smach)) == keys(fitted_params(mach))
    @test keys(report(smach)) == keys(report(mach))

    rm(filename)
    rm("stack_mach.xgboost.model")

    # End to end
    MLJSerialization.save(filename, mach)
    smach = MLJSerialization.machine(filename)
    @test predict(smach, X) == predict(mach, X)

    rm("stack_mach.xgboost.model")
    rm(filename)
end

@testset "Test serializable of pipeline" begin
    # Composite model with some C inside
    filename = "pipe_mach.jls"
    X, y = simpledata()
    pipe = (X -> coerce(X, :x₁=>Continuous)) |> DecisionTreeRegressor()
    mach = machine(pipe, X, y)
    fit!(mach, verbosity=0)

    smach = serializable(filename, mach)

    generic_tests(mach, smach)
    @test MLJBase.predict(smach, X) == MLJBase.predict(mach, X)
    @test keys(fitted_params(smach)) == keys(fitted_params(mach))
    @test keys(report(smach)) == keys(report(mach))
    # Check data has been wiped out from models at the first level of composition
    @test length(machines(glb(smach))) == length(machines(glb(mach)))
    for submach in machines(glb(smach))
        @test submach.data == ()
        @test submach.resampled_data == ()
        @test submach.cache isa Nothing || :data ∉ keys(submach.cache)
    end

    # End to end
    MLJSerialization.save(filename, mach)
    smach = MLJSerialization.machine(filename)
    @test predict(smach, X) == predict(mach, X)

    rm(filename)
end

end # module

true
