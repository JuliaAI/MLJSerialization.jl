module TestMachines

using Serialization
using MLJSerialization
using MLJBase
using Test
using MLJTuning
using MLJEnsembles

include(joinpath(@__DIR__, "_models", "models.jl"))
using .Models


function test_args(mach)
    # Check source nodes are empty if any
    for arg in mach.args
        if arg isa Source 
            @test arg == source()
        end
    end
end

function test_data(mach₁, mach₂)
    @test mach₂.old_rows === nothing != mach₁.old_rows
    @test mach₂.data == () != mach₁.data
    @test mach₂.resampled_data == () != mach₁.resampled_data
    if mach₂ isa NamedTuple
        @test :data ∉ keys(mach₂.cache)
    end
end

function generic_tests(mach₁, mach₂)
    test_args(mach₂)
    test_data(mach₁, mach₂)
    @test mach₂.state == -1
    for field in (:frozen, :model, :old_model, :old_upstream_state, :fit_okay)
        @test getfield(mach₁, field) == getfield(mach₂, field)
    end
end

simpledata(;n=100) = (x₁=rand(n),), rand(n)

@testset "Test serializable method of simple machines" begin
    X, y = simpledata()
    # Simple Pure julia model
    filename = "decisiontree.jls"
    mach = machine(DecisionTreeRegressor(), X, y)
    fit!(mach, verbosity=0)
    smach = serializable(mach)
    @test smach.report == mach.report
    @test smach.fitresult == mach.fitresult
    generic_tests(mach, smach)

    Serialization.serialize(filename, smach)
    smach = Serialization.deserialize(filename)
    restore!(smach)

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
    base_model = DecisionTreeRegressor()
    tuned_model = TunedModel(
        model=base_model,
        tuning=Grid(),
        range=[range(base_model, :min_samples_split, values=[2,3,4])],
    )
    mach = machine(tuned_model, X, y)
    fit!(mach, rows=1:50, verbosity=0)
    smach = serializable(mach)
    @test smach.fitresult isa Machine
    @test smach.report == mach.report
    # There is a machine in the cache, should I call `serializable` on it?
    for i in 1:length(mach.cache)-1
        @test mach.cache[i] == smach.cache[i]
    end
    generic_tests(mach, smach)

    Serialization.serialize(filename, smach)
    smach = Serialization.deserialize(filename)
    restore!(smach)

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

@testset "Test serializable Ensemble machine" begin
    filename = "ensemble_mach.jls"
    X, y = simpledata()
    model = EnsembleModel(model=DecisionTreeRegressor())
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)
    smach = serializable(mach)
    @test smach.report === mach.report
    generic_tests(mach, smach)
    @test smach.fitresult isa MLJEnsembles.WrappedEnsemble
    @test smach.fitresult.atom == model.model

    Serialization.serialize(filename, smach)
    smach = Serialization.deserialize(filename)
    restore!(smach)

    @test MLJBase.predict(smach, X) == MLJBase.predict(mach, X)
    @test fitted_params(smach) isa NamedTuple
    @test report(smach).measures == report(mach).measures
    @test report(smach).oob_measurements isa Missing
    @test report(mach).oob_measurements isa Missing

    rm(filename)

    # End to end
    MLJSerialization.save(filename, mach)
    smach = MLJSerialization.machine(filename)
    @test predict(smach, X) == predict(mach, X)

    rm(filename)

end

@testset "Test serializable of composite machines" begin
    # Composite model with some C inside
    filename = "stack_mach.jls"
    X, y = simpledata()
    model = Stack(
        metalearner = DecisionTreeRegressor(), 
        tree1 = DecisionTreeRegressor(min_samples_split=3),
        tree2 = DecisionTreeRegressor())
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)

    smach = serializable(mach)

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
    restore!(smach)

    @test MLJBase.predict(smach, X) == MLJBase.predict(mach, X)
    @test keys(fitted_params(smach)) == keys(fitted_params(mach))
    @test keys(report(smach)) == keys(report(mach))

    rm(filename)

    # End to end
    MLJSerialization.save(filename, mach)
    smach = MLJSerialization.machine(filename)
    @test predict(smach, X) == predict(mach, X)

    rm(filename)
end

@testset "Test serializable of pipeline" begin
    # Composite model with some C inside
    filename = "pipe_mach.jls"
    X, y = simpledata()
    pipe = (X -> coerce(X, :x₁=>Continuous)) |> DecisionTreeRegressor()
    mach = machine(pipe, X, y)
    fit!(mach, verbosity=0)

    smach = serializable(mach)

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

@testset "Test serializable of nested composite machines" begin
    # Composite model with some C inside
    filename = "nested stack_mach.jls"
    X, y = simpledata()

    pipe = (X -> coerce(X, :x₁=>Continuous)) |> DecisionTreeRegressor()
    model = Stack(
        metalearner = DecisionTreeRegressor(), 
        pipe = pipe)
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)

    save(filename, mach)
    smach = MLJSerialization.machine(filename)

    @test predict(smach, X) == predict(mach, X)

    # Test data as been erased at the first and second level of composition
    submachs = machines(glb(mach))
    for (i, submach) in enumerate(machines(glb(smach)))
        test_data(submachs[i], submach)
        if submach isa Machine{<:Composite,}
            test_data(submachs[i], submach)
        end
    end

    rm(filename)


end

end # module

true
