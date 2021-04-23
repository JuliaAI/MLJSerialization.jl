module TestControls

using Test
using MLJSerialization
using MLJBase
using IterationControl
const IC = IterationControl

using ..DummyModel

@testset "Save" begin
    X, y = make_dummy(N=8);
    @test_throws Exception Save(filename="myfile.jlso")
    c = Save("serialization_test.jlso")
    m = machine(DummyIterativeModel(n=2), X, y)
    fit!(m, verbosity=0)
    state = @test_logs((:info, "Saving \"serialization_test1.jlso\". "),
                       IC.update!(c, m, 2, 1))
    @test state.filenumber == 1
    m.model.n = 5
    fit!(m, verbosity=0)
    state = IC.update!(c, m, 0, 2, state)
    @test state.filenumber == 2
    yhat = predict(IC.expose(m), X);

    deserialized_mach = MLJBase.machine("serialization_test2.jlso")
    yhat2 = predict(deserialized_mach, X)
    @test yhat2 ≈ yhat

    train_mach = machine(DummyIterativeModel(n=5), X, y)
    fit!(train_mach, verbosity=0)
    @test yhat ≈ predict(train_mach, X)
end

end

true
