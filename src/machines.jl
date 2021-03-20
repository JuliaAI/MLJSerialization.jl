using JLSO
import MLJModelInterface: save, restore
import MLJBase: Machine, machine

## SERIALIZATION

# helper:
_filename(file::IO) = string(rand(UInt))
_filename(file::String) = splitext(file)[1]

# saving:
"""
    MLJ.save(filename, mach::Machine; kwargs...)
    MLJ.save(io, mach::Machine; kwargs...)

    MLJBase.save(filename, mach::Machine; kwargs...)
    MLJBase.save(io, mach::Machine; kwargs...)

Serialize the machine `mach` to a file with path `filename`, or to an
input/output stream `io` (at least `IOBuffer` instances are
supported).

The format is JLSO (a wrapper for julia native or BSON serialization).
For some model types, a custom serialization will be additionally performed.

### Keyword arguments

These keyword arguments are passed to the JLSO serializer:

keyword        | values                        | default
---------------|-------------------------------|-------------------------
`format`       | `:julia_serialize`, `:BSON`   | `:julia_serialize`
`compression`  | `:gzip`, `:none`              | `:none`

See [https://github.com/invenia/JLSO.jl](https://github.com/invenia/JLSO.jl)
for details.

Any additional keyword arguments are passed to model-specific
serializers.

Machines are de-serialized using the `machine` constructor as shown in
the example below. Data (or nodes) may be optionally passed to the
constructor for retraining on new data using the saved model.


### Example

    using MLJ
    tree = @load DecisionTreeClassifier
    X, y = @load_iris
    mach = fit!(machine(tree, X, y))

    MLJ.save("tree.jlso", mach, compression=:none)
    mach_predict_only = machine("tree.jlso")
    predict(mach_predict_only, X)

    mach2 = machine("tree.jlso", selectrows(X, 1:100), y[1:100])
    predict(mach2, X) # same as above

    fit!(mach2) # saved learned parameters are over-written
    predict(mach2, X) # not same as above

    # using a buffer:
    io = IOBuffer()
    MLJ.save(io, mach)
    seekstart(io)
    predict_only_mach = machine(io)
    predict(predict_only_mach, X)

!!! warning "Only load files from trusted sources"
    Maliciously constructed JLSO files, like pickles, and most other
    general purpose serialization formats, can allow for arbitrary code
    execution during loading. This means it is possible for someone
    to use a JLSO file that looks like a serialized MLJ machine as a
    [Trojan
    horse](https://en.wikipedia.org/wiki/Trojan_horse_(computing)).

"""
function save(file::Union{String,IO},
              mach::Machine;
              verbosity=1,
              format=:julia_serialize,
              compression=:none,
              kwargs...)
    isdefined(mach, :fitresult)  ||
        error("Cannot save an untrained machine. ")

    # fallback `save` method returns `mach.fitresult` and saves nothing:
    serializable_fitresult =
        save(_filename(file), mach.model, mach.fitresult; kwargs...)

    JLSO.save(file,
              :model => mach.model,
              :fitresult => serializable_fitresult,
              :report => mach.report;
              format=format,
              compression=compression)
end

# deserializing:
function machine(file::Union{String,IO}, args...; cache=true, kwargs...)
    dict = JLSO.load(file)
    model = dict[:model]
    serializable_fitresult = dict[:fitresult]
    report = dict[:report]
    fitresult = restore(_filename(file), model, serializable_fitresult)
    if isempty(args)
        mach = Machine(model, cache=cache)
    else
        mach = machine(model, args..., cache=cache)
    end
    mach.state = 1
    mach.fitresult = fitresult
    mach.report = report
    return mach
end
