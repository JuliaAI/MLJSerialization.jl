###############################################################################
#####                        SAVE AND RELOAD                              #####
###############################################################################
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
              kwargs...)
    isdefined(mach, :fitresult)  ||
        error("Cannot save an untrained machine. ")

    smach = serializable(mach, kwargs...)

    serialize(file, smach)
end

"""
Not sure how to provide new arguments:
    - change mach.data?
"""
function machine(file::Union{String, IO}, raw_arg1=nothing, raw_args...)
    smach = deserialize(file)
    restore!(smach)
    if raw_arg1 !== nothing
        args = source.((raw_arg1, raw_args...))
        MLJBase.check(smach.model, args...; full=true)
        smach.args = args
    end
    return smach
end


###############################################################################
#####                           UTILITIES                                 #####
###############################################################################

maybe_serializable(val) = val
maybe_serializable(val::Machine) = serializable(val)


serializable_cache(cache) = cache
serializable_cache(cache::Tuple) = Tuple(maybe_serializable(val) for val in cache)
function serializable_cache(cache::NamedTuple)
    new_keys = filter(!=(:data), keys(cache))
    return NamedTuple{new_keys}([maybe_serializable(cache[key]) for key in new_keys])
end


setreport!(mach::Machine, report) = 
    setfield!(mach, :report, report)

function setreport!(mach::Machine{<:Composite}, report)
    basereport = MLJBase.report(glb(mach))
    report_additions = Base.structdiff(report, basereport)
    mach.report = merge(basereport, report_additions)
end

###############################################################################
#####         PROBABLY TO BE EXPORTED TO THEIR RESP MODULES               #####
###############################################################################

using MLJTuning
using MLJEnsembles


MLJModelInterface.save(model::MLJTuning.EitherTunedModel, fitresult::Machine; kwargs...) =
    serializable(fitresult, kwargs...)

function MLJModelInterface.restore(model::MLJTuning.EitherTunedModel, fitresult)
    fitresult.fitresult = restore(fitresult.model, fitresult.fitresult)
    return fitresult
end

MLJModelInterface.save(model::MLJEnsembles.EitherEnsembleModel, fitresult; kwargs...) =
    MLJEnsembles.WrappedEnsemble(
        fitresult.atom,
        [save(fitresult.atom, fr, kwargs...) for fr in fitresult.ensemble]
    )

function MLJModelInterface.restore(model::MLJEnsembles.EitherEnsembleModel, fitresult)
    return MLJEnsembles.WrappedEnsemble(
        fitresult.atom,
        [restore(fitresult.atom, fr) for fr in fitresult.ensemble]
    )
end


"""
    save(model::Composite, fitresult; kwargs...)
"""
function save(model::Composite, fitresult; kwargs...)
    # THIS IS WIP: NOT WORKING
    signature = MLJBase.signature(fitresult)

    operation_nodes = values(MLJBase._operation_part(signature))
    report_nodes = values(MLJBase._report_part(signature))

    W = glb(operation_nodes..., report_nodes...)

    nodes_ = filter(x -> !(x isa Source), nodes(W))

    # instantiate node dictionary with source nodes and exception nodes
    # This supposes that exception nodes only occur in the signature otherwise we need 
    # to to this differently
    newnode_given_old =
        IdDict{AbstractNode,AbstractNode}([old => source() for old in sources(W)])
    # Other useful mappings
    newoperation_node_given_old =
        IdDict{AbstractNode,AbstractNode}()
    newreport_node_given_old =
        IdDict{AbstractNode,AbstractNode}()
    newmach_given_old = IdDict{Machine,Machine}()

    # build the new network, nodes are nicely ordered
    for N in nodes_
        # Retrieve the future node's ancestors
        args = [newnode_given_old[arg] for arg in N.args]
        if N.machine === nothing
            newnode_given_old[N] = node(N.operation, args...)
        else
            # The same machine can be associated with multiple nodes
            if N.machine in keys(newmach_given_old)
                m = newmach_given_old[N.machine]
            else
                m = serializable(N.machine)
                m.args = Tuple(newnode_given_old[s] for s in N.machine.args)
                newmach_given_old[N.machine] = m
            end
            newnode_given_old[N] = N.operation(m, args...)
        end
        # Sort nodes according to: operation_node, report_node
        if N in operation_nodes
            newoperation_node_given_old[N] = newnode_given_old[N]
        elseif N in report_nodes
            newreport_node_given_old[N] = newnode_given_old[N]
        end
    end

    newoperation_nodes = Tuple(newoperation_node_given_old[N] for N in
            operation_nodes)
    newreport_nodes = Tuple(newreport_node_given_old[N] for N in
            report_nodes)
    report_tuple =
        NamedTuple{keys(MLJBase._report_part(signature))}(newreport_nodes)
    operation_tuple =
        NamedTuple{keys(MLJBase._operation_part(signature))}(newoperation_nodes)

    newsignature = if isempty(report_tuple)
                        operation_tuple
                    else
                        merge(operation_tuple, (report=report_tuple,))
                    end
    

    newfitresult = MLJBase.CompositeFitresult(newsignature)
    setfield!(newfitresult, :report_additions, report_tuple)

    return newfitresult

end

###############################################################################
#####                  SERIALIZABLE AND RESTORE                           #####
###############################################################################


"""
    serializable(mach::Machine; kwargs...)

Copies the state of the machine to make it serializable.
    - Removes all data from caches, args and data fields
    - Makes all `fitresults` serializable
"""
function serializable(mach::Machine{<:Any, C}; kwargs...) where C
    copymach = machine(mach.model, mach.args..., cache=C)

    for fieldname in fieldnames(Machine)
        if fieldname ∈ (:model, :report)
            continue
        elseif  fieldname == :state
            setfield!(copymach, :state, -1)
        # Wipe data from cache
        elseif fieldname == :cache 
            setfield!(copymach, :cache, serializable_cache(mach.cache))
        elseif fieldname == :args 
            setfield!(copymach, fieldname, ())
        # Let those fields undefined
        elseif fieldname ∈ (:data, :resampled_data, :old_rows)
            continue
        # Make fitresult ready for serialization
        elseif fieldname == :fitresult
            copymach.fitresult = save(mach.model, getfield(mach, fieldname), kwargs...)
        else
            setfield!(copymach, fieldname, getfield(mach, fieldname))
        end
    end

    setreport!(copymach, mach.report)
    
    return copymach
end


function restore!(mach::Machine)
    mach.fitresult = restore(mach.model, mach.fitresult)
    return mach
end

function restore!(mach::Machine{<:Composite})
    glb_node = glb(mach)
    for submach in machines(glb_node)
        restore!(submach)
    end
    return mach
end