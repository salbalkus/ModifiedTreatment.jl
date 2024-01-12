struct ResampledModel <: MMI.Model
    model::MMI.Model
end

function MMI.fit(rse::ResampledModel, verbosity, X...)

    # Here X should be a tuple of equal-length vectors
    # Each vector entry is one "sample" of the data for that particular parameter

    # Fit each sample
    machines = Iterators.map(x -> fit!(machine(rse.model, x...)), zip(X...))

    fitresult = (; machines = machines)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

MMI.predict(rse::ResampledModel, fitresult, X...) = Iterators.map((mach, x) -> MMI.predict(mach, x...), zip(fitresult.machines, zip(X...)))
MMI.transform(rse::ResampledModel, fitresult, X...) = Iterators.map((mach, x) -> MMI.transform(mach, x...), zip(fitresult.machines, zip(X...)))
MMI.inverse_transform(rse::ResampledModel, fitresult, X...) = Iterators.map((mach, x) -> MMI.inverse_transform(mach, x...), zip(fitresult.machines, zip(X...)))


