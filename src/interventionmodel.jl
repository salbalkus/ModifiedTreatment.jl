mutable struct InterventionModel <: MMI.Unsupervised 
    intervention::Intervention
end

function MLJBase.fit(model::InterventionModel, verbosity, ct)
    mach_intervention = machine(model.intervention)
    cts = CausalTables.summarize(ct)
    fitresult = (; 
        cts = cts, 
                )
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MLJBase.predict(model::InterventionModel, fitresult, δ)
    # Output a tuple of summarized L and A
end


