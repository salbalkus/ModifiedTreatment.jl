# Function template
# func(A, δ, O)
# where O is a CausalTable
# optionally, can provide deriv(A; L, δ)

mutable struct Intervention <: MMI.Unsupervised 
    func::Function
    inverse::Function
    deriv::Union{Function, Nothing}
    function Intervention(func, inverse, deriv)
        if !applicable(func, Vector, Float64, CausalTable)
            error("ShiftIntervention: func must be a function of type (Vector, Float64, CausalTable) -> Vector")
        elseif !applicable(inverse, Vector, Float64, CausalTable)
            error("ShiftIntervention: inverse must be a function of type (Vector, Float64, CausalTable) -> Vector")
        elseif !isnothing(deriv) && !applicable(deriv, Vector, Float64, CausalTable)
            error("ShiftIntervention: deriv must be a function of type (Vector, Float64, CausalTable) -> Vector")
        end
        return new(func, inverse, deriv)
    end
end

Intervention(func, inverse) = Intervention(func, inverse, nothing)

function MMI.fit(model::Intervention, verbosity, ct)
    # create first closure
    intervene = (A, δ) -> model.func(A, δ, ct)
    inverse_intervene = (A, δ) -> model.inverse(A, δ, ct)

    fitresult = (; 
        ct = ct, 
        intervene = intervene,
        inverse_intervene = inverse_intervene
                )
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

MMI.predict(model::Intervention, fitresult, δ) = CausalTables.summarize(fitresult.ct)

function MMI.transform(model::Intervention, fitresult, δ)
    # create second closure
    apply_intervention = A -> intervene(A, δ)
    
    # Compute the treatment
    new_treatment = apply_intervention(gettreatment(fitresult.ct))
    new_ct = replace_treatment(fitresult.ct, new_treatment)
    return CausalTables.summarize(new_ct)
end

function MMI.inverse_transform(model::Intervention, fitresult, δ)
    # create second closure
    apply_intervention = A -> inverse_intervene(A, δ)
    
    # Compute the treatment
    new_treatment = apply_intervention(gettreatment(fitresult.ct))
    new_ct = replace_treatment(fitresult.ct, new_treatment)
    return CausalTables.summarize(new_ct)
end

# Create a new CausalTable with an intervened treatment and dropped response
replace_treatment(ct::CausalTable, new_treatment::Vector) = CausalTable(
    merge(NamedTuple{ct.treatment}(new_treatment), TableOperations.select(ct.tbl, ct.controls) |> Tables.columntable)
    gettreatment(ct),
    getresponse(ct),
    getcontrols(ct)
    getgraph(ct),
    getsummaries(ct)
)

