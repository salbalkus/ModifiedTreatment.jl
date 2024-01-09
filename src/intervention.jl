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
            error("Intervention: func must be a function of type (Vector, Float64, CausalTable) -> Vector")
        elseif !applicable(inverse, Vector, Float64, CausalTable)
            error("Intervention: inverse must be a function of type (Vector, Float64, CausalTable) -> Vector")
        elseif !isnothing(deriv) && !applicable(deriv, Vector, Float64, CausalTable)
            error("Intervention: deriv must be a function of type (Vector, Float64, CausalTable) -> Vector")
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
    apply_intervention = A -> fitresult.intervene(A, δ)
    
    result = DiffResults.DiffResult.(x, xd) 
    result = ForwardDiff.derivative!.(result, foo, 1.0)

    DiffResults.value.(result)

    # Compute the treatment
    new_treatment = apply_intervention(gettreatment(fitresult.ct))
    new_ct = replace_treatment(fitresult.ct, new_treatment)
    return CausalTables.summarize(new_ct)
end

function MMI.inverse_transform(model::Intervention, fitresult, δ)
    # create second closure
    apply_intervention = A -> fitresult.inverse_intervene(A, δ)
    
    # Compute the treatment
    new_treatment = apply_intervention(gettreatment(fitresult.ct))
    new_ct = replace_treatment(fitresult.ct, new_treatment)
    return CausalTables.summarize(new_ct)
end

# Create a new CausalTable with an intervened treatment and dropped response
function replace_treatment(ct::CausalTable, new_treatment::Vector)
    tbl = merge(NamedTuple{(ct.treatment,)}((new_treatment,)), Tables.columntable(TableOperations.select(ct.tbl, ct.controls...)))
    return CausalTable(tbl,
    ct.treatment,
    ct.response,
    ct.controls,
    ct.graph,
    ct.summaries
)
end

