
mutable struct InterventionModel <: MMI.Unsupervised end

function MMI.fit(m::InterventionModel, verbosity, O)
    # TODO: Assumes only a single treatment is specified.
    LAs = CausalTables.replacetable(O, TableOperations.select(CausalTables.summarize(O), setdiff(keys(O.tbl), (O.response,))...) |> Tables.columntable)
    summaryvars = [CausalTables.get_var_to_summarize(s) for s in LAs.summaries]
    summarytreatment = keys(LAs.summaries)[summaryvars .== LAs.treatment]
    Lstbl = TableOperations.select(LAs, vcat(LAs.controls, keys(LAs.summaries)[summaryvars .!= LAs.treatment]...)...) |> Tables.columntable
    Ls = CausalTables.replacetable(LAs, Lstbl)
    
    fitresult = (; 
                LAs = LAs,
                A = CausalTables.gettreatment(LAs),
                Ls = Ls,
                summarytreatment = summarytreatment
                )

    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(m::InterventionModel, fitresult, Δ::Intervention)
    Asummaries = TableOperations.select(fitresult.LAs, fitresult.summarytreatment...) |> Tables.columntable
    As = merge(NamedTuple{(fitresult.LAs.treatment,)}((fitresult.A,)), NamedTuple{fitresult.summarytreatment}(Asummaries))
    return fitresult.LAs, fitresult.Ls, As
end

function MMI.transform(m::InterventionModel, fitresult, Δ::Intervention)
    Aδ = apply_intervention(Δ, fitresult.A, fitresult.Ls)
    Aδd = differentiate_intervention(Δ, fitresult.A, fitresult.Ls)

    Aδs = Vector{Pair}(undef, length(fitresult.summarytreatment))
    Aδsd = Vector{Pair}(undef, length(fitresult.summarytreatment))

    for (i, st) in enumerate(fitresult.summarytreatment)
        Δsummary = get_induced_intervention(Δ, fitresult.LAs.summaries[st])
        As = Tables.getcolumn(fitresult.LAs, st)
        Aδs[i] = st => apply_intervention(Δsummary, As, fitresult.Ls)
        Aδsd[i] = st => differentiate_intervention(Δsummary, As, fitresult.Ls)
    end

    t = (fitresult.LAs.treatment,)
    Aderivatives = merge(NamedTuple{t}((Aδd,)), NamedTuple(Aδsd))
    Aδinterventions = merge(NamedTuple{t}((Aδ,)), NamedTuple(Aδs))
    LAδinterventions = CausalTables.replacetable(fitresult.Ls, merge(fitresult.Ls.tbl, Aδinterventions))
    return LAδinterventions, Aderivatives
end

function MMI.inverse_transform(m::InterventionModel, fitresult, Δ::Intervention)
    Aδ = apply_inverse_intervention(Δ, fitresult.A, fitresult.Ls)
    Aδd = differentiate_inverse_intervention(Δ, fitresult.A, fitresult.Ls)

    Aδs = Vector{Pair}(undef, length(fitresult.summarytreatment))
    Aδsd = Vector{Pair}(undef, length(fitresult.summarytreatment))

    for (i, st) in enumerate(fitresult.summarytreatment)
        Δsummary = get_induced_intervention(Δ, fitresult.LAs.summaries[st])
        As = Tables.getcolumn(fitresult.LAs, st)
        Aδs[i] = st => apply_inverse_intervention(Δsummary, As, fitresult.Ls)
        Aδsd[i] = st => differentiate_inverse_intervention(Δsummary, As, fitresult.Ls)
    end

    t = (fitresult.LAs.treatment,)
    Aderivatives = merge(NamedTuple{t}((Aδd,)), NamedTuple(Aδsd))
    Aδinterventions = merge(NamedTuple{t}((Aδ,)), NamedTuple(Aδs))
    LAδinterventions = CausalTables.replacetable(fitresult.Ls, merge(fitresult.Ls.tbl, Aδinterventions))
    return LAδinterventions, Aderivatives
end
