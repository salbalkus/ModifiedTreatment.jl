
mutable struct InterventionModel <: MMI.Unsupervised end

function MMI.fit(m::InterventionModel, verbosity, O)
    LAs, A, L, summaries, treatmentvar = get_summarized_data(O)

    fitresult = (; 
                LAs = LAs,
                A = A,
                L = L,
                summaries = summaries,
                treatmentvar = treatmentvar
                )

    cache = nothing
    report = nothing
    return fitresult, cache, report
end

MMI.predict(m::InterventionModel, fitresult, intervention::Intervention) = (fitresult.LAs, fitresult.L, fitresult.A)
MMI.transform(m::InterventionModel, fitresult, intervention::Intervention) = get_intervened_data(fitresult.LAs, fitresult.A, fitresult.L, intervention, fitresult.summaries, fitresult.treatmentvar)
MMI.inverse_transform(m::InterventionModel, fitresult, intervention::Intervention) = MMI.transform(m, fitresult, inverse(intervention))

function get_summarized_data(O)
    # Apply summary function to the data and select only the variables that are not the response
    Os = CausalTables.summarize(O)
    LAs = CausalTables.replacetable(Os, TableOperations.select(Os, setdiff(Tables.columnnames(Os), (getresponsesymbol(Os),))...) |> Tables.columntable)

    # Collect names of the variables being summarized
    summaries = getsummaries(LAs)
    summarizedvars = [CausalTables.get_var_to_summarize(s) for s in summaries]    

    # Check to make sure treatment is not being summarized multiple times
    treatmentvar = gettreatmentsymbol(LAs)
    if count(==(treatmentvar), summarizedvars) > 1
        error("Treatment variable is being summarized multiple times. This is not allowed.")
    end
    if treatmentvar ∈ keys(summaries) && count(==(CausalTables.get_var_to_summarize(summaries[treatmentvar])), summarizedvars) > 1
        error("Treatment variable is a summary of a variable that is being summarized in other ways. This is not allowed.")
    end

    A = CausalTables.gettreatment(LAs)
    L = CausalTables.getcontrols(LAs)

    return LAs, NamedTuple{(gettreatmentsymbol(LAs),)}((A,)), L, summaries, treatmentvar
end


function get_intervened_data(LAs, A, L, intervention::Intervention, summaries, treatmentvar)
    if treatmentvar ∈ keys(summaries)
        # if the treatment is a network summary, need to construct the induced intervention before applying
        Δ = get_induced_intervention(intervention, summaries[treatmentvar])
    else
        Δ = intervention
    end

    Avec = A[treatmentvar]
    Aδ = apply_intervention(Δ, Avec, L)
    Aδd = differentiate_intervention(Δ, Avec, L)

    t = (treatmentvar,)
    Aderivatives = NamedTuple{t}((Aδd,))#, NamedTuple(Aδsd))
    Aδinterventions = NamedTuple{t}((Aδ,))#, NamedTuple(Aδs))
    LAδinterventions = CausalTables.replacetable(L, merge(gettable(L), Aδinterventions))
    return LAδinterventions, Aderivatives
end

