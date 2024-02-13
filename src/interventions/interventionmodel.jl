
mutable struct InterventionModel <: MMI.Unsupervised end

function MMI.fit(m::InterventionModel, verbosity, O)
    LAs, A, L, summaries, treatmentvar, summarizedvars = get_summarized_data(O)

    fitresult = (; 
                LAs = LAs,
                A = A,
                L = L,
                summaries = summaries,
                treatmentvar = treatmentvar,
                summarizedvars = summarizedvars
                )

    cache = nothing
    report = nothing
    return fitresult, cache, report
end

MMI.predict(m::InterventionModel, fitresult, intervention::Intervention) = (fitresult.LAs, fitresult.L, fitresult.A)
MMI.transform(m::InterventionModel, fitresult, intervention::Intervention) = get_intervened_data(fitresult.A, fitresult.L, intervention, fitresult.summaries, fitresult.treatmentvar, fitresult.summarizedvars)
MMI.inverse_transform(m::InterventionModel, fitresult, intervention::Intervention) = MMI.transform(m, fitresult, inverse(intervention))

function get_summarized_data(O)
    # Apply summary function to the data and select only the variables that are not the response
    Os = CausalTables.summarize(O)
    LAs = CausalTables.replacetable(Os, TableOperations.select(Os, setdiff(Tables.columnnames(Os), (getresponsesymbol(Os),))...) |> Tables.columntable)

    # Collect names of the variables being summarized
    summaries = getsummaries(LAs)
    summarizedvars = NamedTuple([CausalTables.get_var_to_summarize(val) => key for (key, val) in pairs(summaries)])

    # Check to make sure treatment is not being summarized multiple times
    treatmentvar = gettreatmentsymbol(LAs)
    if count(==(treatmentvar), summarizedvars) > 1
        error("Treatment variable is being summarized multiple times. This is not allowed.")
    end
    if treatmentvar ∈ keys(summaries)
        error("Treatment variable is a summarized variable. This is not allowed. Instead, specify `treatment` as the variable to which a direct intervention is being applied, and specify the summarized treatment in `summary` only.")
    end

    # Construct new Tables / CausalTables to return
    if treatmentvar ∈ keys(summarizedvars)
        A = TableOperations.select(LAs, treatmentvar, summarizedvars[treatmentvar]) |> Tables.columntable
    else
        A = TableOperations.select(LAs, treatmentvar) |> Tables.columntable
    end
    controlssymbols = getcontrolssymbols(LAs)

    if any(controlssymbols .∈ [values(summarizedvars)])
        L = replacetable(LAs, TableOperations.select(LAs, controlssymbols..., values(summarizedvars[controlssymbols])...) |> Tables.columntable)
    else
        L = replacetable(LAs, TableOperations.select(LAs, controlssymbols...) |> Tables.columntable)
    end

    return LAs, A, L, summaries, treatmentvar, summarizedvars
end


function get_intervened_data(A, L, intervention::Intervention, summaries, treatmentvar, summarizedvars)

    # Apply direct intervention
    Avec = A[treatmentvar]
    Aδ = apply_intervention(intervention, Avec, L)
    Aδd = differentiate_intervention(intervention, Avec, L)

    t = (treatmentvar,)
    Aderivatives = NamedTuple{t}((Aδd,))
    Aδinterventions = NamedTuple{t}((Aδ,))


    # Apply summary intervention
    if treatmentvar ∈ keys(summarizedvars)
        # Compute the induced intervention by mapping the treatmentvar to its intervention
        summarytreatmentvar = summarizedvars[treatmentvar]
        intervention_s = get_induced_intervention(intervention, summaries[summarytreatmentvar])
        Avecsummary = A[summarytreatmentvar]
        Aδs = apply_intervention(intervention_s, Avecsummary, L)
        Aδsd = differentiate_intervention(intervention_s, Avecsummary, L)

        ts = (summarytreatmentvar,)
        Aderivatives = merge(Aderivatives, NamedTuple{ts}((Aδsd,)))
        Aδinterventions = merge(Aδinterventions, NamedTuple{ts}((Aδs,)))
    end

    LAδinterventions = CausalTables.replacetable(L, merge(gettable(L), Aδinterventions))
    return LAδinterventions, Aderivatives
end

