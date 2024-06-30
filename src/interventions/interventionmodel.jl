
mutable struct InterventionModel <: MMI.Unsupervised end

function MMI.fit(m::InterventionModel, verbosity, O::CausalTable)

    LAs, L, A, treatmentvar, summaryvar = get_summarized_data(O)
    fitresult = (; 
                LAs = LAs,
                L = L,
                A = A,
                treatmentvar = treatmentvar,
                summaryvar = summaryvar
                )

    cache = nothing
    report = nothing
    return fitresult, cache, report
end

MMI.predict(m::InterventionModel, fitresult, intervention::Intervention) = (fitresult.LAs, CausalTables.confounders(LAs), CausalTables.treatment(LAs))
MMI.transform(m::InterventionModel, fitresult, intervention::Intervention) = get_intervened_data(fitresult.A, fitresult.L, intervention, fitresult.summaries, fitresult.treatmentvar, fitresult.summarizedvars)
MMI.inverse_transform(m::InterventionModel, fitresult, intervention::Intervention) = MMI.transform(m, fitresult, inverse(intervention))

function get_summarized_data(O)
    
    # Apply summary function to the data and select only the variables that are not the response
    Os = CausalTables.summarize(O)

    treatment_name = CausalTables.treatmentnames(Os)
    summary_name = CausalTables.treatmentsummarynames(Os)

    # Error handling
    if length(treatment_name) != 1 || length(summary_name) > 1
        throw(ArgumentError("InterventionModel only supports a single treatment variable and a single summary variable."))
    end

    LAs = CausalTables.responseparents(Os)
    L = CausalTables.confounders(LAs)
    A = CausalTables.treatment(LAs)
    
    return LAs, L, A, treatment_name, summary_name
end

function get_intervened_data(LAs, L, intervention::Intervention, treatment_name, summary_name)

    # Get names
    treatment_name = treatmentnames(LAs),
    summary_name = treatmentsummarynames(LAs)
    
    # Transform column vector
    A = Tables.getcolumn(LAs, treatment_name)
    Aδ = apply_intervention(intervention, A, L)
    Aδd = differentiate_intervention(intervention, A, L)

    # Construct tuples for storage in column table format
    t = (treatment_name,)
    Aderivatives = NamedTuple{t}((Aδd,))
    Aδinterventions = NamedTuple{t}((Aδ,))

    # Apply summary intervention #
    if length(summary_name) == 1
        As = Tables.getcolumn(LAs, summary_name)
        Aδs = apply_intervention(intervention_s, As, L)
        Aδsd = differentiate_intervention(intervention_s, As, L)

        ts = (summary_name,)
        Aderivatives = merge(Aderivatives, NamedTuple{ts}((Aδsd,)))
        Aδinterventions = merge(Aδinterventions, NamedTuple{ts}((Aδs,)))
    end

    LAδinterventions = CausalTables.replacetable(LAs, merge(L.data, Aδinterventions))
    return LAδinterventions, Aderivatives
end

