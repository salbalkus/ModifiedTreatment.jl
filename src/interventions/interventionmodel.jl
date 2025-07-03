
mutable struct InterventionModel <: MMI.Unsupervised end

function MMI.fit(m::InterventionModel, verbosity, O::CausalTable)

    LAs, L, A, treatment_name, summary_name = get_summarized_data(O)
    fitresult = (; 
                LAs = LAs,
                L = L,
                A = A,
                treatment_name = treatment_name,
                summary_name = summary_name
                )

    cache = nothing
    report = nothing
    return fitresult, cache, report
end

MMI.predict(m::InterventionModel, fitresult, intervention::Intervention) = (fitresult.LAs, fitresult.L, fitresult.A)
MMI.transform(m::InterventionModel, fitresult, intervention::Intervention) = get_intervened_data(fitresult.LAs, fitresult.L, intervention, fitresult.treatment_name, fitresult.summary_name)
MMI.inverse_transform(m::InterventionModel, fitresult, intervention::Intervention) = MMI.transform(m, fitresult, inverse(intervention))

function get_summarized_data(O)
    
    # Apply summary function to the data and select only the variables that are not the response
    Os = CausalTables.summarize(O)
    #Os = CausalTables.replace(Os; data = Os |> Replace(missing => NaN))

    # Error handling
    nonsummary_treatments = Os.treatment[map(x -> x ∉ keys(Os.summaries), Os.treatment)]
    summary_treatments = Os.treatment[map(x -> x ∈ keys(Os.summaries), Os.treatment)]
    if (length(nonsummary_treatments) > 1) || (length(summary_treatments) > 1)
        throw(ArgumentError("InterventionModel only supports a single treatment variable and a single summary variable."))
    else
        treatment_name = isempty(nonsummary_treatments) ? nothing : nonsummary_treatments[1]
        summary_name = isempty(summary_treatments) ? nothing : summary_treatments[1]
        if !isnothing(summary_name) && !isnothing(treatment_name) && CausalTables.gettarget(Os.summaries[summary_name]) != treatment_name
            throw(ArgumentError("Summary treatment does not summarize the non-summarized treatment."))
        end
    end
    LAs = CausalTables.responseparents(Os)
    L = CausalTables.treatmentparents(Os)
    A = CausalTables.treatment(LAs)
    
    return LAs, L, A, treatment_name, summary_name
end

# TODO: Currently only works for univariate treatment and summary
function get_intervened_data(LAs, L, intervention::Intervention, treatment_name, summary_name)
    
    # Transform column vector
    A = Tables.getcolumn(LAs, treatment_name)
    Aδ = apply_intervention(intervention, L, A)
    Aδd = differentiate_intervention(intervention, L, A)

    # Construct tuples for storage in column table format
    t = (treatment_name,)
    Aderivatives = NamedTuple{t}((Aδd,))
    Aδinterventions = NamedTuple{t}((Aδ,))

    # Apply summary intervention #
    if !isnothing(summary_name)
        intervention_s = get_induced_intervention(intervention, LAs.summaries[summary_name])
        As = Tables.getcolumn(LAs, summary_name)
        Aδs = apply_intervention(intervention_s, L, As)
        Aδsd = differentiate_intervention(intervention_s, L, As)

        ts = (summary_name,)
        Aderivatives = merge(Aderivatives, NamedTuple{ts}((Aδsd,)))
        Aδinterventions = merge(Aδinterventions, NamedTuple{ts}((Aδs,)))
    end

    original_name_order = Tables.columnnames(LAs)
    LAδinterventions = CausalTables.replace(LAs; data = merge(L.data, Aδinterventions) |> TableTransforms.Select(original_name_order)) 

    return LAδinterventions, Aderivatives
end

