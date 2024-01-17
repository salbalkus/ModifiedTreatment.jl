bootstrap(sampler::BootstrapSampler, mach_Qn, mach_Hn, O::CausalTable, δ::Intervention, B::Int64) = [bootstrap_sample(sampler, mach_Qn, mach_Hn, O, δ) for b in 1:B]

function bootstrap(mtpmach::Machine{MTP}, δ::Intervention, B::Int64)
    mach_Qn, mach_Hn = nuisance_machines(mtpmach)
    O = mtpmach.args[1]()
    return bootstrap(mtpmach.model.boot_sampler, mach_Qn, mach_Hn, O, δ, B)
end

function bootstrap_sample(sampler::BootstrapSampler, mach_Qn, mach_Hn, O::CausalTable, δ::Intervention)
    O_sample = bootstrap(sampler, O)
    Y = getresponse(O_sample)
    LAs, LAδ, dAδ, LAδinv, dAδinv = get_summarized_intervened_data(mach_Qn, mach_Hn, O_sample, δ)
    Qn, Qδn, Hn, Hshiftn = estimate_nuisances(mach_Qn, mach_Hn, LAs, LAδ, LAδinv, dAδ, dAδinv)

    outcome_regression_est = outcome_regression_transform(Qδn)
    ipw_est = ipw(Y, Hn)
    onestep_est = onestep(Y, Qn, Qδn, Hn)
    tmle_est = tmle(Y, Qn, Qδn, Hn, Hshiftn)
    return (; outcome_regression = outcome_regression_est, ipw = ipw_est, onestep = onestep_est, tmle = tmle_est)
end

function get_summarized_intervened_data(mach_Qn, mach_Hn, O::CausalTable, δ::Intervention)

    # Apply summary function
    LAs = CausalTables.replacetable(O, TableOperations.select(CausalTables.summarize(O), setdiff(keys(gettable(O)), (getresponsesymbol(O),))...) |> Tables.columntable)

    # Figure out which treatments are summarized and which are purely natural
    treatmentvar = gettreatmentsymbol(LAs)
    summaries = getsummaries(LAs)
    summaryvars = [CausalTables.get_var_to_summarize(s) for s in summaries]
    summarytreatment = keys(summaries)[summaryvars .== treatmentvar]

    # Split the data
    A = gettreatment(LAs)
    Ltbl = TableOperations.select(LAs, vcat(getcontrolssymbols(LAs), keys(summaries)[summaryvars .!= LAs.treatment]...)...) |> Tables.columntable
    Ls = CausalTables.replacetable(LAs, Ltbl)

    LAδ, dAδ = get_intervened_data(LAs, A, Ls, δ, summarytreatment, summaries, treatmentvar)
    LAδinv, dAδinv = get_intervened_data(LAs, A, Ls, inverse(δ), summarytreatment, summaries, treatmentvar)
    return LAs, LAδ, dAδ, LAδinv, dAδinv
end

function get_intervened_data(LAs, A, Ls, Δ::Intervention, summarytreatment, summaries, treatmentvar)

    # Intervene on the treatment
    Aδ = apply_intervention(Δ, A, Ls)
    Aδd = differentiate_intervention(Δ, A, Ls)

    # Intervene on the summarized treatment
    Aδs = Vector{Pair}(undef, length(summarytreatment))
    Aδsd = Vector{Pair}(undef, length(summarytreatment))

    for (i, st) in enumerate(summarytreatment)
        Δsummary = get_induced_intervention(Δ, summaries[st])
        As = Tables.getcolumn(LAs, st)
        Aδs[i] = st => apply_intervention(Δsummary, As, Ls)
        Aδsd[i] = st => differentiate_intervention(Δsummary, As, Ls)
    end

    t = (treatmentvar,)
    Aderivatives = merge(NamedTuple{t}((Aδd,)), NamedTuple(Aδsd))
    Aδinterventions = merge(NamedTuple{t}((Aδ,)), NamedTuple(Aδs))
    LAδinterventions = CausalTables.replacetable(Ls, merge(gettable(Ls), Aδinterventions))
    return LAδinterventions, Aderivatives   
end