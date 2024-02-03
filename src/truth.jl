
# TODO: This function only works if the data are IID.
# Need to implement efficiency bound for non-IID data.

function compute_true_MTP(dgp, data, intervention; direction = :out, controls_iid = true)
    # get symbols
    Asymb = gettreatmentsymbol(data)
    Ysymb = getresponsesymbol(data)
    
    # Organize and transform the data
    Y = getresponse(data)
    intmach = machine(InterventionModel(), data) |> fit!
    LAδs, _ = transform(intmach, intervention)
    LAδsinv, dAδsinv = inverse_transform(intmach, intervention)

    # Compute conditional means of response
    Q0bar_noshift = conmean(dgp, data, Ysymb)
    Q0bar_shift = conmean(dgp, CausalTables.replace(LAδs; tbl = merge(gettable(LAδs), (Y = Y,))), :Y)

    # Compute conditional density ratio of treatment
    g0_Anat= pdf.(condensity(dgp, data, Asymb), Tables.getcolumn(data, Asymb))
    g0_Ainv = pdf.(condensity(dgp, LAδsinv, Asymb), Tables.getcolumn(LAδs, Asymb))
    Hn_aux = g0_Ainv ./ g0_Anat .* prod(dAδsinv)

    # Compute the EIF and get g-computation result
    ψ = mean(Q0bar_shift)
    eif_shift = Hn_aux .* (Y .- Q0bar_noshift) .+ (Q0bar_shift .- ψ)

    if controls_iid
        if nv(getgraph(data)) == 0 # if graph is empty, just use empirical variance
            eff_bound = var(eif_shift)
        else # if graph exists, use estimator from Ogburn 2022
            eff_bound = (transpose(eif_shift) * adjacency_matrix(getgraph(data); dir = direction) * eif_shift) / length(eif_shift)
        end
    else
        error("Non-iid efficiency bound not yet implemented.")
    end

    true_result = (ψ = ψ, eff_bound = eff_bound)
    return true_result
end