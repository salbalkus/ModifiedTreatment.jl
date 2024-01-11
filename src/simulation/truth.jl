
# TODO: This function only works if the data are IID.
# Need to implement efficiency bound for non-IID data.

function compute_true_MTP(dgp, data, intervention)
    # Organize and transform the data
    Y = getresponse(data)
    intmach = machine(InterventionModel(), data) |> fit!
    LAδs, _ = transform(intmach, intervention)
    LAδsinv, dAδsinv = inverse_transform(intmach, intervention)

    # Compute conditional means of response
    Q0bar_noshift = conmean(dgp, data, :Y)
    Q0bar_shift = conmean(dgp, LAδs, :Y)

    # Compute conditional density ratio of treatment
    g0_Anat= pdf.(condensity(dgp, data, :A), Tables.getcolumn(data, :A))
    g0_Ainv = pdf.(condensity(dgp, LAδsinv, :A), Tables.getcolumn(LAδs, :A))
    Hn_aux = g0_Ainv ./ g0_Anat .* prod(dAδsinv)

    # Compute the EIF and get g-computationa result
    eif_shift = Hn_aux .* (Y .- Q0bar_noshift) .+ (Q0bar_shift .- mean(Q0bar_shift))
    true_result = (ψ = mean(Q0bar_shift), eff_bound = var(eif_shift))
    return true_result
end