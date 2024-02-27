
# TODO: This function only works if the data are IID.
# Need to implement efficiency bound for non-IID data.

function compute_true_MTP(dgp, data, intervention; direction = :out, controls_iid = true)

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
    Hn_aux = ones(length(Y))
    for col in keys(dAδsinv)
        g0_Anat= pdf.(condensity(dgp, data, col), Tables.getcolumn(data, col))
        g0_Ainv = pdf.(condensity(dgp, LAδsinv, col), Tables.getcolumn(LAδs, col))
        Hn_aux = Hn_aux .* (g0_Ainv ./ g0_Anat)
    end
    Hn_aux = Hn_aux .* prod(dAδsinv)
    

    # Compute the EIF and get g-computation result
    ψ = mean(Q0bar_shift)
    D = Hn_aux .* (Y .- Q0bar_noshift) .+ (Q0bar_shift .- ψ)

    if controls_iid
        if nv(getgraph(data)) == 0 # if graph is empty, just use empirical variance
            eff_bound = var(D)
        else # if graph exists, use estimator from Ogburn 2022
            G = get_dependency_neighborhood(getgraph(data))
            eff_bound = matrixvar(D, G)
        end
    else
        error("Non-iid efficiency bound not yet implemented.")
    end

    true_result = (ψ = ψ, eff_bound = eff_bound)
    return true_result
end