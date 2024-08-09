

function compute_true_MTP(dgp, data, intervention)

    # TODO: Currently assumes Y is univariate
    Ysymb = data.response[1]
    
    # Organize and transform the data
    Y = Tables.getcolumn(CausalTables.response(data), Ysymb)
    intmach = machine(InterventionModel(), data) |> fit!
    LAs, _, _ = MLJBase.predict(intmach, intervention)
    LAδs, _ = transform(intmach, intervention)
    LAδsinv, dAδsinv = inverse_transform(intmach, intervention)

    # Compute conditional means of response
    Q0bar_noshift = conmean(dgp, data, Ysymb)
    Q0bar_shift = conmean(dgp, CausalTables.replace(LAδs; tbl = merge(LAδs.data, (Y = Y,))), Ysymb)

    # Compute conditional density ratio of treatment
    Hn_aux = ones(length(Y))
    for col in keys(dAδsinv)
        g0_Anat= pdf.(condensity(dgp, LAs, col), Tables.getcolumn(LAs, col))
        g0_Ainv = pdf.(condensity(dgp, LAδsinv, col), Tables.getcolumn(LAδsinv, col))
        Hn_aux = Hn_aux .* (g0_Ainv ./ g0_Anat)
    end
    Hn_aux = Hn_aux .* prod(dAδsinv)
    
    # Compute the EIF and get g-computation result
    ψ = mean(Q0bar_shift)

    GA = CausalTables.adjacency_matrix(data)
    GD = CausalTables.dependency_matrix(data)
    D = Hn_aux .* (Y .- Q0bar_noshift) .+ Q0bar_shift
    eff_bound = network_variance(D, GA, GD) * length(D)
    true_result = (ψ = ψ, ψ_dif = ψ - mean(Y), eff_bound = eff_bound)
    return true_result
end