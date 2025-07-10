mutable struct SumRatioHSE <: Condensity.ConDensityRatioEstimatorFixed
    location_model::MMI.Supervised
    scale_model::MMI.Supervised
    density_model::Condensity.DensityEstimator
    r_density
    resampling::MT.ResamplingStrategy

    function SumRatioHSE(location_model::MMI.Supervised, scale_model::MMI.Supervised, density_model::Condensity.DensityEstimator, r_density, resampling::MT.ResamplingStrategy)
        new(location_model, scale_model, density_model, r_density, resampling)
    end
end

function fit_density(model::SumRatioHSE, verbosity, X, y, ys, G)
    
    # Fit the location model
    location_mach = machine(model.location_model, X, y) |> fit!
    μ = MMI.predict(location_mach, X)

    # Fit the scale model
    ε = @. y - μ
    ε2 = @. ε^2
    min_obs_ε2 = 2*minimum(ε2)
    scale_mach = machine(model.scale_model, X, ε2) |> fit!

    # Fit the density model
    σ2 = MMI.predict(scale_mach, X)
    σ2[σ2 .<= 0] .= min_obs_ε2
    ε = @. ε / sqrt(σ2)

    # Bound the errors
    ε = @. max(ε, -10)
    ε = @. min(ε, 10)

    tuned_density_model = MT.TunedModel(
        # TODO: Pick better default bandwidth?
        model = model.density_model,
        # TODO: Choose better MT.TuningStrategy
        tuning = MT.Grid(resolution = 100),
        resampling = model.resampling,
        measure = negmeanloglik,
        operation = MMI.predict,
        range = model.r_density,
        )
    
    density_mach = fit!(machine(tuned_density_model, (ε = ε,), zeros(length(ε))), verbosity = -1)

    μs = G * μ
    σ2s = G * σ2
    σ2s[σ2s .<= 0] .= min_obs_ε2
    rootσ2s = sqrt.(σ2s)
    εs = (ys .- μs) ./ rootσ2s

    # Bound the errors
    εs = @. max(ε, -10)
    εs = @. min(ε, 10)

    tuned_density_model_sum = MT.TunedModel(
        # TODO: Pick better default bandwidth?
        model = model.density_model,
        # TODO: Choose better MT.TuningStrategy
        tuning = MT.Grid(resolution = 100),
        resampling = model.resampling,
        measure = negmeanloglik,
        operation = MMI.predict,
        range = model.r_density,
        )

    sum_density_mach = fit!(machine(tuned_density_model_sum, (ε = εs,), zeros(length(ε))), verbosity = -1)


    return location_mach, scale_mach, density_mach, sum_density_mach, min_obs_ε2
end

# Assumes the first column of y is the non-summarized variable
function MMI.fit(model::SumRatioHSE, verbosity, X, y)

    treatmentnames = collect(Tables.columnnames(y))
    y_vec = Tables.getcolumn(y, treatmentnames[1])
    ys_vec = Tables.getcolumn(y, treatmentnames[2])
    G = X.arrays[X.summaries[treatmentnames[2]].matrix]
    location_mach, scale_mach, density_mach, sum_density_mach, min_obs_ε2 = fit_density(model, verbosity, X, y_vec, ys_vec, G)
    
    fitresult = (location_mach = location_mach,  
                 scale_mach = scale_mach, 
                 density_mach = density_mach, 
                 sum_density_mach = sum_density_mach,
                 min_obs_ε2 = min_obs_ε2, 
                 treatmentnames = treatmentnames
                 )
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function predict_density(location_mach, scale_mach, density_mach, sum_density_mach, min_obs_ε2, X, y, ys, G)

    # Get residual model predictions
    μ = MMI.predict(location_mach, X)
    σ2 = MMI.predict(scale_mach, X)
    σ2[σ2 .<= 0] .= min_obs_ε2
    rootσ2 = @. sqrt(σ2)

    # Return density of standardized residual 
    ε = @. (y - μ) / rootσ2
    density  = MMI.predict(density_mach, (ε = ε,)) ./ rootσ2

    # Now compute the same, but for the summaries
    μs = G * μ
    σ2s = G * σ2
    σ2s[σ2s .<= 0] .= min_obs_ε2
    rootσ2s = sqrt.(σ2s)
    εs = (ys .- μs) ./ rootσ2s
    density_s = MMI.predict(sum_density_mach, (ε = εs,)) ./ rootσ2s

    return density, density_s
end

function MMI.predict(model::SumRatioHSE, fitresult, Xy_nu, Xy_de) 

    # split off components of CausalTable
    y_nu = Tables.getcolumn(Xy_nu, fitresult.treatmentnames[1])
    ys_nu = Tables.getcolumn(Xy_nu, fitresult.treatmentnames[2])
    X_nu = CausalTables.reject(Xy_nu, fitresult.treatmentnames)
    G_nu = CausalTables.adjacency_matrix(Xy_nu)


    y_de = Tables.getcolumn(Xy_de, fitresult.treatmentnames[1])
    ys_de = Tables.getcolumn(Xy_de, fitresult.treatmentnames[2])
    X_de = CausalTables.reject(Xy_de, fitresult.treatmentnames)
    G_de = CausalTables.adjacency_matrix(Xy_de)

    # compute density
    g_nu, gs_nu = predict_density(fitresult.location_mach, 
                                fitresult.scale_mach, 
                                fitresult.density_mach, 
                                fitresult.sum_density_mach,
                                fitresult.min_obs_ε2, 
                                X_nu, y_nu, ys_nu, G_nu)
    g_de, gs_de = predict_density(fitresult.location_mach, 
                                    fitresult.scale_mach, 
                                    fitresult.density_mach, 
                                    fitresult.sum_density_mach,
                                    fitresult.min_obs_ε2, 
                                 X_de, y_de, ys_de, G_de)
    
    # Use the densities to compute the density ratio
    Hn = (g_nu ./ g_de) .* (gs_nu ./ gs_de)

    # Bound
    Hn[Hn .> 5] .= 5
    Hn[isnan.(Hn)] .= 0.0
    return Hn
end