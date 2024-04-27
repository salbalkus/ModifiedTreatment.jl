# Define a struct to hold the vector of Machine objects
mutable struct CrossFitModel <: MMI.Model
    model::MMI.Model
    resampling::MT.ResamplingStrategy
end

function MMI.fit(cfm::CrossFitModel, verbosity, X, y)

    # Get the indices for each fold
    tt_pairs = MLJBase.train_test_pairs(cfm.resampling, 1:DataAPI.nrow(X), X, y)

    # Fit each fold
    machines = [fit!(machine(cfm.model, X, y), rows = training_rows) for (training_rows, _) in tt_pairs] 

    fitresult = (; machines = machines, tt_pairs = tt_pairs)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

MMI.predict(cfm::CrossFitModel, fitresult, X...)  = vcat([MMI.predict(fitresult.machines[i], [Tables.subset(x, fitresult.tt_pairs[i][2]) for x in X]...) for i in 1:length(fitresult.machines)]...)
MMI.transform(cfm::CrossFitModel, fitresult, X...)  = vcat([MMI.transform(fitresult.machines[i], [Tables.subset(x, fitresult.tt_pairs[i][2]) for x in X]...) for i in 1:length(fitresult.machines)]...)

# TODO: Should we somehow specify this model must be a conditional density estimator specifically?
struct DecomposedPropensityRatio <: MMI.Model
    model::MMI.Model
end

function MMI.fit(dp::DecomposedPropensityRatio, verbosity, X, Y)

    # ASSUME that treatment names are in REVERSE order of conditional dependence
    # TODO: fix this!
    cur_treatment_names = reverse(Tables.columnnames(Y))
    XY = merge(Tables.columntable(X), Tables.columntable(Y))
    all_col_names = Tables.columnnames(XY)
    machines = Vector{Machine}(undef, length(cur_treatment_names))
    inclusions = Vector{Tuple}(undef, length(cur_treatment_names))

    for k in 1:length(cur_treatment_names)

        # track which variables are being included at each iteration
        inclusions[k] = all_col_names

        # Construct a table of the target
        target = cur_treatment_names[k]
        Yk = TableOperations.select(Y, target) |> Tables.columntable

        # Construct a covariate table that includes future targets as covariates
        # This iteratively removes the target from the table of all variables
        all_col_names = filter(x -> x != target, all_col_names)
        XY = TableOperations.select(XY, all_col_names...) |> Tables.columntable

        # Fit the propensity score ratio model of the current target, controlling for subsequent targets
        machines[k] = fit!(machine(dp.model, XY, Yk))
    end

    fitresult = (; machines = machines, inclusions = inclusions)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(::DecomposedPropensityRatio, fitresult, Xy_nu, Xy_de)
    # return the product of the ratios, based on the decomposition provided in Forastiere et al. (2021)
    output = ones(DataAPI.nrow(Xy_nu))
    for i in 1:length(fitresult.machines)
        Xy_nu_i = replacetable(Xy_nu, TableOperations.select(Xy_nu, fitresult.inclusions[i]...) |> Tables.columntable)
        Xy_de_i = replacetable(Xy_de, TableOperations.select(Xy_de, fitresult.inclusions[i]...) |> Tables.columntable)
        pred = MMI.predict(fitresult.machines[i], Xy_nu_i, Xy_de_i)
        output = output .* pred
    end
    return output
end

mutable struct SuperLearnerDeterministic <: MMI.Deterministic 
    models::Array{MMI.Deterministic, 1}
    resampling::MT.ResamplingStrategy
    SuperLearnerDeterministic(models::Array{MMI.Deterministic, 1}, resampling = CV()) = new(models, resampling)
end

function MMI.fit(sl::SuperLearnerDeterministic, verbosity, X, y)

    measurements = map(m -> evaluate(m, X, y, resampling = sl.resampling, measure = rmse, verbosity = -1).measurement, sl.models)
    best_model = sl.models[argmin(measurements)]
    
    best_mach = machine(best_model, X, y) |> fit!
    
    fitresult = (; best_mach = best_mach,)
    cache = nothing
    report = (; measurements = measurements, best_model = best_model)
    return fitresult, cache, report
end

MMI.predict(sl::SuperLearnerDeterministic, fitresult, X) = MMI.predict(fitresult.best_mach, X)

mutable struct SuperLearnerProbabilistic <: MMI.Probabilistic 
    models::Array{MMI.Probabilistic , 1}
    resampling::MT.ResamplingStrategy
    SuperLearnerProbabilistic(models::Array{MMI.Probabilistic , 1}, resampling = CV()) = new(models, resampling)
end

function MMI.fit(sl::SuperLearnerProbabilistic, verbosity, X, y)

    measurements = map(m -> evaluate(m, X, y, resampling = sl.resampling, measure = log_loss, verbosity = -1).measurement, sl.models)
    best_model = sl.models[argmin(measurements)]
    
    best_mach = machine(best_model, X, y) |> fit!
    
    fitresult = (; best_mach = best_mach,)
    cache = nothing
    report = (; measurements = measurements, best_model = best_model)
    return fitresult, cache, report
end

MMI.predict(sl::SuperLearnerProbabilistic, fitresult, X) = MMI.predict(fitresult.best_mach, X)





SuperLearner(models::Array{MMI.Deterministic, 1}, resampling) = SuperLearnerDeterministic(models, resampling)
SuperLearner(models::Array{MMI.Probabilistic, 1}, resampling) = SuperLearnerProbabilistic(models, resampling)



