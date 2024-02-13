# Define a struct to hold the vector of Machine objects
struct CrossFitModel <: MMI.Model
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

struct DecomposedPropensityRatio
    model::Condensity.ConDensityRatioEstimator
end

function MMI.fit(dp::DecomposedPropensityRatio, verbosity, X::CausalTable, Y)

    cur_treatment_names = Tables.columnnames(Y)
    XY = merge(gettable(X), Y)
    all_col_names = Tables.columnnames(XY)
    machines = Vector{Machine}(undef, length(cur_treatment_names))
    

    for k in (length(treatment_names):-1:1)

        # Construct a table of the target
        target = cur_treatment_names[k]
        Yk = TableOperations.select(Y, target) |> Tables.columntable

        # Construct a covariate table that includes future targets as covariates
        # This iteratively removes the target from the table of all variables
        cur_treatment_names = setdiff(all_col_names, target)
        XY = TableOperations.select(XY, cur_treatment_names) |> Tables.columntable

        # Fit the propensity score ratio model of the current target, controlling for subsequent targets
        machines[k] = fit!(machine(dp.model, XY, Yk), verbosity)
    end

    fitresult = (; machines = machines,)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(model::DecomposedPropensityRatio, fitresult, Xy_nu, Xy_de)
    # return the product of the ratios, based on the decomposition provided in Forastiere et al. (2021)
    return prod([MMI.predict(mach, Xy_nu, Xy_de) for mach in fitresult.machines])
end