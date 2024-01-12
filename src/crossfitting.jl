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

