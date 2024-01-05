

mutable struct OutcomeRegressor <: Static end
MMI.transform(est::OutcomeRegressor, Qδn::Array) = (ψ = mean(Qδn),)

abstract type CausalEstimator <: MMI.Unsupervised end

mutable struct IPW <: CausalEstimator end
MLJBase.fit(::IPW, verbosity, Y) = (fitresult = (; Y = Y), cache = nothing, report = nothing)

function MLJBase.transform(::IPW, fitresult, Hn)
    ψ = sum(Hn .* fitresult.Y) / sum(Hn)
    σ2 = var(Hn .* (fitresult.Y .- ψ)) / length(Hn)
    return (; ψ = ψ, σ2 = σ2)
end

mutable struct OneStep <: CausalEstimator end
MLJBase.fit(::OneStep, verbosity, Y, Qn) = (fitresult = (; Y = Y, Qn = Qn), cache = nothing, report = nothing)

function MLJBase.transform(::OneStep, fitresult, Qδn, Hn)
    D = eif(Hn, fitresult.Y, fitresult.Qn, Qδn)
    ψ = mean(D)
    σ2 = var(D) / length(D)
    return (; ψ = ψ, σ2 = σ2)
end

mutable struct TMLE <: CausalEstimator end

function MLJBase.fit(::TMLE, verbosity, Y, Qn)
    scaler = StatsBase.fit(UnitRangeTransform, Y, dims = 1)
    Y01 = StatsBase.transform(scaler, Y)
    Qn01 = StatsBase.transform(scaler, Qn)
    bound!(Qn01; lower = UNIT_LOWER_BOUND, upper = UNIT_UPPER_BOUND)
    (fitresult = (; Y = Y, Qn = Qn, Y01 = Y01, Qn01 = Qn01, scaler = scaler), cache = nothing, report = nothing)
end

function MLJBase.transform(::TMLE, fitresult, Qδn, Hn, Hshiftn)
    fit_data = MLJBase.table(hcat(fitresult.Y01, Hn), names = ["Y", "Hn"])

    # Fit the logistic regression model
    # The 0 + is needed to fit the model with an intercept of 0
    fluct_model = GLM.glm(@GLM.formula(Y ~ 0 + Hn), fit_data, Binomial(); offset = qlogis.(fitresult.Qn01), verbose=false)
    
    # Get the predictions from the logistic regression
    Qδn01 = StatsBase.transform(fitresult.scaler, Qδn)
    bound!(Qδn01; lower = UNIT_LOWER_BOUND, upper = UNIT_UPPER_BOUND)
    predict_data = MLJBase.table(reshape(Hshiftn, length(Hshiftn), 1), names = ["Hn"])
    Qstar = StatsBase.reconstruct(fitresult.scaler, GLM.predict(fluct_model, predict_data, offset = qlogis.(Qδn01)))
    ψ = mean(Qstar)

    # Estimate variance
    D = eif(Hn, fitresult.Y, fitresult.Qn, Qδn)
    σ2 = var(D) / length(D)
    return (; ψ = ψ, σ2 = σ2)
end

mutable struct EIFConservative <: MMI.Unsupervised end

MLJBase.fit(::EIFConservative, verbosity, Y, Qn, ct) = (fitresult = (; Y = Y, Qn = Qn, G = CausalTables.getgraph(ct)), cache = nothing, report = nothing)

function MLJBase.transform(::EIFConservative, fitresult, Qδn, Hn)
    D = eif(Hn, fitresult.Y, fitresult.Qn, Qδn)
    σ2c = transpose(D) * adjacency_matrix(fitresult.G) * D / sum(adjacency_matrix(fitresult.G))
    return (; σ2c = σ2c)
end




