
eif(Hn, Y, Qn, Qδn) = Hn .* (Y .- Qn) .+ Qδn

outcome_regression_transform(Qδn::Node) = node(Qδn -> outcome_regression_transform(Qδn), Qδn)
outcome_regression_transform(Qδn::Array) = (; ψ = mean(Qδn))

# define basic estimators
function ipw(Y::Array, Hn::Array)
    ψ = sum(Hn .* Y) / sum(Hn)
    σ2 = var(Hn .* (Y .- ψ)) / length(Hn)
    return (; ψ = ψ, σ2 = σ2)
end

function onestep(Y::Array, Qn::Array, Qδn::Array, Hn::Array)
    D = eif(Hn, Y, Qn, Qδn)
    ψ = mean(D)
    σ2 = var(D) / length(D)
    return (; ψ = ψ, σ2 = σ2)
end

function tmle(Y::Array, Qn::Array, Qδn::Array, Hn::Array, Hshiftn::Array)
    scaler = StatsBase.fit(UnitRangeTransform, Y, dims = 1)
    Y01 = StatsBase.transform(scaler, Y)
    Qn01 = StatsBase.transform(scaler, Qn)
    bound!(Qn01; lower = UNIT_LOWER_BOUND, upper = UNIT_UPPER_BOUND)
    return tmle_fromscaled(Y, Qn, Y01, Qn01, Qδn, Hn, Hshiftn, scaler)
end

function tmle_fromscaled(Y::Array, Qn::Array, Y01::Array, Qn01::Array, Qδn::Array, Hn::Array, Hshiftn::Array, scaler)
    fit_data = MLJBase.table(hcat(Y01, Hn), names = ["Y", "Hn"])
    # Fit the logistic regression model
    # The 0 + is needed to fit the model with an intercept of 0
    fluct_model = GLM.glm(@GLM.formula(Y ~ 0 + Hn), fit_data, Binomial(); offset = qlogis.(Qn01), verbose=false)
    
    # Get the predictions from the logistic regression
    Qδn01 = StatsBase.transform(scaler, Qδn)
    bound!(Qδn01; lower = UNIT_LOWER_BOUND, upper = UNIT_UPPER_BOUND)
    predict_data = MLJBase.table(reshape(Hshiftn, length(Hshiftn), 1), names = ["Hn"])
    Qstar01 = GLM.predict(fluct_model, predict_data, offset = qlogis.(Qδn01))
    Qstar = StatsBase.reconstruct(scaler, identity.(Qstar01)) # `identity`` strips the vector of the "Missing" type
    ψ = mean(Qstar)

    # Estimate variance
    D = eif(Hn, Y, Qn, Qδn)
    σ2 = var(D) / length(D)
    return (; ψ = ψ, σ2 = σ2)
end

# Define MLJ-type estimaator machines for the learning network

abstract type CausalEstimator <: MMI.Unsupervised end

mutable struct IPW <: CausalEstimator end
MMI.fit(::IPW, verbosity, Y) = (fitresult = (; Y = Y), cache = nothing, report = nothing)
MMI.transform(::IPW, fitresult, Hn) = ipw(fitresult.Y, Hn)


mutable struct OneStep <: CausalEstimator end
MMI.fit(::OneStep, verbosity, Y, Qn) = (fitresult = (; Y = Y, Qn = Qn), cache = nothing, report = nothing)
MMI.transform(::OneStep, fitresult, Qδn, Hn) = onestep(fitresult.Y, fitresult.Qn, Qδn, Hn)

mutable struct TMLE <: CausalEstimator end

function MMI.fit(::TMLE, verbosity, Y, Qn)
    scaler = StatsBase.fit(UnitRangeTransform, Y, dims = 1)
    Y01 = StatsBase.transform(scaler, Y)
    Qn01 = StatsBase.transform(scaler, Qn)
    bound!(Qn01; lower = UNIT_LOWER_BOUND, upper = UNIT_UPPER_BOUND)
    (fitresult = (; Y = Y, Qn = Qn, Y01 = Y01, Qn01 = Qn01, scaler = scaler), cache = nothing, report = nothing)
end

MMI.transform(::TMLE, fitresult, Qδn, Hn, Hshiftn) = tmle_fromscaled(fitresult.Y, fitresult.Qn, fitresult.Y01, fitresult.Qn01, Qδn, Hn, Hshiftn, fitresult.scaler)

mutable struct MultiplierBootstrap <: CausalEstimator 
    B::Int
end

MMI.fit(::MultiplierBootstrap, verbosity, Y, Qn) = (fitresult = (; Y = Y, Qn = Qn), cache = nothing, report = nothing)

function MMI.transform(sampler::MultiplierBootstrap, fitresult, Qδn, Hn) 
    n = length(fitresult.Y)
    W = rand(Normal(1, 1), n, sampler.B)

    ξ = eif(Hn, Y, Qn, Qδn)
    Wξ = W .* ξ

    ψ = mean(ξ)
    σ2naive = var(ξ)
    Wψ = mean(Wξ, dims = 1)

    # n's are slightly wrong here, fix them later
    return var(Wψ) - σ2naive / n
end

