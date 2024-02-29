

# CausalEstimatorResult objects
abstract type CausalEstimatorResult end

mutable struct PlugInResult <: CausalEstimatorResult
    ψ::Estimate
    σ2boot::Estimate
end
PlugInResult(ψ) = PlugInResult(ψ, nothing)

mutable struct IPWResult <: CausalEstimatorResult
    ψ::Estimate
    σ2::Estimate
    σ2net::Estimate
    σ2boot::Estimate
end
IPWResult(ψ, σ2) = IPWResult(ψ, σ2, nothing, nothing)
IPWResult(ψ, σ2, σ2net) = IPWResult(ψ, σ2, σ2net, nothing)


mutable struct OneStepResult <: CausalEstimatorResult
    ψ::Estimate
    σ2::Estimate
    σ2net::Estimate
    σ2boot::Estimate
end
OneStepResult(ψ, σ2) = OneStepResult(ψ, σ2, nothing, nothing)
OneStepResult(ψ, σ2, σ2net) = OneStepResult(ψ, σ2, σ2net, nothing)


mutable struct TMLEResult <: CausalEstimatorResult
    ψ::Estimate
    σ2::Estimate
    σ2net::Estimate
    σ2boot::Estimate
end
TMLEResult(ψ, σ2) = TMLEResult(ψ, σ2, nothing, nothing)
TMLEResult(ψ, σ2, σ2net) = TMLEResult(ψ, σ2, σ2net, nothing)


# functions to compute estimators from nuisance parameters

eif(Hn, Y, Qn, Qδn) = Hn .* (Y .- Qn) .+ Qδn

plugin_transform(Qδn::Vector) = PlugInResult(mean(Qδn))
plugin_transform(Qδn::Node) = node(Qδn -> plugin_transform(Qδn), Qδn)

# define basic estimators
function ipw(Y::Array, Hn::Array, G::AbstractMatrix)

    # use stabilized point estimate
    ψ = mean((Hn ./ mean(Hn)) .* Y) 

    # use non-stabilized variance estimate
    estimating_function = (Hn .* Y)
    estimating_function = estimating_function .- mean(estimating_function)
    σ2 = (estimating_function' * estimating_function) / (length(estimating_function)^2)

    if isnothing(G) || size(G, 1) == 0
        return IPWResult(ψ, σ2)
    else
        σ2net = cov_unscaled(estimating_function, G) / (length(estimating_function)^2)
        return IPWResult(ψ, σ2, σ2net)
    end
end

function onestep(Y::Array, Qn::Array, Qδn::Array, Hn::Array, G::AbstractMatrix)
    D = eif(Hn, Y, Qn, Qδn)
    ψ = mean(D)
    D = D .- ψ
    σ2 = mean(D.^2) / length(D)
    if isnothing(G) || size(G, 1) == 0
        return OneStepResult(ψ, σ2)
    else
        σ2net = cov_unscaled(D, G) / (length(D)^2)
        return OneStepResult(ψ, σ2, σ2net)
    end
end

function tmle(Y::Array, Qn::Array, Qδn::Array, Hn::Array, Hshiftn::Array, G::AbstractMatrix)
    scaler = StatsBase.fit(UnitRangeTransform, Y, dims = 1)
    Y01 = StatsBase.transform(scaler, Y)
    Qn01 = StatsBase.transform(scaler, Qn)
    bound!(Qn01; lower = UNIT_LOWER_BOUND, upper = UNIT_UPPER_BOUND)
    return tmle_fromscaled(Y, Qn, Y01, Qn01, Qδn, Hn, Hshiftn, G, scaler)
end

function tmle_fromscaled(Y::Array, Qn::Array, Y01::Array, Qn01::Array, Qδn::Array, Hn::Array, Hshiftn::Array, G::AbstractMatrix, scaler)
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
    D = eif(Hn, Y, Qn, Qδn) .- ψ
    σ2 = mean(D.^2) / length(D)
    if isnothing(G) || size(G, 1) == 0
        return TMLEResult(ψ, σ2)
    else
        σ2net = cov_unscaled(D, G) / (length(D)^2)
        return TMLEResult(ψ, σ2, σ2net)
    end
end

# Define MLJ-type estimaator machines for the learning network

abstract type CausalEstimator <: MMI.Unsupervised end

mutable struct IPW <: CausalEstimator end
MMI.fit(::IPW, verbosity, Y, G) = (fitresult = (; Y = Y, G = G), cache = nothing, report = nothing)
MMI.transform(::IPW, fitresult, Hn) = ipw(fitresult.Y, Hn, fitresult.G)


mutable struct OneStep <: CausalEstimator end
MMI.fit(::OneStep, verbosity, Y, Qn, G) = (fitresult = (; Y = Y, Qn = Qn, G = G), cache = nothing, report = nothing)
MMI.transform(::OneStep, fitresult, Qδn, Hn) = onestep(fitresult.Y, fitresult.Qn, Qδn, Hn, fitresult.G)

mutable struct TMLE <: CausalEstimator end

function MMI.fit(::TMLE, verbosity, Y, Qn, G)
    scaler = StatsBase.fit(UnitRangeTransform, Y, dims = 1)
    Y01 = StatsBase.transform(scaler, Y)
    Qn01 = StatsBase.transform(scaler, Qn)
    bound!(Qn01; lower = UNIT_LOWER_BOUND, upper = UNIT_UPPER_BOUND)
    (fitresult = (; Y = Y, Qn = Qn, Y01 = Y01, Qn01 = Qn01, G = G, scaler = scaler), cache = nothing, report = nothing)
end

MMI.transform(::TMLE, fitresult, Qδn, Hn, Hshiftn) = tmle_fromscaled(fitresult.Y, fitresult.Qn, fitresult.Y01, fitresult.Qn01, Qδn, Hn, Hshiftn, fitresult.G, fitresult.scaler)

mutable struct MultiplierBootstrap <: CausalEstimator 
    B::Int
end

MMI.fit(::MultiplierBootstrap, verbosity, Y, Qn) = (fitresult = (; Y = Y, Qn = Qn), cache = nothing, report = nothing)

function MMI.transform(sampler::MultiplierBootstrap, fitresult, Qδn, Hn) 
    n = length(fitresult.Y)

    # Compute the EIF components
    ξ = eif(Hn, Y, Qn, Qδn)
    ψ = mean(ξ)
    D = ξ .- ψ
    σ2naive = var(D)


    # Randomly perturb the EIF over many samples
    W = rand(Normal(1, 1), n, sampler.B)
    WD = W .* D

    # Compute many "perturbed resamples" of ψ, compute their variance, and subtract off the naive bias
    Wψ = mean(WD, dims = 1)
    return (var(Wψ) - σ2naive) / n
end

mutable struct MTPResult
    result::Union{CausalEstimatorResult, NamedTuple}
    mtp::Machine
    intervention::Intervention
end
getestimate(x::MTPResult) = x.result
getmtp(x::MTPResult) = x.mtp
getintervention(x::MTPResult) = x.intervention

function ψ(x::MTPResult)
    est = getestimate(x)
    if typeof(est) == CausalEstimatorResult
        return est.ψ
    else
        return map(e -> e.ψ, est)
    end
end
function σ2(x::MTPResult)
    est = getestimate(x)
    if typeof(est) == CausalEstimatorResult
        return est.σ2
    else
        return map(e -> hasproperty(e, :σ2) ? e.σ2 : nothing, est)
    end
end
function σ2boot(x::MTPResult)
    est = getestimate(x)
    if typeof(est) == CausalEstimatorResult
        return est.σ2boot
    else
        return map(e -> hasproperty(e, :σ2boot) ?  e.σ2boot : nothing, est)
    end
end
function σ2net(x::MTPResult)
    est = getestimate(x)
    if typeof(est) == CausalEstimatorResult
        return est.σ2net
    else
        return map(e -> hasproperty(e, :σ2net) ?  e.σ2net : nothing, est)
    end
end






