mutable struct MTP <: UnsupervisedNetworkComposite
    mean_estimator
    density_ratio_estimator
    boot_sampler
    cv_splitter
    confidence::Float64
end

MTP(mean_estimator, density_ratio_estimator, boot_sampler, cv_splitter) = MTP(mean_estimator, density_ratio_estimator, boot_sampler, cv_splitter, 0.95)

function MLJBase.prefit(mtp::MTP, verbosity, O::CausalTable)

    δ = source(0)
    O = source(O)
    Y = getresponse(O)

    model_intervention = InterventionModel()
    mach_intervention = machine(model_intervention, O)

    LAs, Ls, As = MMI.predict(mach_intervention, δ)

    LAδs, dAδs = transform(intmach, intervention)
    LAδsinv, dAδsinv = inverse_transform(intmach, intervention)

    if isnothing(mtp.cv_splitter)
        mach_mean = machine(mtp.mean_estimator, LAs, Y)
        mach_density = machine(mtp.density_ratio_estimator, Ls, As)
    else
        mach_mean = machine(CrossFitModel(:mean_estimator, mtp.cv_splitter), LAs, Y)
        mach_density = machine(CrossFitModel(:density_ratio_estimator, mtp.cv_splitter), Ls, As)
    end

    # Get outcome regression predictions
    Qn = MMI.predict(mach_mean, LAs)
    Qδn = MMI.predict(mach_mean, LAδs)

    # Get Density Ratio
    Hn = MMI.transform(mach_density, LAδsinv, LAs) .* dAδs
    Hshiftn = MMI.transform(mach_density, LAs, LAδs) .* dAδsinv

    static_vars = node((Y, Qn) -> (Y = Y, Qn = Qn), Y, Qn)
    policy_vars = node((Qδn, Hn, Hshiftn) -> (Qδn = Qδn, Hn = Hn, Hshiftn = Hshiftn), Qδn, Hn, Hshiftn)

    estimators = (OutcomeRegressor(), IPW(), OneStep(), TMLE())
    estimates = estimate_causal(estimators, static_vars, policy_vars)

    # Conservative EIF variance estimate
    mach_consvar = machine(MTP.EIFConservative(), Y, Qn, gdf_source) |> fit!
    consvar = MMI.transform(mach_consvar, Qδn, Hn)
    onestep = merge(estimates[3], consvar)
    tmle = merge(estimates[4], consvar)

    # Bootstrap
    #=
    if mtp.boot_sampler.B > 0
        bootstrap_estimates = bootstrap_causal(mtp, O, δ, estimators)
    end

    for (i, be) in enumerate(bootstrap_estimates)
        estimates[i] = merge(estimates[i], be)
    end
    =#

    return (; 
        outcome_regression = outcome_regression,
        ipw = ipw,
        onestep = onestep,
        tmle = tmle,
        report = (; 
            Qn = Qn,
            Qδn = Qδn,
            Hn = Hn,
            Hshiftn = Hshiftn,
        ),
        #fitted_params = params
    )    

end

estimate_causal(estimators::Tuple, static_vars::AbstractNode, policy_vars::AbstractNode) = node((SV, P) -> estimate_causal(estimators, S, P), static_vars, policy_vars)
function estimate_causal(estimators::Tuple, static_vars::NamedTuple, policy_vars::NamedTuple)
    
    mach_outcome_regression = machine(estimators[1]) |> fit!
    mach_ipw = machine(estimators[2], static_vars.Y) |> fit!
    mach_onestep = machine(estimators[3], static_vars.Y, static_vars.Qn) |> fit!
    mach_tmle = machine(estimators[4], static_vars.Y, static_vars.Qn) |> fit!

    outcome_regression = MMI.transform(mach_outcome_regression, static_vars.Qn)
    ipw = MMI.transform(mach_ipw, policy_vars.Hn)
    onestep = MMI.transform(mach_onestep, policy_vars.Qδn, policy_vars.Hn)
    tmle = MMI.transform(mach_tmle, policy_vars.Qδn, policy_vars.Hn, policy_vars.Hshiftn)

    return [outcome_regression, ipw, onestep, tmle]
end

#=
function bootstrap_estimates(mtp::MTP, O::AbstractNode, δ::AbstractNode, estimators::Tuple)
    
end
=#