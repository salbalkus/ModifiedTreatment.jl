mutable struct MTP <: UnsupervisedNetworkComposite
    mean_estimator
    density_ratio_estimator
    boot_sampler
    cv_splitter
    confidence::Float64
end

MTP(mean_estimator, density_ratio_estimator, boot_sampler, cv_splitter) = MTP(mean_estimator, density_ratio_estimator, boot_sampler, cv_splitter, 0.95)

function MLJBase.prefit(mtp::MTP, verbosity, O::CausalTable)
    estimators = (OutcomeRegressor(), IPW(), OneStep(), TMLE())
    δ = source(IdentityIntervention())
    Os = source(O)

    Y = getresponse(Os)

    LAs, Ls, As, LAδs, dAδs, LAδsinv, dAδsinv = intervene_on_data(Os, δ)
    mach_mean, mach_density = crossfit_nuisance_estimators(mtp, Y, LAs, Ls, As)

    Qn, Qδn, Hn, Hshiftn = estimate_nuisances(mach_mean, mach_density, LAs, LAδs, LAδsinv, dAδs, dAδsinv)
    outcome_regression, ipw, onestep, tmle = estimate_causal_parameters(estimators, Y, Qn, Qδn, Hn, Hshiftn)

    #=
    # Get outcome regression predictions
    Qn = MMI.predict(mach_mean, LAs)
    Qδn = MMI.predict(mach_mean, LAδs)

    # Get Density Ratio
    Hn = MMI.predict(mach_density, LAδsinv, LAs) * prod(dAδs)
    Hshiftn = MMI.predict(mach_density, LAs, LAδs) * prod(dAδsinv)

    outcome_regression, ipw, onestep, tmle = estimate_causal(estimators, Y, Qn, Qδn, Hn, Hshiftn)
    =#
    # Conservative EIF variance estimate
    #mach_consvar = machine(MTP.EIFConservative(), Y, Qn, gdf_source) |> fit!
    #consvar = MMI.transform(mach_consvar, Qδn, Hn)
    #onestep = merge(estimates[3], consvar)
    #tmle = merge(estimates[4], consvar)

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

# Define custom functions akin to `predict` that yield the result of each estimation strategy from a learning network machine
outcome_regression(machine, δnew) = MLJBase.unwrap(machine.fitresult).outcome_regression(δnew)
ipw(machine, δnew) = MLJBase.unwrap(machine.fitresult).ipw(δnew)
onestep(machine, δnew) = MLJBase.unwrap(machine.fitresult).onestep(δnew)
tmle(machine, δnew) = MLJBase.unwrap(machine.fitresult).tmle(δnew)

function intervene_on_data(Os, δ)
    model_intervention = InterventionModel()
    mach_intervention = machine(model_intervention, Os) |> fit!

    tmp = MMI.predict(mach_intervention, δ)
    LAs, Ls, As = tmp[1], tmp[2], tmp[3]
    
    tmp = MMI.transform(mach_intervention, δ)
    LAδs, dAδs = tmp[1], tmp[2]

    tmp = MMI.inverse_transform(mach_intervention, δ)
    LAδsinv, dAδsinv = tmp[1], tmp[2]

    return LAs, Ls, As, LAδs, dAδs, LAδsinv, dAδsinv
end

function crossfit_nuisance_estimators(mtp, Y, LAs, Ls, As)
    if isnothing(mtp.cv_splitter)
        mach_mean = machine(mtp.mean_estimator, LAs, Y)
        mach_density = machine(mtp.density_ratio_estimator, Ls, As)
    else
        mach_mean = machine(CrossFitModel(mtp.mean_estimator, mtp.cv_splitter), LAs, Y)
        mach_density = machine(CrossFitModel(mtp.density_ratio_estimator, mtp.cv_splitter), Ls, As)
    end

    return mach_mean, mach_density
end

function estimate_nuisances(mach_mean, mach_density, LAs, LAδs, LAδsinv, dAδs, dAδsinv)
    # Get Conditional Mean
    Qn = MMI.predict(mach_mean, LAs)
    Qδn = MMI.predict(mach_mean, LAδs)

    # Get Density Ratio
    Hn = MMI.predict(mach_density, LAδsinv, LAs) * prod(dAδs)
    Hshiftn = MMI.predict(mach_density, LAs, LAδs) * prod(dAδsinv)

    return Qn, Qδn, Hn, Hshiftn
end

function estimate_causal_parameters(estimators::Tuple, Y, Qn, Qδn, Hn, Hshiftn)
    mach_outcome_regression = machine(estimators[1])
    mach_ipw = machine(estimators[2], Y) |> fit!
    mach_onestep = machine(estimators[3], Y, Qn) |> fit!
    mach_tmle = machine(estimators[4], Y, Qn) |> fit!

    outcome_regression = MMI.transform(mach_outcome_regression, Qδn)
    ipw = MMI.transform(mach_ipw, Hn)
    onestep = MMI.transform(mach_onestep, Qδn, Hn)
    tmle = MMI.transform(mach_tmle, Qδn, Hn, Hshiftn)

    return outcome_regression, ipw, onestep, tmle
end

#=
function bootstrap_estimates(mtp::MTP, O::AbstractNode, δ::AbstractNode, estimators::Tuple)
    
end
=#