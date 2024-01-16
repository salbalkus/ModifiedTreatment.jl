mutable struct MTP <: UnsupervisedNetworkComposite
    mean_estimator
    density_ratio_estimator
    boot_sampler
    cv_splitter
    confidence::Float64
end

MTP(mean_estimator, density_ratio_estimator, boot_sampler, cv_splitter) = MTP(mean_estimator, density_ratio_estimator, boot_sampler, cv_splitter, 0.95)

function MLJBase.prefit(mtp::MTP, verbosity, O::CausalTable)
    estimators = (IPW(), OneStep(), TMLE())
    δ = source(IdentityIntervention())
    Os = source(O)

    Y = getresponse(Os)
    model_intervention = InterventionModel()
    LAs, Ls, As, LAδs, dAδs, LAδsinv, dAδsinv = intervene_on_data(model_intervention, Os, δ)
    mach_mean, mach_density = crossfit_nuisance_estimators(mtp, Y, LAs, Ls, As)

    Qn, Qδn, Hn, Hshiftn = estimate_nuisances(mach_mean, mach_density, LAs, LAδs, LAδsinv, dAδs, dAδsinv)
    outcome_regression, ipw, onestep, tmle = estimate_causal_parameters(estimators, Y, Qn, Qδn, Hn, Hshiftn)

    # Conservative EIF variance estimate
    #mach_consvar = machine(MTP.EIFConservative(), Y, Qn, gdf_source) |> fit!
    #consvar = MMI.transform(mach_consvar, Qδn, Hn)
    #onestep = merge(estimates[3], consvar)
    #tmle = merge(estimates[4], consvar)

    return (; 
        outcome_regression = outcome_regression_final,
        ipw = ipw_final,
        onestep = onestep_final,
        tmle = tmle_final,
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

function intervene_on_data(model_intervention, Os, δ)
    # TODO: Bug here; this is retraining multiple times (during bootstrap?) which we don't want
    mach_intervention = machine(model_intervention, Os)

    tmp1 = MMI.predict(mach_intervention, δ)
    tmp2 = MMI.transform(mach_intervention, δ)
    tmp3 = MMI.inverse_transform(mach_intervention, δ)

    LAs, Ls, As = tmp1[1], tmp1[2], tmp1[3]
    LAδs, dAδs = tmp2[1], tmp2[2]
    LAδsinv, dAδsinv = tmp3[1], tmp3[2]

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
    mach_ipw = machine(estimators[1], Y) |> fit!
    mach_onestep = machine(estimators[2], Y, Qn) |> fit!
    mach_tmle = machine(estimators[3], Y, Qn) |> fit!

    outcome_regression_est = outcome_regression_transform(Qδn)
    ipw_est = MMI.transform(mach_ipw, Hn)
    onestep_est = MMI.transform(mach_onestep, Qδn, Hn)
    tmle_est = MMI.transform(mach_tmle, Qδn, Hn, Hshiftn)

    return outcome_regression_est, ipw_est, onestep_est, tmle_est
end

