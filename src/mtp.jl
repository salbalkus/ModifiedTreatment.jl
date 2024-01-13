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

    if mtp.boot_sampler.B > 0
        outcome_regression_boot, ipw_boot, onestep_boot, tmle_boot = bootstrap_estimates(mtp, estimators, mach_mean, mach_density, Os, δ)
    end

    outcome_regression_final = merge(outcome_regression, outcome_regression_boot)
    ipw_final = merge(ipw, ipw_boot)
    onestep_final = merge(onestep, onestep_boot)
    tmle_final = merge(tmle, tmle_boot)

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
    mach_intervention = machine(model_intervention, Os) |> fit!

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

function bootstrap_estimates(mtp, estimators, mach_mean, mach_density, Os, δ)
    Os_b = bootstrap_samples(mtp.boot_sampler, Os)
    Y = node(Os_b -> Iterators.map(getresponse, Os_b), Os_b)
    model_intervention = ResampledModel(InterventionModel())
    LAs, _, _, LAδs, dAδs, LAδsinv, dAδsinv = intervene_on_data(model_intervention, Os_b, δ)
    Qn, Qδn, Hn, Hshiftn = resample_nuisances(mach_mean, mach_density, LAs, LAδs, LAδsinv, dAδs, dAδsinv)
    
    outcome_regression, ipw, onestep, tmle = resample_causal_parameters(estimators, Y, Qn, Qδn, Hn, Hshiftn)

    return collect_bootstrapped_estimates(outcome_regression), collect_bootstrapped_estimates(ipw), collect_bootstrapped_estimates(onestep), collect_bootstrapped_estimates(tmle)
end

#lazy_iterate_predict(mach, X::Node...) = node(X -> Iterators.map(x -> MMI.predict(mach, x...), zip(X...)), X)
lazy_iterate_predict(mach, X::Node) = node(X -> Iterators.map(x -> MMI.predict(mach, x), X), X)
lazy_iterate_predict(mach, X1::Node, X2::Node) = node((X1, X2) -> Iterators.map(x -> MMI.predict(mach, x...), zip(X1, X2)), X1, X2)
lazy_multiply_derivative(X::Node, dX::Node) = node((X, dX) -> Iterators.map((x, dx) -> x * prod(dx), X, dX), X, dX)


function resample_nuisances(mach_mean, mach_density, LAs, LAδs, LAδsinv, dAδs, dAδsinv)
    # Get Conditional Mean
    Qn = lazy_iterate_predict(mach_mean, LAs)
    Qδn = lazy_iterate_predict(mach_mean, LAδs)

    # Get Density Ratio
    Hn_noderiv = lazy_iterate_predict(mach_density, LAδsinv, LAs)
    Hn = lazy_multiply_derivative(Hn_noderiv, dAδs)

    Hshiftn_noderiv = lazy_iterate_predict(mach_density, LAs, LAδs)
    Hshiftn = lazy_multiply_derivative(Hshiftn_noderiv, dAδs)

    return Qn, Qδn, Hn, Hshiftn
end

function resample_causal_parameters(estimators::Tuple, Y, Qn, Qδn, Hn, Hshiftn)
    
    outcome_regression = node(Qδn -> map(outcome_regression_transform, Qδn), Qδn)
    
    mach_ipw = machine(ResampledModel(estimators[1]), Y) |> fit!
    mach_onestep = machine(ResampledModel(estimators[2]), Y, Qn) |> fit!
    mach_tmle = machine(ResampledModel(estimators[3]), Y, Qn) |> fit!

    ipw = MMI.transform(mach_ipw, Hn)
    onestep = MMI.transform(mach_onestep, Qδn, Hn)
    tmle = MMI.transform(mach_tmle, Qδn, Hn, Hshiftn)

    return outcome_regression, ipw, onestep, tmle
end

collect_bootstrapped_estimates(estimates::AbstractNode) = node(collect_bootstrapped_estimates, estimates)
collect_bootstrapped_estimates(estimates) = (boot = [e.ψ for e in estimates],)

