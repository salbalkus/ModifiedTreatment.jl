mutable struct MTP <: UnsupervisedNetworkComposite
    mean_estimator
    density_ratio_estimator
    cv_splitter
    confidence::Float64
end

MTP(mean_estimator, density_ratio_estimator, cv_splitter) = MTP(mean_estimator, density_ratio_estimator, cv_splitter, 0.95)

function MLJBase.prefit(mtp::MTP, verbosity, O::CausalTable, Δ::Intervention)
    δ = source(Δ)
    Os = source(O)

    Y = getresponse(Os)
    model_intervention = InterventionModel()
    LAs, Ls, As, LAδs, dAδs, LAδsinv, dAδsinv = intervene_on_data(model_intervention, Os, δ)
    mach_mean, mach_density = crossfit_nuisance_estimators(mtp, Y, LAs, Ls, As)

    Qn, Qδn, Hn, Hshiftn = estimate_nuisances(mach_mean, mach_density, LAs, LAδs, LAδsinv, dAδs, dAδsinv)
    outcome_regression_est, ipw_est, onestep_est, tmle_est = estimate_causal_parameters(Y, Qn, Qδn, Hn, Hshiftn)

    return (; 
        outcome_regression = outcome_regression_est,
        ipw = ipw_est,
        onestep = onestep_est,
        tmle = tmle_est,
        report = (; 
            LAs = LAs,
            LAδs = LAδs,
            LAδsinv = LAδsinv,
            dAδs = dAδs,
            dAδsinv = dAδsinv,
            Qn = Qn,
            Qδn = Qδn,
            Hn = Hn,
            Hshiftn = Hshiftn,
        ),
        nuisance_machines = (;
            machine_mean = mach_mean,
            machine_density = mach_density,
        )
    )    

end

# Define custom functions akin to `predict` that yield the result of each estimation strategy from a learning network machine
outcome_regression(machine, δnew::Intervention) = MTPResult(MLJBase.unwrap(machine.fitresult).outcome_regression(δnew), machine, δnew)
ipw(machine, δnew::Intervention) = MTPResult(MLJBase.unwrap(machine.fitresult).ipw(δnew), machine, δnew)
onestep(machine, δnew::Intervention) = MTPResult(MLJBase.unwrap(machine.fitresult).onestep(δnew), machine, δnew)
tmle(machine, δnew::Intervention) = MTPResult(MLJBase.unwrap(machine.fitresult).tmle(δnew), machine, δnew)

estimate(machine::Machine, δnew::Intervention) = MTPResult(
    (or = MLJBase.unwrap(machine.fitresult).outcome_regression(δnew),
     ipw = MLJBase.unwrap(machine.fitresult).ipw(δnew),
     onestep = MLJBase.unwrap(machine.fitresult).onestep(δnew),
     tmle = MLJBase.unwrap(machine.fitresult).tmle(δnew)
    ),
    machine,
    δnew
)

# Define custom function to extract the nuisance estimators from the learning network machine
nuisance_machines(machine::Machine{MTP}) = MLJBase.unwrap(machine.fitresult).nuisance_machines

function intervene_on_data(model_intervention, Os, δ)
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
    dprmodel = DecomposedPropensityRatio(mtp.density_ratio_estimator)

    if isnothing(mtp.cv_splitter)
        mach_mean = machine(mtp.mean_estimator, LAs, Y)
        mach_density = machine(dprmodel, Ls, As)
    else
        mach_mean = machine(CrossFitModel(mtp.mean_estimator, mtp.cv_splitter), LAs, Y)
        mach_density = machine(CrossFitModel(dprmodel, mtp.cv_splitter), Ls, As)
    end

    return mach_mean, mach_density
end

function estimate_nuisances(mach_mean, mach_density, LAs, LAδs, LAδsinv, dAδs, dAδsinv)
    # Get Conditional Mean
    Qn = MMI.predict(mach_mean, LAs)
    Qδn = MMI.predict(mach_mean, LAδs)

    # Get Density Ratio
    Hn = MMI.predict(mach_density, LAδsinv, LAs) * prod(dAδsinv)

    # TODO: Check if we actually don't need to multiply by a derivative here
    Hshiftn = MMI.predict(mach_density, LAs, LAδs)

    return Qn, Qδn, Hn, Hshiftn
end

function estimate_causal_parameters(Y, Qn, Qδn, Hn, Hshiftn)
    mach_ipw = machine(IPW(), Y) |> fit!
    mach_onestep = machine(OneStep(), Y, Qn) |> fit!
    mach_tmle = machine(TMLE(), Y, Qn) |> fit!

    outcome_regression_est = outcome_regression_transform(Qδn)
    ipw_est = MMI.transform(mach_ipw, Hn)
    onestep_est = MMI.transform(mach_onestep, Qδn, Hn)
    tmle_est = MMI.transform(mach_tmle, Qδn, Hn, Hshiftn)

    return outcome_regression_est, ipw_est, onestep_est, tmle_est
end

