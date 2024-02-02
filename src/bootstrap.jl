bootstrap(sampler::BootstrapSampler, mach_Qn, mach_Hn, O::CausalTable, δ::Intervention, B::Int64) = [bootstrap_sample(sampler, mach_Qn, mach_Hn, O, δ) for b in 1:B]

function bootstrap(mtpmach::Machine{MTP}, δ::Intervention, B::Int64)
    mach_Qn, mach_Hn = nuisance_machines(mtpmach)
    O = mtpmach.args[1]()
    return bootstrap(mtpmach.model.boot_sampler, mach_Qn, mach_Hn, O, δ, B)
end

function bootstrap_sample(sampler::BootstrapSampler, mach_Qn, mach_Hn, O::CausalTable, δ::Intervention)
    O_sample = bootstrap(sampler, O)
    Y = getresponse(O_sample)
    LAs, LAδ, dAδ, LAδinv, dAδinv = get_summarized_intervened_data(O_sample, δ)
    Qn, Qδn, Hn, Hshiftn = estimate_nuisances(mach_Qn, mach_Hn, LAs, LAδ, LAδinv, dAδ, dAδinv)

    outcome_regression_est = outcome_regression_transform(Qδn)
    ipw_est = ipw(Y, Hn)
    onestep_est = onestep(Y, Qn, Qδn, Hn)
    tmle_est = tmle(Y, Qn, Qδn, Hn, Hshiftn)
    return (; outcome_regression = outcome_regression_est, ipw = ipw_est, onestep = onestep_est, tmle = tmle_est)
end

function get_summarized_intervened_data(O::CausalTable, δ::Intervention)
    LAs, A, L, summaries, treatmentvar = get_summarized_data(O)
    LAδ, dAδ = get_intervened_data(LAs, A, L, δ, summaries, treatmentvar)
    LAδinv, dAδinv = get_intervened_data(LAs, A, L, inverse(δ), summaries, treatmentvar)
    return LAs, LAδ, dAδ, LAδinv, dAδinv
end

