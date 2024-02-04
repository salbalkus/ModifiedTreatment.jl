

bootstrap(results, sampler::BootstrapSampler, mach_Qn, mach_Hn, O::CausalTable, δ::Intervention, B::Int64) = [bootstrap_sample(results, sampler, mach_Qn, mach_Hn, O, δ) for b in 1:B]

# TODO: These should only work with results of type CausalEstimator
function bootstrap(mtpmach::Machine{MTP}, δ::Intervention, B::Int64, results)
    mach_Qn, mach_Hn = nuisance_machines(mtpmach)
    O = mtpmach.args[1]()
    return bootstrap(results, mtpmach.model.boot_sampler, mach_Qn, mach_Hn, O, δ, B)
end

function bootstrap_sample(results, sampler::BootstrapSampler, mach_Qn, mach_Hn, O::CausalTable, δ::Intervention)
    O_sample = bootstrap(sampler, O)
    Y = getresponse(O_sample)
    LAs, LAδ, dAδ, LAδinv, dAδinv = get_summarized_intervened_data(O_sample, δ)
    Qn, Qδn, Hn, Hshiftn = estimate_nuisances(mach_Qn, mach_Hn, LAs, LAδ, LAδinv, dAδ, dAδinv)
    nuisance_tuple = (Y, Qn, Qδn, Hn, Hshiftn)
    return [estimate(val, nuisance_tuple).ψ for (key, val) in results]
end

function get_summarized_intervened_data(O::CausalTable, δ::Intervention)
    LAs, A, L, summaries, treatmentvar = get_summarized_data(O)
    LAδ, dAδ = get_intervened_data(LAs, A, L, δ, summaries, treatmentvar)
    LAδinv, dAδinv = get_intervened_data(LAs, A, L, inverse(δ), summaries, treatmentvar)
    return LAs, LAδ, dAδ, LAδinv, dAδinv
end

estimate(::OutcomeRegressionResult, nuisances) = outcome_regression_transform(nuisances[3])
estimate(::IPWResult, nuisances) = ipw(nuisances[1], nuisances[4])
estimate(::OneStepResult, nuisances) = onestep(nuisances[1], nuisances[2], nuisances[3], nuisances[4])
estimate(::TMLEResult, nuisances) = tmle(nuisances[1], nuisances[2], nuisances[3], nuisances[4], nuisances[5])

function bootstrap!(mtpmach::Machine{MTP}, δ::Intervention, B::Int64; results...)
    output = var.(zip(bootstrap(mtpmach, δ, B, results)...))
    for (i, result) in enumerate(results)
        result[2].σ2boot = output[i]
    end
end

