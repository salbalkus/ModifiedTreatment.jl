# Interface endpoints
# TODO: Automatically select appropriate bootstrap sampler
function bootstrap(resampler::BootstrapSampler, mtpresult::MTPResult, B::Int64)
    mtpmach = getmtp(mtpresult)
    mach_Qn, mach_Hn = nuisance_machines(mtpmach)
    O = mtpmach.args[1]()

    # Get the types of interventions that we've applied and want to bootstrap
    results = getestimate(mtpresult)
    types = typeof.(collect(values(results)))

    # Collect bootstrapped estimates
    bootstrap_samples = Array{Float64}(undef, B, length(results))
    for b in 1:B
        bootstrap_samples[b, :] = bootstrap_sample(mtpresult, resampler, mach_Qn, mach_Hn, O, types)
    end

    output = var(bootstrap_samples, dims = 1)
    return NamedTuple{keys(results)}((output[1, i] for i in 1:length(results)))
end

function bootstrap!(resampler::BootstrapSampler, mtpresult::MTPResult, B::Int64)
    bootstrapped_variances = bootstrap(resampler, mtpresult, B)
    # assign bootstrapped variances in-place
    for (key, val) in pairs(getestimate(mtpresult))
        val.σ2boot = bootstrapped_variances[key]
    end
end

function bootstrap_sample(mtpresult::MTPResult, sampler::BootstrapSampler, mach_Qn, mach_Hn, O::CausalTable, types::Vector{DataType})
    O_sample = bootstrap(sampler, O)
    Y = getresponse(O_sample)
    G = get_dependency_neighborhood(getgraph(O_sample))
    LAs, LAδ, dAδ, LAδinv, dAδinv = get_summarized_intervened_data(O_sample, mtpresult, types)

    # Get Conditional Mean
    if needs_conditional_mean(types)
        Qn = MMI.predict(mach_Qn, LAs)
        Qδn = MMI.predict(mach_Qn, LAδ)
    else
        Qn, Qδn = nothing
    end
    
    # Get Density Ratios
    if needs_propensity(types)
        Hn = MMI.predict(mach_Hn, LAδinv, LAs) * prod(dAδinv)
    else
        Hn = nothing
    end
    
    # TODO: Check if we actually don't need to multiply by a derivative here
    if needs_shifted_propensity(types)
        Hshiftn = MMI.predict(mach_Hn, LAs, LAδ)
    else
        Hshiftn = nothing
    end
    
    # Compute the causal estimates based on nuisance parameters
    nuisance_tuple = (Y, G, Qn, Qδn, Hn, Hshiftn)
    return [estimate(val, nuisance_tuple).ψ for val in values(getestimate(mtpresult))]
end

# Dispatch on the type of estimator we want to use
estimate(::PlugInResult, nuisances) = plugin_transform(nuisances[4])
estimate(r::IPWResult, nuisances) = r.stabilized ? sipw(nuisances[1], nuisances[5], nuisances[2]) : ipw(nuisances[1], nuisances[5], nuisances[2])
estimate(::OneStepResult, nuisances) = onestep(nuisances[1], nuisances[3], nuisances[4], nuisances[5], nuisances[2])
estimate(::TMLEResult, nuisances) = tmle(nuisances[1], nuisances[3], nuisances[4], nuisances[5], nuisances[6], nuisances[2])

# Functions to determine which types of estimator needs which nuisance parameter
needs_conditional_mean(types) = PlugInResult ∈ types || OneStepResult ∈ types || TMLEResult ∈ types
needs_propensity(types) = IPWResult ∈ types || OneStepResult ∈ types || TMLEResult ∈ types
needs_shifted_propensity(types) = TMLEResult ∈ types

function get_summarized_intervened_data(O::CausalTable, mtpresult::MTPResult, types::Vector{DataType})
    δ = getintervention(mtpresult)
    LAs, A, L, summaries, treatmentvar, summarizedvars = get_summarized_data(O)

    # Based on the types of interventions, compute the nuisances that we need
    LAδ, dAδ, LAδinv, dAδinv = (nothing for _ = 1:4)
    if needs_conditional_mean(types)
        LAδ, dAδ = get_intervened_data(A, L, δ, summaries, treatmentvar, summarizedvars)
    end
    if needs_propensity(types)
        LAδinv, dAδinv = get_intervened_data(A, L, inverse(δ), summaries, treatmentvar, summarizedvars)
    end
    return LAs, LAδ, dAδ, LAδinv, dAδinv
end



