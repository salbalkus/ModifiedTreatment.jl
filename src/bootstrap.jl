# Interface endpoints
# TODO: Automatically select appropriate bootstrap sampler


"""
    bootstrap(resampler::BootstrapSampler, mtpresult::MTPResult, B::Int64)

The `bootstrap` function performs bootstrap resampling to provide an estimate of each estimator's variance.

# Arguments
- `resampler::BootstrapSampler`: The resampler object used to generate bootstrap samples.
- `mtpresult::MTPResult`: The result object obtained from fitting an MTP model.
- `B::Int64`: The number of bootstrap samples (replications).

# Returns
A named tuple containing the bootstrapped estimates of treatment effects.

"""
function bootstrap(resampler::BootstrapSampler, mtpresult::MTPResult, B::Int64)
    
    # Extract the MTP machine and its nuisance models that user has already fit
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

    # Package outputs
    output = var(bootstrap_samples, dims = 1)
    return NamedTuple{keys(results)}((output[1, i] for i in 1:length(results)))
end

"""
    bootstrap!(resampler::BootstrapSampler, mtpresult::MTPResult, B::Int64)

The `bootstrap` function performs bootstrap resampling to provide an estimate of each estimator's variance. 
This is a convience function to append the bootstrapped variance directly to a previously-computed MTP output.

# Arguments
- `resampler::BootstrapSampler`: The resampler object used to generate bootstrap samples.
- `mtpresult::MTPResult`: The result object obtained from fitting an MTP model.
- `B::Int64`: The number of bootstrap samples (replications).

# Returns
A named tuple containing the bootstrapped estimates of treatment effects.

"""
function bootstrap!(resampler::BootstrapSampler, mtpresult::MTPResult, B::Int64)
    bootstrapped_variances = bootstrap(resampler, mtpresult, B)
    # assign bootstrapped variances in-place
    for (key, val) in pairs(getestimate(mtpresult))
        val.σ2boot = bootstrapped_variances[key]
    end
end

# Helper function to compute bootstrapped estimates
function bootstrap_sample(mtpresult::MTPResult, sampler::BootstrapSampler, mach_Qn, mach_Hn, O::CausalTable, types::Vector{DataType})
    
    # Extract the varianbles needed to bootstrap
    O_sample = bootstrap(sampler, O)
    Y = getresponse(O_sample)
    G = get_dependency_neighborhood(getgraph(O_sample))

    # Apply intervention and summarize the bootstrapped sample
    LAs, LAδ, dAδ, LAδinv, dAδinv = get_summarized_intervened_data(O_sample, mtpresult, types)

    # Get Conditional Mean predicts
    if needs_conditional_mean(types)
        Qn = MMI.predict(mach_Qn, LAs)
        Qδn = MMI.predict(mach_Qn, LAδ)
    else
        Qn, Qδn = nothing
    end
    
    # Get Density Ratio predictions
    if needs_propensity(types)
        Hn = MMI.predict(mach_Hn, LAδinv, LAs) * prod(dAδinv)
    else
        Hn = nothing
    end
    
    if needs_shifted_propensity(types)
        Hshiftn = MMI.predict(mach_Hn, LAs, LAδ)
    else
        Hshiftn = nothing
    end
    
    # Compute the causal estimates based on nuisance parameters for each estimate that's been previously output
    nuisance_tuple = (Y, G, Qn, Qδn, Hn, Hshiftn)
    return [estimate(val, nuisance_tuple).ψ for val in values(getestimate(mtpresult))]
end

# Dispatch the `estimate` function on the type of estimator we want to use
# Relies on functions rom `causalestimators.jl`
estimate(::PlugInResult, nuisances) = plugin_transform(nuisances[4])
estimate(r::IPWResult, nuisances) = r.stabilized ? sipw(nuisances[1], nuisances[5], nuisances[2]) : ipw(nuisances[1], nuisances[5], nuisances[2])
estimate(::OneStepResult, nuisances) = onestep(nuisances[1], nuisances[3], nuisances[4], nuisances[5], nuisances[2])
estimate(::TMLEResult, nuisances) = tmle(nuisances[1], nuisances[3], nuisances[4], nuisances[5], nuisances[6], nuisances[2])

# Functions to determine which types of estimator needs which nuisance parameter
needs_conditional_mean(types) = PlugInResult ∈ types || OneStepResult ∈ types || TMLEResult ∈ types
needs_propensity(types) = IPWResult ∈ types || OneStepResult ∈ types || TMLEResult ∈ types
needs_shifted_propensity(types) = TMLEResult ∈ types

# Helper function to get the intervened data from a bootstrapped sample
function get_summarized_intervened_data(O::CausalTable, mtpresult::MTPResult, types::Vector{DataType})

    # Extract the intervention applied earlier
    δ = getintervention(mtpresult)

    # Compute summary functions and construct the necessary data structures
    LAs, A, L, summaries, treatmentvar, summarizedvars = get_summarized_data(O)

    # Based on the types of interventions, compute the necessary nuisance estimates
    LAδ, dAδ, LAδinv, dAδinv = (nothing for _ = 1:4)
    if needs_conditional_mean(types)
        LAδ, dAδ = get_intervened_data(A, L, δ, summaries, treatmentvar, summarizedvars)
    end
    if needs_propensity(types)
        LAδinv, dAδinv = get_intervened_data(A, L, inverse(δ), summaries, treatmentvar, summarizedvars)
    end
    return LAs, LAδ, dAδ, LAδinv, dAδinv
end



