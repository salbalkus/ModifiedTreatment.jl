
# Model to handle bootstrapped lazy outputs
mutable struct ResampledModel{T <: MMI.Model} <: MMI.Model
    model::T
end

function MMI.fit(rse::ResampledModel, verbosity, X...)

    # Here X should be a tuple of equal-length vectors
    # Each vector entry is one "sample" of the data for that particular parameter

    # Fit each sample
    machines = Iterators.map(x -> fit!(machine(rse.model, x...)), zip(X...))

    fitresult = (; machines = machines)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

# Predict on Generators
MMI.transform(rse::ResampledModel, fitresult, X::Base.Generator...) = Iterators.map(x -> MMI.transform(x...), zip(fitresult.machines, X...))

# Special case of InterventionModel
function MMI.predict(rse::ResampledModel{InterventionModel}, fitresult, δ::Intervention)
    result = Iterators.map(mach -> MMI.predict(mach, δ), fitresult.machines)
    return Iterators.map(r -> getindex(r, 1), result), Iterators.map(r -> getindex(r, 2), result), Iterators.map(r -> getindex(r, 3), result)
end
function MMI.transform(rse::ResampledModel{InterventionModel}, fitresult, δ::Intervention)
    result = Iterators.map(mach -> MMI.transform(mach, δ), fitresult.machines)
    return Iterators.map(r -> getindex(r, 1), result), Iterators.map(r -> getindex(r, 2), result)
end
function MMI.inverse_transform(rse::ResampledModel{InterventionModel}, fitresult, δ::Intervention)
    result = Iterators.map(mach -> MMI.inverse_transform(mach, δ), fitresult.machines)
    return Iterators.map(r -> getindex(r, 1), result), Iterators.map(r -> getindex(r, 2), result)
end


# Bootstrapping
abstract type BootstrapSampler end

bootstrap_samples(sampler::BootstrapSampler, O::AbstractNode) = node(O -> bootstrap_samples(sampler, O), O)
bootstrap_samples(sampler::BootstrapSampler, O::CausalTable) = Iterators.map(i -> bootstrap_sample(sampler, O), 1:sampler.B)

mutable struct BasicSampler <: BootstrapSampler
    B::Int64
end
bootstrap_sample(sampler::BasicSampler, O::CausalTable) = Tables.subset(O, rand(1:DataAPI.nrow(O), DataAPI.nrow(O)))
    