module ModifiedTreatment

    using MLJModelInterface
    using MLJTuning
    using MLJBase
    import MLJModels: Standardizer
    using StatisticalMeasures
    using GLM
    using StatsBase
    using Graphs
    using Tables
    using TableTransforms
    using DataAPI
    using CausalTables
    using Condensity

    using LinearAlgebra
    import Combinatorics: combinations
    using SparseArrays

    import Base: merge, getindex, iterate, prod, zip, collect, reduce

    const MMI = MLJModelInterface
    const MT = MLJTuning

    # Define custom types
    Estimate = Union{Float64, Nothing}
    AbstractGraphOrNothing = Union{AbstractGraph, Nothing}

    include("nodeutils.jl")
    export treatment, response, confounders, summarize
    export merge, getindex, iterate, length

    include("utils.jl")

    include("interventions/intervention.jl")
    include("interventions/linearshift.jl")
    include("interventions/interventionmodel.jl")
    export Intervention, IdentityIntervention
    export LinearShift, AdditiveShift, MultiplicativeShift
    export apply_intervention, inverse, differentiate_intervention, get_induced_intervention
    export InterventionModel

    include("crossfitting.jl")
    export CrossFitModel, DecomposedPropensityRatio, SuperLearner

    include("resampling.jl")
    export ResampledModel, BootstrapSampler, BasicSampler, ClusterSampler, VertexMooNSampler, VertexSampler

    include("causalestimators.jl")
    export estimate_plugin, OutcomeRegressor, IPW, OneStep, TMLE
    export getestimate, getmtp, getintervention

    include("truth.jl")
    export compute_true_MTP

    include("mtp.jl")
    export MTP, intervene_on_data, crossfit_nuisance_estimators, estimate_nuisances, estimate_causal_parameters, bootstrap_estimates
    export resample_nuisances, resample_causal_parameters, lazy_iterate_predict, collect_bootstrapped_estimates
    export plugin, ipw, onestep, tmle, nuisance_machines, estimate
    export ψ, σ2, σ2boot, σ2net

    include("bootstrap.jl")
    export bootstrap, bootstrap!

    # general
    export prefit, fit, transform, inverse_transform, predict

end
