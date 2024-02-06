module ModifiedTreatment

    using MLJModelInterface
    using MLJTuning
    using MLJBase
    using GLM
    using StatsBase
    using Graphs
    using Tables
    using TableOperations
    using DataAPI
    using CausalTables

    using LinearAlgebra
    import Combinatorics: combinations
    using SparseArrays

    import Base: merge, getindex, iterate, prod, zip, collect, reduce

    const MMI = MLJModelInterface
    const MT = MLJTuning

    include("nodeutils.jl")
    export gettreatment, getresponse, getgraph, getsummaries, summarize
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
    export CrossFitModel

    include("resampling.jl")
    export ResampledModel, BootstrapSampler, BasicSampler, ClusterSampler, VertexMooNSampler, VertexSampler

    include("causalestimators.jl")
    export estimate_outcome_regression, OutcomeRegressor, IPW, OneStep, TMLE
    export getestimate, getmtp, getintervention

    include("truth.jl")
    export compute_true_MTP

    include("mtp.jl")
    export MTP, intervene_on_data, crossfit_nuisance_estimators, estimate_nuisances, estimate_causal_parameters, bootstrap_estimates
    export resample_nuisances, resample_causal_parameters, lazy_iterate_predict, collect_bootstrapped_estimates
    export outcome_regression, ipw, onestep, tmle, nuisance_machines, estimate
    export ψ, σ2, σ2boot

    include("bootstrap.jl")
    export bootstrap, bootstrap!

    # general
    export prefit, fit, transform, inverse_transform, predict
    export replace_treatment

end
