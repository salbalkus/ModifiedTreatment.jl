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

    import LinearAlgebra: diagm, I
    import Combinatorics: combinations
    using SparseArrays

    import Base: merge, getindex, iterate, prod, zip, collect, reduce

    const MMI = MLJModelInterface
    const MT = MLJTuning

    include("nodeutils.jl")
    include("utils.jl")

    include("interventions/intervention.jl")
    include("interventions/linearshift.jl")
    include("interventions/interventionmodel.jl")

    include("crossfitting.jl")
    include("resampling.jl")

    include("causalestimators.jl")
    include("mtp.jl")
    include("bootstrap.jl")
    include("simulation/truth.jl")

    # interventions
    export Intervention, IdentityIntervention
    export LinearShift, AdditiveShift, MultiplicativeShift
    export apply_intervention, inverse, differentiate_intervention, get_induced_intervention

    # intervention model
    export InterventionModel

    # crossfitting
    export CrossFitModel

    # resampling
    export ResampledModel, BootstrapSampler, BasicSampler, ClusterSampler, VertexMooNSampler, VertexSampler
    export bootstrap

    # causalestimators
    export estimate_outcome_regression, OutcomeRegressor, IPW, OneStep, TMLE

    # simulation
    export compute_true_MTP

    # mtp
    export MTP, intervene_on_data, crossfit_nuisance_estimators, estimate_nuisances, estimate_causal_parameters, bootstrap_estimates
    export resample_nuisances, resample_causal_parameters, lazy_iterate_predict, collect_bootstrapped_estimates
    export outcome_regression, ipw, onestep, tmle, nuisance_machines

    # nodeutils
    export gettreatment, getresponse, getgraph, getsummaries, summarize
    export merge, getindex, iterate, length

    # general
    export prefit, fit, transform, inverse_transform, predict
    export replace_treatment



end
