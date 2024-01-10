module ModifiedTreatment

    using MLJModelInterface
    using MLJBase
    using GLM
    using StatsBase
    using Graphs
    using Tables
    using TableOperations
    using CausalTables


    import Base: merge

    const MMI = MLJModelInterface

    include("nodeutils.jl")
    include("utils.jl")

    include("interventions/intervention.jl")
    include("interventions/linearshift.jl")
    include("interventionmodel.jl")

    include("causalestimators.jl")
    include("mtp.jl")

    # interventions
    export Intervention
    export LinearShift, AdditiveShift, MultiplicativeShift
    export apply_intervention, apply_inverse_intervention 
    export differentiate_intervention, differentiate_inverse_intervention
    export get_induced_intervention

    # intervention model
    export InterventionModel

    # mtp
    export MTP, estimate_causal

    # general
    export prefit, fit, transform, inverse_transform, predict
    export replace_treatment



end
