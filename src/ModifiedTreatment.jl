module ModifiedTreatment

    using MLJModelInterface
    using GLM
    using StatsBase
    using Graphs
    using Tables
    using TableOperations
    using CausalTables


    import Base: merge

    const MMI = MLJModelInterface

    include("intervention.jl")

    export Intervention
    export LinearShift, AdditiveShift, MultiplicativeShift
    export apply_intervention, apply_inverse_intervention 
    export differentiate_intervention, differentiate_inverse_intervention
    export get_induced_intervention

    export fit, transform, inverse_transform, predict
    export replace_treatment

end
