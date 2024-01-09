module ModifiedTreatment

    using MLJModelInterface
    using GLM
    using StatsBase
    using Tables
    using TableOperations
    using CausalTables


    import Base: merge

    const MMI = MLJModelInterface

    include("intervention.jl")

    export Intervention
    export fit, transform, inverse_transform, predict
    export replace_treatment

end
