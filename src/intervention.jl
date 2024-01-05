abstract type Intervention <: MMI.Static end

mutable struct ShiftIntervention <: Intervention 
    func::Function
    ShiftIntervention(func) = applicable(func, Vector, Float64) ? new(func) : error("ShiftIntervention: func must be a function of type (Vector{Float64}, Float64) -> Vector{Float64}")
end

MLJBase.transform(model::ShiftIntervention, _, ct, δ) = A .+ δ
MLJBase.inverse_transform(model::ShiftIntervention, _, ct, δ) = A .- δ