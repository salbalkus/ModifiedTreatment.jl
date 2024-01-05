module ModifiedTreatment

using CausalTables
using MLJModelInterface
using GLM
using StatsBase

import Base: merge

const MMI = MLJModelInterface

include("intervention.jl")

export Intervention, ShiftIntervention
export fit, transform, inverse_transform, predict

end
