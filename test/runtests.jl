using Test
using ModifiedTreatment
using CausalTables
using Distributions
using Condensity 


dgp_iid = DataGeneratingProcess([
        :L1 => (; O...) -> DiscreteUniform(1, 5),
        :A => (; O...) -> (@. Normal(O[:L1], 1)),
        :Y => (; O...) -> (@. Normal(O[:A] + 0.2 * O[:L1], 1))
    ],
    :A, :Y, [:L]
    )

data_iid = rand(dgp_iid, 10)

#@testset "ModifiedTreatment.jl" begin
    
shift = ShiftIntervention((a, δ) -> a + δ)
    

#end
