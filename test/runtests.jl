using Test
using ModifiedTreatment
using CausalTables
using Distributions
using MLJBase
#using Condensity 

distseq = [
        :L1 => (; O...) -> DiscreteUniform(1, 5),
        :A => (; O...) -> (@. Normal(O[:L1], 1)),
        :Y => (; O...) -> (@. Normal(O[:A] + 0.2 * O[:L1], 1))
    ]
dgp_iid = DataGeneratingProcess(distseq, :A, :Y, [:L1])


data_iid = rand(dgp_iid, 10)

#@testset "Intervention" begin
    
additive(A, δ, O) = A .+ δ
inv_additive(A, δ, O) = A .- δ

intervention = Intervention(additive, inv_additive)
mach = machine(intervention, data_iid) |> fit!

predict(mach, 1.0)

    

#end
