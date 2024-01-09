using Test
using ModifiedTreatment
using CausalTables
using Distributions
using MLJBase
using ForwardDiff
using DiffResults

#using Condensity 

distseq = [
        :L1 => (; O...) -> DiscreteUniform(1, 5),
        :A => (; O...) -> (@. Normal(O[:L1], 1)),
        :Y => (; O...) -> (@. Normal(O[:A] + 0.2 * O[:L1], 1))
    ]
dgp_iid = DataGeneratingProcess(distseq, :A, :Y, [:L1])

data_iid = rand(dgp_iid, 10)

#@testset "Intervention" begin
    shift = 1.0
    additive(A, δ, O) = A + δ

    foo = (A) -> additive(A, shift, data_iid)
    x = gettreatment(data_iid)
    xd = gettreatment(data_iid)

    #result = ForwardDiff.gradient!(result, foo, x)
    

    ForwardDiff.derivative.(foo, 1.0)
    
    inv_additive(A, δ, O) = A - δ

    intervention = Intervention(additive, inv_additive)
    mach = machine(intervention, data_iid) |> fit!

    @test predict(mach, shift).tbl == data_iid.tbl
    @test all(gettreatment(transform(mach, shift)) .== gettreatment(data_iid) .+ shift)
    @test all(gettreatment(inverse_transform(mach, shift)) .== gettreatment(data_iid) .- shift)


#end
