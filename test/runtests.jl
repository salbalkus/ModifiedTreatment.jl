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
    int1 = AdditiveShift(0.5)
    int2 = MultiplicativeShift(1.5)
    int3 = LinearShift(1.5, 0.5)

    L = getcontrols(data_iid)
    A = gettreatment(data_iid)

    apply_intervention(int3, A, L)

    A .* 1.5 .+ 0.5

    @test all(apply_intervention(int1, A, L) .== A .+ 0.5)
    @test all(apply_inverse_intervention(int1, A, L) .== A .- 0.5)
    @test all(differentiate_intervention(int1, A, L) .== 1)
    @test all(differentiate_inverse_intervention(int1, A, L) .== 1)

    @test all(apply_intervention(int2, A, L) .== A .* 1.5)
    @test all(apply_inverse_intervention(int2, A, L) .== A ./ 1.5)
    @test all(differentiate_intervention(int2, A, L) .== 1.5)
    @test all(differentiate_inverse_intervention(int2, A, L) .== 1/1.5)

    @test all(apply_intervention(int3, A, L) .== A .* 1.5 .+ 0.5)
    @test all(apply_inverse_intervention(int3, A, L) .== (A .- 0.5) ./ 1.5)
    @test all(differentiate_intervention(int3, A, L) .== 1.5)
    @test all(differentiate_inverse_intervention(int3, A, L) .== 1/1.5)

    neighborsum = NeighborSum(:A)
    inducedint = get_induced_intervention(int3, neighborsum)
    apply_intervention(inducedint, A, L)

    @test all(apply_intervention(inducedint, A, L) .== A .* 1.5 .+ 0.5)
    @test all(apply_inverse_intervention(int3, A, L) .== (A .- 0.5) ./ 1.5)
    @test all(differentiate_intervention(int3, A, L) .== 1.5)
    @test all(differentiate_inverse_intervention(int3, A, L) .== 1/1.5)


#end
