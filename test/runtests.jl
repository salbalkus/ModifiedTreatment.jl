using Test
using ModifiedTreatment
using CausalTables
using Distributions
using MLJBase
using Graphs

#using Condensity 
neighborsum = NeighborSum(:A)
distseq = Vector{Pair{Symbol, CausalTables.ValidDGPTypes}}([
        :L1 => (; O...) -> DiscreteUniform(1, 5),
        :A => (; O...) -> (@. Normal(O[:L1], 1)),
        :A_s => neighborsum,
        :Y => (; O...) -> (@. Normal(O[:A] + O[:A_s] + 0.2 * O[:L1], 1))
    ])

dgp = DataGeneratingProcess(n -> erdos_renyi(n, 0.4), distseq; treatment = :A, response = :Y, controls = [:L1]);
data = rand(dgp, 10)

@testset "Intervention" begin
    int1 = AdditiveShift(0.5)
    int2 = MultiplicativeShift(1.5)
    int3 = LinearShift(1.5, 0.5)

    L = getcontrols(data)
    A = gettreatment(data)

    @test apply_intervention(int1, A, L) == A .+ 0.5
    @test apply_inverse_intervention(int1, A, L) == A .- 0.5
    @test differentiate_intervention(int1, A, L) == 1
    @test differentiate_inverse_intervention(int1, A, L) == 1

    @test apply_intervention(int2, A, L) == A .* 1.5
    @test apply_inverse_intervention(int2, A, L) == A ./ 1.5
    @test differentiate_intervention(int2, A, L) == 1.5
    @test differentiate_inverse_intervention(int2, A, L) == 1/1.5

    @test apply_intervention(int3, A, L) == A .* 1.5 .+ 0.5
    @test apply_inverse_intervention(int3, A, L) == (A .- 0.5) ./ 1.5
    @test differentiate_intervention(int3, A, L) == 1.5
    @test differentiate_inverse_intervention(int3, A, L) == 1/1.5

    inducedint = get_induced_intervention(int3, neighborsum)

    @test apply_intervention(inducedint, A, L) == A .* 1.5 .+ adjacency_matrix(data.graph) * (ones(nv(data.graph)) .* 0.5)
    @test apply_inverse_intervention(inducedint, A, L) == (A .- adjacency_matrix(data.graph) * (ones(nv(data.graph)) .* 0.5)) ./ 1.5
    @test all(differentiate_intervention(inducedint, A, L) .== 1.5)
    @test all(differentiate_inverse_intervention(inducedint, A, L) .== 1/1.5)

end

@testset "InterventionModel" begin
    intervention = LinearShift(1.5, 0.5)
    intmach = machine(InterventionModel(), data) |> fit!
    LAs, Ls, As = predict(intmach, intervention)  
    @test Ls.tbl == (L1 = data.tbl.L1,)
    @test As == (A = data.tbl.A, A_s = data.tbl.A_s)
    @test LAs.tbl == (L1 = data.tbl.L1, A = data.tbl.A, A_s = data.tbl.A_s)
    
    LAδs, Aδsd = transform(intmach, intervention)
    @test LAδs.tbl.A ≈ data.tbl.A .* 1.5 .+ 0.5
    @test LAδs.tbl.A_s ≈ adjacency_matrix(data.graph) * ((data.tbl.A .* 1.5) .+ 0.5)
    @test Aδsd.A == 1.5
    @test Aδsd.A == 1.5

    LAδsinv, Aδsdinv = inverse_transform(intmach, intervention)
    @test LAδsinv.tbl.A ≈ (data.tbl.A .- 0.5) ./ 1.5
    @test LAδsinv.tbl.A_s ≈ adjacency_matrix(data.graph) * ((data.tbl.A .- 0.5) ./ 1.5)
    @test Aδsdinv.A == 1/1.5
    @test Aδsdinv.A == 1/1.5
end
