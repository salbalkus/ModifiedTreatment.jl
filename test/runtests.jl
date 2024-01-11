using Test
using ModifiedTreatment
using CausalTables
using Distributions
using MLJBase
using MLJLinearModels
using MLJModels
using Graphs
using Condensity

# function for testing approximate equality of statistical estimators
within(x, truth, ϵ) = abs(x - truth) < ϵ

distseq = Vector{Pair{Symbol, CausalTables.ValidDGPTypes}}([
        :L1 => (; O...) -> DiscreteUniform(1, 5),
        :L1_s => NeighborSum(:L1),
        :A => (; O...) -> (@. Normal(O[:L1] + 0.1 * O[:L1_s], 0.5)),
        :A_s => NeighborSum(:A),
        :Y => (; O...) -> (@. Normal(O[:A] + 0.1 * O[:A_s] + 0.2 * O[:L1] + 10, 1))
    ])

dgp = DataGeneratingProcess(n -> erdos_renyi(n, 3/n), distseq; treatment = :A, response = :Y, controls = [:L1]);
data = rand(dgp, 100)

@testset "Intervention" begin
    int1 = AdditiveShift(0.5)
    int2 = MultiplicativeShift(1.5)
    int3 = LinearShift(1.5, 0.5)

    L = CausalTables.getcontrols(data)
    A = CausalTables.gettreatment(data)

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

    inducedint = get_induced_intervention(int3, NeighborSum(:A))

    @test apply_intervention(inducedint, A, L) == A .* 1.5 .+ adjacency_matrix(data.graph) * (ones(nv(data.graph)) .* 0.5)
    @test apply_inverse_intervention(inducedint, A, L) == (A .- adjacency_matrix(data.graph) * (ones(nv(data.graph)) .* 0.5)) ./ 1.5
    @test all(differentiate_intervention(inducedint, A, L) .== 1.5)
    @test all(differentiate_inverse_intervention(inducedint, A, L) .== 1/1.5)

end

@testset "InterventionModel" begin
    intervention = LinearShift(1.5, 0.5)
    intmach = machine(InterventionModel(), data) |> fit!
    LAs, Ls, As = predict(intmach, intervention)  

    @test Ls.tbl == (L1 = data.tbl.L1, L1_s = data.tbl.L1_s)
    @test As == (A_s = data.tbl.A_s, A = data.tbl.A)
    @test LAs.tbl == (L1 = data.tbl.L1, L1_s = data.tbl.L1_s, A = data.tbl.A, A_s = data.tbl.A_s)

    LAδs, dAδs = transform(intmach, intervention)
    @test LAδs.tbl.A ≈ data.tbl.A .* 1.5 .+ 0.5
    @test LAδs.tbl.A_s ≈ adjacency_matrix(data.graph) * ((data.tbl.A .* 1.5) .+ 0.5)
    @test dAδs.A == 1.5

    LAδsinv, dAδsinv = inverse_transform(intmach, intervention)
    @test LAδsinv.tbl.A ≈ (data.tbl.A .- 0.5) ./ 1.5
    @test LAδsinv.tbl.A_s ≈ adjacency_matrix(data.graph) * ((data.tbl.A .- 0.5) ./ 1.5)
    @test dAδsinv.A == 1/1.5
end

distseq = Vector{Pair{Symbol, CausalTables.ValidDGPTypes}}([
        :L1 => (; O...) -> DiscreteUniform(1, 5),
        :A => (; O...) -> (@. Normal(O[:L1], 1)),
        :Y => (; O...) -> (@. Normal(O[:A] + 0.5 * O[:L1] + 10, 0.5))
    ])

dgp = DataGeneratingProcess(distseq, :A, :Y, [:L1]);
data_large = rand(dgp, 10^6)

intervention = LinearShift(1.1, 0.5)
truth = compute_true_MTP(dgp, data_large, intervention)

@testset "MTP IID" begin

    moe = 0.1

    mean_estimator = LinearRegressor()
    density_ratio_estimator = DensityRatioPropensity(OracleDensityEstimator(dgp))
    boot_sampler = nothing
    cv_splitter = nothing

    mtp = MTP(mean_estimator, density_ratio_estimator, boot_sampler, cv_splitter)
    mtpmach = machine(mtp, data_large) |> fit!
        
    output_or = outcome_regression(mtpmach, intervention)
    @test within(output_or.ψ, truth.ψ, moe)
    
    output_ipw = ipw(mtpmach, intervention)
    @test within(output_ipw.ψ, truth.ψ, moe)
    
    output_onestep = onestep(mtpmach, intervention)
    @test within(output_ipw.ψ, truth.ψ, moe)

    output_tmle = tmle(mtpmach, intervention)
    @test within(output_tmle.ψ, truth.ψ, moe)

end