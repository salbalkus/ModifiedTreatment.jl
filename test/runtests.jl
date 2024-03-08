using Test
using ModifiedTreatment
using CausalTables
using Distributions
using MLJBase
using MLJLinearModels
using MLJModels
using Graphs
using Condensity

using Tables
using TableOperations
using DataAPI

using Random
#using Logging

Random.seed!(1)

# function for testing approximate equality of statistical estimators
within(x, truth, ϵ) = abs(x - truth) < ϵ

distseqiid = [
    :L1 => (; O...) -> DiscreteUniform(1, 5),
    :A => (; O...) -> (@. Normal(O[:L1], 1)),
    :Y => (; O...) -> (@. Normal(O[:A] + 0.5 * O[:L1] + 10, 0.5))
]
dgp_iid = DataGeneratingProcess(distseqiid; treatment = :A, response = :Y, controls = [:L1]);
data_iid = rand(dgp_iid, 100)

distseqnet = Vector{Pair{Symbol, CausalTables.ValidDGPTypes}}([
        :L1 => (; O...) -> DiscreteUniform(1, 5),
        :L1_s => Sum(:L1, include_self = false),
        :A => (; O...) -> (@. Normal(O[:L1], 0.5)),
        :A_s => Sum(:A, include_self = false),
        :Y => (; O...) -> (@. Normal(O[:A] + 0.1 * O[:L1_s] + 0.1 * O[:L1] + 10, 1))
    ])

K = 1
dgp_net = DataGeneratingProcess(n -> random_regular_graph(n, K), distseqnet; 
                            treatment = :A, response = :Y, controls = [:L1]);
data_net = rand(dgp_net, 100)

@testset "Intervention" begin
    m = 1.5
    a = 0.5
    int1 = AdditiveShift(a)
    int2 = MultiplicativeShift(m)
    int3 = LinearShift(m, a)

    inv_int1 = inverse(int1)
    inv_int2 = inverse(int2)
    inv_int3 = inverse(int3)

    L = CausalTables.getcontrols(data_net)
    A = CausalTables.gettreatment(data_net)

    @test apply_intervention(int1, A, L) ≈ A .+ 0.5
    @test apply_intervention(inv_int1, A, L) ≈ A .- 0.5
    @test differentiate_intervention(int1, A, L) ≈ 1
    @test differentiate_intervention(inv_int1, A, L) ≈ 1
    
    @test apply_intervention(int2, A, L) ≈ A .* 1.5
    @test apply_intervention(inv_int2, A, L) ≈ A ./ 1.5
    @test differentiate_intervention(int2, A, L) ≈ 1.5
    @test differentiate_intervention(inverse(int2), A, L) ≈ 1/1.5

    @test apply_intervention(int3, A, L) ≈ A .* 1.5 .+ 0.5
    @test apply_intervention(inv_int3, A, L) ≈ (A .- 0.5) ./ 1.5
    @test differentiate_intervention(int3, A, L) ≈ 1.5
    @test differentiate_intervention(inv_int3, A, L) ≈ 1/1.5

    inducedint = get_induced_intervention(int3, Sum(:A, include_self = false))
    inv_inducedint = inverse(inducedint)
    
    A .* m .+ adjacency_matrix(data_net.graph) * (ones(nv(data_net.graph)) .* a)
    apply_intervention(inducedint, A, L)

    @test apply_intervention(inducedint, A, L) ≈ A .* m .+ adjacency_matrix(data_net.graph) * (ones(nv(data_net.graph)) .* a)
    @test apply_intervention(inv_inducedint, A, L) ≈ (A .- adjacency_matrix(data_net.graph) * (ones(nv(data_net.graph)) .* a)) ./ m
    @test all(differentiate_intervention(inducedint, A, L) .≈ 1.5)
    @test all(differentiate_intervention(inv_inducedint, A, L) .≈ 1/1.5)
end



@testset "InterventionModel" begin
    intervention = LinearShift(1.5, 0.5)
    intmach = machine(InterventionModel(), data_net) |> fit!
    LAs, Ls, As = predict(intmach, intervention) 
    @test gettable(Ls) == (L1 = data_net.tbl.L1, L1_s = data_net.tbl.L1_s)
    @test As == (A = data_net.tbl.A, A_s = data_net.tbl.A_s,)
    @test LAs.tbl == (L1 = data_net.tbl.L1, L1_s = data_net.tbl.L1_s, A = data_net.tbl.A, A_s = data_net.tbl.A_s)
    
    LAδs, dAδs = transform(intmach, intervention)
    @test LAδs.tbl.A ≈ (data_net.tbl.A .* 1.5) .+ 0.5
    @test LAδs.tbl.A_s ≈ adjacency_matrix(data_net.graph) * ((data_net.tbl.A .* 1.5) .+ 0.5)
    @test dAδs.A_s == 1.5
    @test dAδs.A == 1.5

    LAδsinv, dAδsinv = inverse_transform(intmach, intervention)
    @test LAδsinv.tbl.A ≈ (data_net.tbl.A .- 0.5) ./ 1.5
    @test LAδsinv.tbl.A_s ≈ adjacency_matrix(data_net.graph) * ((data_net.tbl.A .- 0.5) ./ 1.5)
    @test dAδsinv.A_s == 1/1.5
    @test dAδsinv.A == 1/1.5
end

@testset "DecomposedPropensityRatio on Network" begin
    LA = replacetable(data_net, TableOperations.select(data_net, :L1, :L1_s, :A, :A_s) |> Tables.columntable)    
    ratio_model = DecomposedPropensityRatio(DensityRatioPlugIn(OracleDensityEstimator(dgp_net)))
    L = TableOperations.select(data_net, :L1, :L1_s) |> Tables.columntable
    A = TableOperations.select(data_net, :A, :A_s) |> Tables.columntable
    mach_ratio = machine(ratio_model, L, A) |> fit!
    LAδ = replacetable(LA, (L1 = Tables.getcolumn(L, :L1), L1_s = Tables.getcolumn(L, :L1_s), A = Tables.getcolumn(A, :A), A_s = Tables.getcolumn(A, :A_s)))
    
    @test all(MLJBase.predict(mach_ratio, LA, LAδ) .== 1.0)
    
    LAδ = replacetable(LA, (L1 = Tables.getcolumn(L, :L1), L1_s = Tables.getcolumn(L, :L1_s), A = Tables.getcolumn(A, :A) .+ 0.1, A_s = Tables.getcolumn(A, :A_s) .+ (adjacency_matrix(getgraph(data_net)) * (ones(nv(data_net.graph)) .* 0.1))))
    
    g0shift = pdf.(condensity(dgp_net, LAδ, :A), Tables.getcolumn(LAδ, :A)) .* pdf.(condensity(dgp_net, LAδ, :A_s), Tables.getcolumn(LAδ, :A_s))
    g0 = pdf.(condensity(dgp_net, LA, :A), Tables.getcolumn(LA, :A)) .* pdf.(condensity(dgp_net, LA, :A_s), Tables.getcolumn(LA, :A_s))
    
    foo = MLJBase.predict(mach_ratio, LA, LAδ)
    true_ratio = g0 ./ g0shift
    @test foo ≈ true_ratio
end

@testset "CrossFitModel" begin   
    # Test a regression model
    LA = replacetable(data_net, TableOperations.select(data_net, :L1, :A) |> Tables.columntable)    
    Y = Tables.getcolumn(LA, :A) .+ 0.5 .* Tables.getcolumn(LA, :L1) .+ 10
    mean_estimator = MLJLinearModels.LinearRegressor()
    mean_crossfit = CrossFitModel(mean_estimator, CV())
    mach_mean = machine(mean_crossfit, LA, Y) |> fit!
    pred_mean = MLJBase.predict(mach_mean, LA)
    @test cor(Y, pred_mean) ≈ 1.0

    # TODO: Test this for network data. Note that currently CrossFitModel requires IID data because
    # if data are split using vanilla CV, the summary functions will no longer be correct
    ratio_model = DecomposedPropensityRatio(DensityRatioPlugIn(OracleDensityEstimator(dgp_iid)))
    ratio_crossfit = CrossFitModel(ratio_model, CV())
    L = TableOperations.select(data_net, :L1) |> Tables.columntable
    A = TableOperations.select(data_net, :A) |> Tables.columntable
    mach_ratio = machine(ratio_crossfit, L, A) |> fit!
    LAδ = replacetable(LA, (L1 = Tables.getcolumn(L, :L1), A = Tables.getcolumn(A, :A)))

    @test all(MLJBase.predict(mach_ratio, LA, LAδ) .== 1.0)

    LAδ = replacetable(LA, (L1 = Tables.getcolumn(L, :L1), A = Tables.getcolumn(A, :A) .+ 0.1))
    g0shift = pdf.(condensity(dgp_iid, LAδ, :A), Tables.getcolumn(LAδ, :A))
    g0 = pdf.(condensity(dgp_iid, LA, :A), Tables.getcolumn(LA, :A))
    
    foo = MLJBase.predict(mach_ratio, LA, LAδ)
    true_ratio = g0 ./ g0shift

    @test foo ≈ true_ratio
end

@testset "MTP IID" begin
  
    Random.seed!(1)
    
    data_large = rand(dgp_iid, 10^4)
    intervention = LinearShift(1.1, 0.5)
    truth = compute_true_MTP(dgp_iid, data_large, intervention)

    moe = 0.2

    mean_estimator = LinearRegressor()
    density_ratio_estimator = DensityRatioPlugIn(OracleDensityEstimator(dgp_iid))
    boot_sampler = BasicSampler()
    cv_splitter = nothing#CV(nfolds = 5)

    mtp = MTP(mean_estimator, density_ratio_estimator, cv_splitter)
    mtpmach = machine(mtp, data_large, intervention) |> fit!
    output = ModifiedTreatment.estimate(mtpmach, intervention)
    
    ψ_est = ψ(output)

    @test within(ψ_est.plugin, truth.ψ, moe)
    @test within(ψ_est.ipw, truth.ψ, moe)
    @test within(ψ_est.onestep, truth.ψ, moe)
    @test within(ψ_est.tmle, truth.ψ, moe)
    
    σ2_est = values(σ2(output))
    @test isnothing(σ2_est[1])
    #@test all(σ2_est[2:end] .< moe)

    @test all(isnothing.(values(σ2boot(output))))
    @test all(isnothing.(values(σ2net(output))))

    # TODO: Add better tests to ensure the bootstrap is working correctly
    B = 10
    ModifiedTreatment.bootstrap!(BasicSampler(), output, B)
    output
    @test all(values(σ2boot(output)) .< moe)
end 

@testset "MTP Network" begin
    Random.seed!(1)
    moe = 0.2

    distseqnet = @dgp(
        L1 ~ DiscreteUniform(1, 5),
        A ~ (@. Normal(:L1, 0.5)),
        As = Sum(:A, include_self = false),
        Y ~ (@. Normal(:A + :As + 0.1 * :L1 + 10, 1))
    );

    # Note this only yields clusters for K = 1, not any other K
    dgp_net = DataGeneratingProcess(n -> random_regular_graph(n, 1), distseqnet; 
                            treatment = :A, response = :Y, controls = [:L1]);

    
    data_vlarge = rand(dgp_net, 10^6)
    data_large = rand(dgp_net, 10^4)
    intervention = AdditiveShift(0.1)
    
    truth = compute_true_MTP(dgp_net, data_vlarge, intervention)
    mean_estimator = LinearRegressor()
    density_ratio_estimator = DensityRatioPlugIn(OracleDensityEstimator(dgp_net))
    cv_splitter = nothing#CV(nfolds = 5)

    mtp = MTP(mean_estimator, density_ratio_estimator, cv_splitter)
    mtpmach = machine(mtp, data_large, intervention) |> fit!

    output = ModifiedTreatment.estimate(mtpmach, intervention)
    ψ_est = ψ(output)

    @test within(ψ_est.plugin, truth.ψ, moe)
    @test within(ψ_est.ipw, truth.ψ, moe)
    @test within(ψ_est.sipw, truth.ψ, moe)
    @test within(ψ_est.onestep, truth.ψ, moe)
    @test within(ψ_est.tmle, truth.ψ, moe)

    σ2_est = values(σ2(output))
    @test isnothing(σ2_est[1])
    
    σ2net_est  = values(σ2net(output))
    @test isnothing(σ2_est[1])

    @test all(isnothing.(values(σ2boot(output))))

    # TODO: Add better tests to ensure the bootstrap is working correctly
    # Test the cluster bootstrap
    B = 100
    clustersampler = ClusterSampler(2)
    ModifiedTreatment.bootstrap!(clustersampler, output, B)  
    σ2boot_est = values(σ2boot(output))
    @test all(values(σ2boot_est) .< moe)

    # Ensure bootstrap and network variance estimator yield roughly same variance
    @test all(within.(σ2net_est[2:end], σ2boot_est[2:end], moe))

    # Test graph updating scheme
    # and that the basic bootstrap works on a graph with otherwise-IID data
    data_small = rand(dgp_net, 10^4)
    mtpmach2 = machine(mtp, data_small, intervention) |> fit!
    output2 = ModifiedTreatment.estimate(mtpmach2, intervention)
    basicsampler = BasicSampler()
    ModifiedTreatment.bootstrap!(basicsampler, output2, B)  
    σ2boot_est2 = values(σ2boot(output2))
    @test all(σ2boot_est2 .< moe)
    @test all(σ2boot_est .- σ2boot_est2 .< moe * 0.1)
end

"""
@testset "Super Learning" begin

    dgp = DataGeneratingProcess(
        @dgp(
            L1 ~ DiscreteUniform(1, 4),
            L2 ~ Poisson(10),
            L3 ~ Exp(1)
            A ~ @. Normal(log(:L2) + :L3^2, 0.5),
            Y ~ @. Normal(:A + :L1 * (0.5 * sin(:L2) + 0.1 * :L3)  + 10, 1)
        );
    )
end
"""

