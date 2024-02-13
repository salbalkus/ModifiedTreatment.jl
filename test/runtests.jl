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
                            treatment = :A_s, response = :Y, controls = [:L1, :L1_s, :A]);
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
    @test Ls.tbl == (L1 = data_net.tbl.L1, L1_s = data_net.tbl.L1_s, A = data_net.tbl.A)
    @test As == (A_s = data_net.tbl.A_s,)

    As
    @test LAs.tbl == (L1 = data_net.tbl.L1, L1_s = data_net.tbl.L1_s, A = data_net.tbl.A, A_s = data_net.tbl.A_s)

    LAδs, dAδs = transform(intmach, intervention)
    @test LAδs.tbl.A_s ≈ adjacency_matrix(data_net.graph) * ((data_net.tbl.A .* 1.5) .+ 0.5)
    @test dAδs.A_s == 1.5

    LAδsinv, dAδsinv = inverse_transform(intmach, intervention)
    @test LAδsinv.tbl.A_s ≈ adjacency_matrix(data_net.graph) * ((data_net.tbl.A .- 0.5) ./ 1.5)
    @test dAδsinv.A_s == 1/1.5
end

@testset "CrossFitModel" begin   
    # Test a regression model
    LA = replacetable(data_iid, TableOperations.select(data_iid, :L1, :A) |> Tables.columntable)    
    Y = Tables.getcolumn(LA, :A) .+ 0.5 .* Tables.getcolumn(LA, :L1) .+ 10
    mean_estimator = MLJLinearModels.LinearRegressor()
    mean_crossfit = CrossFitModel(mean_estimator, CV())
    mach_mean = machine(mean_crossfit, LA, Y) |> fit!
    pred_mean = MLJBase.predict(mach_mean, LA)
    @test cor(Y, pred_mean) ≈ 1.0
    ratio_model = DensityRatioPropensity(OracleDensityEstimator(dgp_iid))
    ratio_crossfit = CrossFitModel(ratio_model, CV())
   
    L = replacetable(data_iid, TableOperations.select(data_iid, :L1) |> Tables.columntable)
    A = replacetable(data_iid, TableOperations.select(data_iid, :A) |> Tables.columntable)
    mach_ratio = machine(ratio_crossfit, L, A) |> fit!
    
    LAδ = replacetable(LA, (L1 = Tables.getcolumn(L, :L1), A = Tables.getcolumn(A, :A) ))
    @test all(MLJBase.predict(mach_ratio, LA, LAδ) .== 1.0)

    LAδ = replacetable(LA, (L1 = Tables.getcolumn(L, :L1), A = Tables.getcolumn(A, :A) .+ 0.1))
    g0shift = pdf.(condensity(dgp_iid, L, :A), Tables.getcolumn(LAδ, :A))
    g0 = pdf.(condensity(dgp_iid, L, :A), Tables.getcolumn(LA, :A))
    MLJBase.predict(mach_ratio, LA, LAδ)
    
    foo = MLJBase.predict(mach_ratio, LA, LAδ)
    true_ratio = g0 ./ g0shift

    @test foo ≈ true_ratio
end

@testset "MTP IID" begin
  
    Random.seed!(1)
    
    data_large = rand(dgp_iid, 10^4)

    intervention = LinearShift(1.1, 0.5)
    truth = compute_true_MTP(dgp_iid, data_large, intervention)

    moe = 0.1

    mean_estimator = LinearRegressor()
    density_ratio_estimator = DensityRatioPropensity(OracleDensityEstimator(dgp_iid))
    boot_sampler = BasicSampler()
    cv_splitter = nothing#CV(nfolds = 5)

    mtp = MTP(mean_estimator, density_ratio_estimator, cv_splitter)
    mtpmach = machine(mtp, data_large, intervention) |> fit!
    output = ModifiedTreatment.estimate(mtpmach, intervention)
    
    ψ_est = ψ(output)

    @test within(ψ_est.or, truth.ψ, moe)
    @test within(ψ_est.ipw, truth.ψ, moe)
    @test within(ψ_est.onestep, truth.ψ, moe)
    @test within(ψ_est.tmle, truth.ψ, moe)
    

    # TODO: Add better tests to ensure the bootstrap is working correctly
    B = 10
    ModifiedTreatment.bootstrap!(BasicSampler(), output, B)
    @test all(values(σ2boot(output)) .< moe)
end 


#@testset "MTP Network" begin
    Random.seed!(1)
    moe = 0.1

    distseqnet = Vector{Pair{Symbol, CausalTables.ValidDGPTypes}}([
        :L1 => (; O...) -> DiscreteUniform(1, 5),
        :A => (; O...) -> (@. Normal(0.2 * O[:L1], 0.5)),
        :A_s => Sum(:A, include_self = true),
        :Y => (; O...) -> (@. Normal(O[:A_s] + 0.1 * O[:L1] + 10, 1))
    ])

    distseqnet = @dgp(
        L1 ~ DiscreteUniform(1, 5),
        A ~ (@. Normal(0.2 * :L1, 0.5)),
        As = Sum(:A, include_self = false),
        Y ~ (@. Normal(:As + 0.1 * :L1 + 10, 1))
    );

    # Note this only yields clusters for K = 1, not any other K
    dgp_net = DataGeneratingProcess(n -> random_regular_graph(n, 1), distseqnet; 
                            treatment = :As, response = :Y, controls = [:L1]);

    map(get_var_to_summarize, getsummaries(data_large))
    
    data_vlarge = rand(dgp_net, 10^6)
    data_large = rand(dgp_net, 10^3)

    intervention = AdditiveShift(0.1)
    
    truth = compute_true_MTP(dgp_net, data_vlarge, intervention)
    mean_estimator = LinearRegressor()
    density_ratio_estimator = DensityRatioPropensity(OracleDensityEstimator(dgp_net))
    cv_splitter = nothing#CV(nfolds = 5)

    mtp = MTP(mean_estimator, density_ratio_estimator, cv_splitter)
    mtpmach = machine(mtp, data_large, intervention) |> fit!

    output = ModifiedTreatment.estimate(mtpmach, intervention)
    ψ_est = ψ(output)
    truth
    @test within(ψ_est.or, truth.ψ, moe)
    @test within(ψ_est.ipw, truth.ψ, moe)
    @test within(ψ_est.onestep, truth.ψ, moe)
    @test within(ψ_est.tmle, truth.ψ, moe)
    
    # TODO: Add better tests to ensure the bootstrap is working correctly

    # Test the cluster bootstrap
    B = 10
    clustersampler = ClusterSampler(2)
    ModifiedTreatment.bootstrap!(clustersampler, output, B)  

    σ2boot_est = σ2boot(output)
    values(σ2boot_est) .* 10^3
    @test all(values(σ2boot_est) .< moe)

    # Test graph updating scheme
    data_small = rand(dgp_net, 10^4)
    mtpmach2 = machine(mtp, data_small, intervention) |> fit!
    output2 = ModifiedTreatment.estimate(mtpmach2, intervention)
    ModifiedTreatment.bootstrap!(clustersampler, output2, B)  
    

    σ2boot_est2 = σ2boot(output2)
    values(σ2boot_est2) .* 10^2
    @test all(values(σ2boot_est) .< moe)
end

#@testset "Super Learning" begin

    dgp = DataGeneratingProcess(
        @dgp(
            L1 ~ DiscreteUniform(1, 4),
            L2 ~ Poisson(10),
            L3 ~ Exp(1)
            A ~ @. Normal(log(:L2) + :L3^2, 0.5),
            Y ~ @. Normal(:A + :L1 * (0.5 * sin(:L2) + 0.1 * :L3)  + 10, 1)
        );
    )
#end

