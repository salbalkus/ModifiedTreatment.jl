using Test
using ModifiedTreatment
using CausalTables, Condensity
using MLJ
using MLJBase
using Revise
using Distributions, Graphs
using Tables, TableTransforms, DataAPI
using DensityRatioEstimation 

# Regressors
DeterministicConstantRegressor = @load DeterministicConstantRegressor pkg=MLJModels
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
KNNRegressor = @load KNNRegressor pkg=NearestNeighborModels

# Classifiers
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
LGBMClassifier = @load LGBMClassifier pkg=LightGBM


using Random
#using Logging

Random.seed!(1)

# function for testing approximate equality of statistical estimators
within(x, truth, ϵ) = abs(x - truth) < ϵ

dgp_iid = @dgp(
    L1 ~ DiscreteUniform(1, 5),
    A ~ (@. Normal(L1, 1)),
    Y ~ (@. Normal(A + 0.5 * L1 + 10, 0.5))
)
scm_iid = StructuralCausalModel(dgp_iid; treatment = :A, response = :Y, confounders = [:L1]);
data_iid = rand(scm_iid, 100)

dgp_net = @dgp(
    L1 ~ Binomial(5, 0.4),
    #ER = Graphs.adjacency_matrix(Graphs.random_regular_graph(length(L1), 2)),
    ER = Graphs.adjacency_matrix(Graphs.erdos_renyi(length(L1), 5/length(L1))),
    L1_s $ Sum(:L1, :ER),
    A ~ (@. Normal(L1 + 1, 0.5)),
    A_s $ Sum(:A, :ER),
    Y ~ (@. Normal(A + 0.5 * A_s + L1 + L1_s + 10, 0.5))
);

scm_net =  StructuralCausalModel(
    dgp_net;
    treatment = :A,
    response = :Y,
    confounders = [:L1]
    )
data_net = summarize(rand(scm_net, 100))


@testset "Intervention" begin
    m = 1.5
    a = 0.5
    int1 = AdditiveShift(a)
    int2 = MultiplicativeShift(m)
    int3 = LinearShift(m, a)

    inv_int1 = ModifiedTreatment.inverse(int1)
    inv_int2 = ModifiedTreatment.inverse(int2)
    inv_int3 = ModifiedTreatment.inverse(int3)

    L = CausalTables.confounders(data_net)
    A = Tables.getcolumn(CausalTables.treatment(data_net), data_net.treatment[1])

    @test apply_intervention(int1, L, A) ≈ A .+ 0.5
    @test apply_intervention(inv_int1, L, A) ≈ A .- 0.5
    @test differentiate_intervention(int1, L, A) ≈ 1
    @test differentiate_intervention(inv_int1, L, A) ≈ 1
    
    @test apply_intervention(int2, L, A) ≈ A .* 1.5
    @test apply_intervention(inv_int2, L, A) ≈ A ./ 1.5
    @test differentiate_intervention(int2, L, A) ≈ 1.5
    @test differentiate_intervention(ModifiedTreatment.inverse(int2), L, A) ≈ 1/1.5

    @test apply_intervention(int3, L, A) ≈ A .* 1.5 .+ 0.5
    @test apply_intervention(inv_int3, L, A) ≈ (A .- 0.5) ./ 1.5
    @test differentiate_intervention(int3, L, A) ≈ 1.5
    @test differentiate_intervention(inv_int3, L, A) ≈ 1/1.5

    inducedint = get_induced_intervention(int3, Sum(:A, :ER))
    inv_inducedint = ModifiedTreatment.inverse(inducedint)
    
    @test apply_intervention(inducedint, L, A) ≈ A .* m .+ data_net.arrays.ER * (ones(size(data_net.arrays.ER, 1)) .* a)
    @test apply_intervention(inv_inducedint, L, A) ≈ (A .- data_net.arrays.ER * (ones(size(data_net.arrays.ER, 1)) .* a)) ./ m
    @test all(differentiate_intervention(inducedint, L, A) .≈ 1.5)
    @test all(differentiate_intervention(inv_inducedint, L, A) .≈ 1/1.5)
end

@testset "InterventionModel" begin
    intervention = LinearShift(1.5, 0.5)
    intmach = machine(ModifiedTreatment.InterventionModel(), data_net) |> fit!
    LAs, Ls, As = predict(intmach, intervention)
    @test Ls.data == (L1 = data_net.data.L1, L1_s = data_net.data.L1_s,)
    @test As.data == (A = data_net.data.A, A_s = data_net.data.A_s,)
    @test LAs.data == (L1 = data_net.data.L1, A = data_net.data.A, L1_s = data_net.data.L1_s, A_s = data_net.data.A_s)
    
    LAδs, dAδs = transform(intmach, intervention)
    LAδs
    @test LAδs.data.A ≈ (data_net.data.A .* 1.5) .+ 0.5
    @test LAδs.data.A_s ≈ data_net.arrays.ER * ((data_net.data.A .* 1.5) .+ 0.5)
    @test dAδs.A_s == 1.5
    @test dAδs.A == 1.5

    LAδsinv, dAδsinv = inverse_transform(intmach, intervention)
    @test LAδsinv.data.A ≈ (data_net.data.A .- 0.5) ./ 1.5
    @test LAδsinv.data.A_s ≈ data_net.arrays.ER * ((data_net.data.A .- 0.5) ./ 1.5)
    @test dAδsinv.A_s == 1/1.5
    @test dAδsinv.A == 1/1.5
end

@testset "Truth" begin
    scm_test = StructuralCausalModel(
        @dgp(
            L ~ Bernoulli(0.0),
            A ~ (@. Normal(L, 0.01)),
            ER = Graphs.adjacency_matrix(Graphs.random_regular_graph(length(A), 2)),
            As $ Sum(:A, :ER),
            Y ~ (@. Normal(L + A + As, 0.01))
        ),
        treatment = :A,
        response = :Y,
        confounders = [:L]
    )
    data_test = rand(scm_test, 10000)
    intervention = AdditiveShift(1.0)
    
    ψ = compute_true_MTP(scm_test, data_test, intervention).ψ

    @test within(ψ, 3, 0.01)
end

@testset "DecomposedPropensityRatio on Network" begin
    
    LA = CausalTables.replace(data_net; data = data_net |> TableTransforms.Select(:L1, :L1_s, :A, :A_s))    
    ratio_model = DecomposedPropensityRatio(DensityRatioPlugIn(OracleDensityEstimator(scm_net)))
    L = data_net |> Select(:L1, :L1_s) |> Tables.columntable
    A = data_net |> Select(:A, :A_s) |> Tables.columntable
    mach_ratio = machine(ratio_model, L, A) |> fit!
    LAδ = CausalTables.replace(LA; data= merge(L, (A = Tables.getcolumn(A, :A), A_s = Tables.getcolumn(A, :A_s))))
    predict(mach_ratio, LA, LAδ)
    @test all(MLJ.predict(mach_ratio, LA, LAδ) .== 1.0)
    
    LAδ = CausalTables.replace(LA; data = merge(L, (A = Tables.getcolumn(A, :A) .+ 0.1, A_s = Tables.getcolumn(A, :A_s) .+ (data_net.arrays.ER * (ones(size(data_net.arrays.ER, 1)) .* 0.1)))))
    
    g0shift = pdf.(condensity(scm_net, LAδ, :A), Tables.getcolumn(LAδ, :A)) .* pdf.(condensity(scm_net, LAδ, :A_s), Tables.getcolumn(LAδ, :A_s))
    g0 = pdf.(condensity(scm_net, LA, :A), Tables.getcolumn(LA, :A)) .* pdf.(condensity(scm_net, LA, :A_s), Tables.getcolumn(LA, :A_s))
    
    foo = MLJ.predict(mach_ratio, LA, LAδ)
    true_ratio = g0 ./ g0shift
    @test foo ≈ true_ratio
end

#@testset "SuperLearner" begin
    Random.seed!(1)
    # Start by testing the deterministic version
    X = data_iid |> TableTransforms.Select(:L1, :A)
    y = Tables.getcolumn(data_iid, data_iid.response[1])
    sl = SuperLearner([DeterministicConstantRegressor(), LinearRegressor(), DecisionTreeRegressor(), KNNRegressor()], CV())
    
    mach = machine(sl, X, y) |> fit!
    yhat = predict(mach, X)

    @test length(yhat) == length(y)
    @test typeof(report(mach).best_model) <: LinearRegressor  # should select the linear model as the best

    # Now test the probabilistic version
    y = coerce(y .> 15, OrderedFactor)
    sl = SuperLearner([ConstantClassifier(), LogisticClassifier(), LGBMClassifier(objective = "binary", metric = ["binary_logloss"], linear_tree = true)], CV())
    mach = machine(sl, X, y) |> fit!
    yhat = pdf.(predict(mach, X), true)

    @test length(yhat) == length(y)
    @test typeof(report(mach).best_model) <: LogisticClassifier  # should select the linear model as the best
    @test mean((yhat .> 0.5) .== y) > 0.9 # should get at least 90% accuracy
end

@testset "CrossFitModel" begin   
    # Test a regression model
    LA = CausalTables.replace(data_net; data = data_net |> CausalTables.Select(:L1, :A))    
    Y = Tables.getcolumn(LA, :A) .+ 0.5 .* Tables.getcolumn(LA, :L1) .+ 10
    mean_estimator = LinearRegressor()
    mean_crossfit = CrossFitModel(mean_estimator, CV())
    mach_mean = machine(mean_crossfit, LA, Y) |> fit!
    pred_mean = MLJ.predict(mach_mean, LA)
    @test cor(Y, pred_mean) ≈ 1.0

    # TODO: Test this for network data. Note that currently CrossFitModel requires IID data because
    # if data are split using vanilla CV, the summary functions will no longer be correct
    ratio_model = DecomposedPropensityRatio(DensityRatioPlugIn(OracleDensityEstimator(scm_iid)))
    ratio_crossfit = CrossFitModel(ratio_model, CV())
    L = data_net |> CausalTables.Select(:L1) |> Tables.columntable
    A = data_net |> CausalTables.Select(:A) |> Tables.columntable
    mach_ratio = machine(ratio_crossfit, L, A) |> fit!
    LAδ = CausalTables.replace(LA; data = (L1 = Tables.getcolumn(L, :L1), A = Tables.getcolumn(A, :A)))

    @test all(MLJ.predict(mach_ratio, LA, LAδ) .== 1.0)

    LAδ = CausalTables.replace(LA; data = (L1 = Tables.getcolumn(L, :L1), A = Tables.getcolumn(A, :A) .+ 0.1))
    g0shift = pdf.(condensity(scm_iid, LAδ, :A), Tables.getcolumn(LAδ, :A))
    g0 = pdf.(condensity(scm_iid, LA, :A), Tables.getcolumn(LA, :A))
    
    foo = MLJ.predict(mach_ratio, LA, LAδ)
    true_ratio = g0 ./ g0shift

    @test foo ≈ true_ratio
end

@testset "MTP IID" begin
  
    Random.seed!(1)
    
    data_large = rand(scm_iid, 10^3)
    intervention = LinearShift(1.001, 0.1)
    truth = compute_true_MTP(scm_iid, data_large, intervention)

    moe = 0.1

    mean_estimator = LinearRegressor()
    boot_sampler = BasicSampler()
    cv_splitter = CV(nfolds = 5)

    # Probabilistic Classifier
    sl = SuperLearner([LogisticClassifier(), 
                       LGBMClassifier(objective = "cross_entropy", linear_tree = true),
                       LGBMClassifier(objective = "cross_entropy", linear_tree = true, min_data_in_leaf = 10)
                       ], CV())
    density_ratio_estimator = DensityRatioClassifier(sl)
    mtp = ModifiedTreatment.MTP(mean_estimator, density_ratio_estimator, cv_splitter)
    mtpmach = machine(mtp, data_large, intervention) |> fit!
    output = ModifiedTreatment.estimate(mtpmach, intervention)

    # KLEIP
    density_ratio_estimator = DensityRatioKLIEP([0.5, 1.0, 2.0], [100])
    mtp = MTP(mean_estimator, density_ratio_estimator, nothing)
    mtpmachk = machine(mtp, data_large, intervention) |> fit!
    output_kliep = ModifiedTreatment.estimate(mtpmachk, intervention)

    density_ratio_oracle = DensityRatioPlugIn(OracleDensityEstimator(scm_iid))
    mtp_oracle = MTP(mean_estimator, density_ratio_oracle, nothing)
    mtpmach_oracle = machine(mtp_oracle, data_large, intervention) |> fit!
    output_oracle = ModifiedTreatment.estimate(mtpmach_oracle, intervention)

    ψ_est = ψ(output)
    ψ_est2 = ψ(output_kliep)
    ψ_oracle = ψ(output_oracle)

    @test within(ψ_est.plugin, truth.ψ, moe)
    @test within(ψ_est.sipw, truth.ψ, moe)
    @test within(ψ_est.onestep, truth.ψ, moe)
    @test within(ψ_est.tmle, truth.ψ, moe)

    maximum(abs.(report(mtpmach_oracle).Hn .- report(mtpmachk).Hn))

    # ensure ratio nuisance is similar
    @test maximum(abs.(report(mtpmach_oracle).Hn .- report(mtpmach).Hn)) < moe
    σ2net(output)
    σ2_est = values(σ2(output))
    
    @test isnothing(σ2_est[1])
    #@test all(σ2_est[2:end] .< moe)
    isnothing.(values(σ2net(output)))
    @test all(isnothing.(values(σ2boot(output))))

    # TODO: Add better tests to ensure the bootstrap is working correctly
    #B = 10
    #ModifiedTreatment.bootstrap!(BasicSampler(), output, B)
    #@test all(values(σ2boot(output)) .< moe)
end 

@testset "MTP Network" begin
    Random.seed!(1)
    moe = 1.0

    distseqnet = @dgp(
        L1 ~ DiscreteUniform(1, 4),
        ER1 = Graphs.adjacency_matrix(Graphs.erdos_renyi(length(L1), 3/length(L1))),
        L2 ~ Binomial(3, 0.4),
        L2s $ Sum(:L2, :ER1),
        L3 ~ Beta(3, 2),
        L4 ~ Poisson(3),
        ER2 = Graphs.adjacency_matrix(Graphs.erdos_renyi(length(L1), 3/length(L1))),
        L4s $ Sum(:L4, :ER2),
        reg = 0.5 * (L1 + L2) + L3 + 0.05 * (L4 + L2s) + 0.01 * L4s,
        A ~ (@. Normal(reg + 1, 1)),
        As $ Sum(:A, :ER1),
        Y ~ (@. Normal(A + 0.5 * As + reg + 100, 1))
    );

    distseqnet = @dgp(
        L1 ~ Normal(0, 2),
        ER = Graphs.adjacency_matrix(Graphs.erdos_renyi(length(L1), 3/length(L1))),
        L2 ~ Normal(1, 1),
        L3 ~ Bernoulli(0.3),
        L4 ~ Bernoulli(0.6),
        A ~ (@. Normal(0.5 * (L1 + L2 + L3) + 0.1 * L4 + 1, 1)),
        As $ Sum(:A, :ER),
        Y ~ (@. Normal(A + As + 0.1 * L1 + 0.2 * L2 + 0.1 * L3 + 0.2 * L4 + 100, 1))
    );


    scm_net =  CausalTables.StructuralCausalModel(
        distseqnet;
        treatment = :A,
        response = :Y,
        confounders = [:L1, :L2, :L3, :L4]
        )


    n_large = 1000
    data_vlarge = rand(scm_net, 10^6)
    data_large = rand(scm_net, n_large)
    intervention = AdditiveShift(0.1)
    
    truth = compute_true_MTP(scm_net, data_vlarge, intervention)
    mean_estimator = LinearRegressor()
    #density_ratio_estimator = DensityRatioPlugIn(OracleDensityEstimator(scm_net))
    sl = SuperLearner([LogisticClassifier(), 
                       LGBMClassifier(objective = "cross_entropy", linear_tree = true),
                       LGBMClassifier(objective = "cross_entropy", linear_tree = true, min_data_in_leaf = 10)
                       ], CV())
    density_ratio_estimator = DensityRatioClassifier(sl)
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
    @test !all(within.(values(σ2_est)[4:5] .* n_large, truth.eff_bound, moe))
    
    values(σ2_est)[4:5] .* n_large
    truth.eff_bound
    σ2net_est  = values(σ2net(output))
    @test isnothing(σ2_est[1])
    @test all(within.(values(σ2net_est)[4:5] .* n_large, truth.eff_bound, moe * 10))
    @test all(isnothing.(values(σ2boot(output))))

    # TODO: Add better tests to ensure the bootstrap is working correctly
    # Test the cluster bootstrap
    """
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
    """
end


