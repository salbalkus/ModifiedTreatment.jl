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
KNNRegressor = @load KNNRegressor pkg=NearestNeighborModels
LGBMRegressor = @load LGBMRegressor pkg=LightGBM
XGBoostRegressor = @load XGBoostRegressor pkg=XGBoost

# Classifiers
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
LGBMClassifier = @load LGBMClassifier pkg=LightGBM
XGBoostClassifier = @load XGBoostClassifier pkg=XGBoost

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
    #ER = Graphs.adjacency_matrix(Graphs.erdos_renyi(length(L1), 5/length(L1))),
    ER = Graphs.adjacency_matrix(Graphs.static_scale_free(length(L1), 2 * length(L1), 3.5)),
    #ER = Graphs.adjacency_matrix(Graphs.watts_strogatz(length(L1), 5, 0.5)),
    L1_s $ Sum(:L1, :ER),
    A ~ (@. Normal(L1 + 1, 1.0)),
    A_s $ Sum(:A, :ER),
    Y ~ (@. Normal(A + 0.5 * A_s + L1 + 10, 0.5))
);

scm_net =  StructuralCausalModel(
    dgp_net;
    treatment = :A,
    response = :Y,
    confounders = [:L1]
    )
data_net = rand(scm_net, 100)
data_net_sum = summarize(data_net)

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
    
    @test Ls.data == (L1 = data_net_sum.data.L1, L1_s = data_net_sum.data.L1_s,)
    @test As.data == (A = data_net_sum.data.A, A_s = data_net_sum.data.A_s,)
    @test LAs.data == (L1 = data_net_sum.data.L1, A = data_net_sum.data.A, L1_s = data_net_sum.data.L1_s, A_s = data_net_sum.data.A_s)
    
    LAδs, dAδs = transform(intmach, intervention)
    
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
    
    truth = compute_true_MTP(scm_test, data_test, intervention).ψ

    @test within(truth, 3, 0.01)
end

@testset "DecomposedPropensityRatio on Network" begin
    LA = CausalTables.replace(data_net; data = data_net_sum |> TableTransforms.Select(:L1, :L1_s, :A, :A_s))    
    ratio_model = DecomposedPropensityRatio(DensityRatioPlugIn(OracleDensityEstimator(scm_net)))
    L = data_net_sum |> Select(:L1, :L1_s) |> Tables.columntable
    A = data_net_sum |> Select(:A, :A_s) |> Tables.columntable
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

@testset "SuperLearner" begin
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

    sl = XGBoostClassifier(objective = "binary:logistic")
    density_ratio_estimator = DensityRatioClassifier(sl)

    # TODO: Test this for network data.
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
                       LGBMClassifier(objective = "binary", linear_tree = true),
                       LGBMClassifier(objective = "binary", linear_tree = true, min_data_in_leaf = 10)
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
    B = 1000
    ModifiedTreatment.bootstrap!(BasicSampler(), output, B)

    output
    σ2boot(output)
    σ2(output)
    
    @test all(values(σ2boot(output)) .< moe)

end 

#@testset "MTP Network" begin
    Random.seed!(1)
    moe = 1.0

    distseqnet = @dgp(
        L1 ~ DiscreteUniform(1, 4),
        ER1 = Graphs.adjacency_matrix(erdos_renyi(length(L1), 4/length(L1))),
        ER2 = ER1,
        L1s $ Sum(:L1, :ER1),
        L2 ~ Binomial(3, 0.4),
        #ER2 = Graphs.adjacency_matrix(Graphs.erdos_renyi(length(L1), 4/length(L1))),
        L2s $ Sum(:L2, :ER2),
        L3 ~ Beta(3, 2),
        L4 ~ Poisson(5),
        L3s $ AllOrderStatistics(:L3, :ER1),
        L4s $ AllOrderStatistics(:L4, :ER1),
        reg = (@. log(L3) + sqrt(L4)),
        A ~ (@. Normal(reg + 1, 1)),
        As $ Sum(:A, :ER1),
        Y ~ (@. Normal(A + 0.5 * As + 0.5 * L1s + 0.05 * L2s, 0.1))
    );

    scm_net =  CausalTables.StructuralCausalModel(
        distseqnet;
        treatment = :A,
        response = :Y,
        confounders = [:L1, :L2, :L3, :L4]
        )

    n_large = 10000
    data_vlarge = rand(scm_net, 10^6)
    data_large = rand(scm_net, n_large)

    intervention = AdditiveShift(0.1)
    
    truth = compute_true_MTP(scm_net, data_vlarge, intervention)
    #mean_estimator = LinearRegressor()
    mean_estimator = XGBoostRegressor(objective = "reg:squarederror")
    #density_ratio_estimator = DensityRatioPlugIn(OracleDensityEstimator(scm_net))
    sl = XGBoostClassifier(objective = "binary:logistic", eta = 0.01)
    density_ratio_estimator = DensityRatioClassifier(sl)
    cv_splitter = CV(nfolds = 5)

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
    
    σ2net_est = values(σ2net(output))
    values(σ2net_est)[3:4] .* n_large
    @test isnothing(σ2_est[1])
    @test all(within.(values(σ2net_est)[4:5] .* n_large, truth.eff_bound, moe * 20))
    @test all(isnothing.(values(σ2boot(output))))

    # TODO: Add better tests to ensure the bootstrap is working correctly
    # Test the cluster bootstrap
    B = 100
    ModifiedTreatment.bootstrap!(BasicSampler(), output, B)
    σ2boot_est = σ2boot(output)
    @test !all(within.(values()[4:5] .* n_large, truth.eff_bound, moe))
    @test all(values(σ2boot(output)) .< moe)

    sim = []
    using Logging
    disable_logging(Logging.Info)
    disable_logging(Logging.Warn)
    for i in 1:100
        data_large = rand(scm_net, n_large)
        mtpmach = fit!(machine(mtp, data_large, intervention), verbosity=0) 
        output = ModifiedTreatment.estimate(mtpmach, intervention)
        push!(sim, ψ(output).tmle)
    end
    var(sim)

    r = report(mtpmach)
    D  = r.Hn .* (data_large.data.Y .- r.Qn) .+ r.Qδn
    G = CausalTables.dependency_matrix(data_large)
    D = D .- ModifiedTreatment.neighbor_center(D, data_large.arrays.ER)
    
    transpose(D) * G * D / n_large^2

    gamma = 20
    using SparseArrays
    using LinearAlgebra
    V = Matrix(G .+ (gamma * sparse(I, n_large, n_large)))
    #V = Matrix(gamma .* sparse(I, n_large, n_large) .+ 1)
    edist = MvNormal(V)

    e = rand(edist, 100000)
    fullvar = var(mean(D .* e, dims = 1))
    v = var(D) / n_large
    
    cv = (fullvar .- (v .* gamma))
    bootvar = v + cv

end


