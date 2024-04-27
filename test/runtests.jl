using Test
using ModifiedTreatment
using CausalTables
using Distributions
using Graphs
using Condensity
using Tables
using TableOperations
using DataAPI
using MLJ
using DensityRatioEstimation

# Regressors
DeterministicConstantRegressor = @load DeterministicConstantRegressor pkg=MLJModels
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
KNNRegressor = @load KNNRegressor pkg=NearestNeighborModels

# Classifiers
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels

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

distseqnet = @dgp(
    L1 ~ Binomial(5, 0.4),
    L1_s = Sum(:L1, include_self = false),
    F = Friends(),
    A ~ (@. Normal(:L1 + 1, 0.5)),
    A_s = Sum(:A, include_self = false),
    Y ~ (@. Normal(:A + 0.5 * :A_s + :L1 + :L1_s + :F + 10, 0.5))
);

dgp_net =  DataGeneratingProcess(
    n -> Graphs.random_regular_graph(n, 2),
    distseqnet;
    treatment = :A,
    response = :Y,
    controls = [:L1]
    )
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
    @test gettable(Ls) == (L1 = data_net.tbl.L1, L1_s = data_net.tbl.L1_s, F = data_net.tbl.F,)
    @test As == (A = data_net.tbl.A, A_s = data_net.tbl.A_s,)
    @test gettable(LAs) == (L1 = data_net.tbl.L1, L1_s = data_net.tbl.L1_s, F = data_net.tbl.F, A = data_net.tbl.A, A_s = data_net.tbl.A_s)
    
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
    LAδ = replacetable(LA, merge(L, (A = Tables.getcolumn(A, :A), A_s = Tables.getcolumn(A, :A_s))))
    
    @test all(MLJ.predict(mach_ratio, LA, LAδ) .== 1.0)
    
    LAδ = replacetable(LA, merge(L, (A = Tables.getcolumn(A, :A) .+ 0.1, A_s = Tables.getcolumn(A, :A_s) .+ (adjacency_matrix(getgraph(data_net)) * (ones(nv(data_net.graph)) .* 0.1)))))
    
    g0shift = pdf.(condensity(dgp_net, LAδ, :A), Tables.getcolumn(LAδ, :A)) .* pdf.(condensity(dgp_net, LAδ, :A_s), Tables.getcolumn(LAδ, :A_s))
    g0 = pdf.(condensity(dgp_net, LA, :A), Tables.getcolumn(LA, :A)) .* pdf.(condensity(dgp_net, LA, :A_s), Tables.getcolumn(LA, :A_s))
    
    foo = MLJ.predict(mach_ratio, LA, LAδ)
    true_ratio = g0 ./ g0shift
    @test foo ≈ true_ratio
end

@testset "SuperLearner" begin
    Random.seed!(1)
    # Start by testing the deterministic version
    X = TableOperations.select(data_iid, :L1, :A) |> Tables.columntable
    y = getresponse(data_iid)
    sl = SuperLearner([DeterministicConstantRegressor(), LinearRegressor(), DecisionTreeRegressor(), KNNRegressor()], CV())
    mach = machine(sl, X, y) |> fit!
    yhat = predict(mach, X)

    @test length(yhat) == length(y)
    @test typeof(report(mach).best_model) <: LinearRegressor  # should select the linear model as the best


    # Now test the probabilistic version
    y = coerce(getresponse(data_iid) .> 15, OrderedFactor)
    
    sl = SuperLearner([ConstantClassifier(), LogisticClassifier(), DecisionTreeClassifier(), KNNClassifier()], CV())
    mach = machine(sl, X, y) |> fit!
    yhat = pdf.(predict(mach, X), true)

    @test length(yhat) == length(y)
    @test typeof(report(mach).best_model) <: KNNClassifier  # should select the linear model as the best
    @test mean((yhat .> 0.5) .== y) > 0.9 # should get at least 90% accuracy
end

@testset "CrossFitModel" begin   
    # Test a regression model
    LA = replacetable(data_net, TableOperations.select(data_net, :L1, :A) |> Tables.columntable)    
    Y = Tables.getcolumn(LA, :A) .+ 0.5 .* Tables.getcolumn(LA, :L1) .+ 10
    mean_estimator = LinearRegressor()
    mean_crossfit = CrossFitModel(mean_estimator, CV())
    mach_mean = machine(mean_crossfit, LA, Y) |> fit!
    pred_mean = MLJ.predict(mach_mean, LA)
    @test cor(Y, pred_mean) ≈ 1.0

    # TODO: Test this for network data. Note that currently CrossFitModel requires IID data because
    # if data are split using vanilla CV, the summary functions will no longer be correct
    ratio_model = DecomposedPropensityRatio(DensityRatioPlugIn(OracleDensityEstimator(dgp_iid)))
    ratio_crossfit = CrossFitModel(ratio_model, CV())
    L = TableOperations.select(data_net, :L1) |> Tables.columntable
    A = TableOperations.select(data_net, :A) |> Tables.columntable
    mach_ratio = machine(ratio_crossfit, L, A) |> fit!
    LAδ = replacetable(LA, (L1 = Tables.getcolumn(L, :L1), A = Tables.getcolumn(A, :A)))

    @test all(MLJ.predict(mach_ratio, LA, LAδ) .== 1.0)

    LAδ = replacetable(LA, (L1 = Tables.getcolumn(L, :L1), A = Tables.getcolumn(A, :A) .+ 0.1))
    g0shift = pdf.(condensity(dgp_iid, LAδ, :A), Tables.getcolumn(LAδ, :A))
    g0 = pdf.(condensity(dgp_iid, LA, :A), Tables.getcolumn(LA, :A))
    
    foo = MLJ.predict(mach_ratio, LA, LAδ)
    true_ratio = g0 ./ g0shift

    @test foo ≈ true_ratio
end

#@testset "MTP IID" begin
  
    Random.seed!(1)
    
    data_large = rand(dgp_iid, 10^3)
    intervention = LinearShift(1.001, 0.1)
    truth = compute_true_MTP(dgp_iid, data_large, intervention)

    moe = 0.1

    mean_estimator = LinearRegressor()
    boot_sampler = BasicSampler()
    cv_splitter = CV(nfolds = 5)

    # Probabilistic Classifier
    density_ratio_estimator = DensityRatioClassifier(LogisticClassifier())
    mtp = MTP(mean_estimator, density_ratio_estimator, cv_splitter)
    mtpmach = machine(mtp, data_large, intervention) |> fit!
    output = ModifiedTreatment.estimate(mtpmach, intervention)

    # KLEIP
    density_ratio_estimator = DensityRatioKLIEP([0.5, 1.0, 2.0], [100])
    mtp = MTP(mean_estimator, density_ratio_estimator, nothing)
    mtpmachk = machine(mtp, data_large, intervention) |> fit!
    output_kliep = ModifiedTreatment.estimate(mtpmachk, intervention)

    density_ratio_oracle = DensityRatioPlugIn(OracleDensityEstimator(dgp_iid))
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
    
    σ2_est = values(σ2(output))
    @test isnothing(σ2_est[1])
    #@test all(σ2_est[2:end] .< moe)

    @test all(isnothing.(values(σ2boot(output))))
    @test all(isnothing.(values(σ2net(output))))

    # TODO: Add better tests to ensure the bootstrap is working correctly
    B = 10
    ModifiedTreatment.bootstrap!(BasicSampler(), output, B)
    @test all(values(σ2boot(output)) .< moe)
end 

#@testset "MTP Network" begin
    Random.seed!(1)
    moe = 1.0

    distseqnet = @dgp(
        L ~ DiscreteUniform(1, 4),
        L2 ~ Binomial(3, 0.4),
        L2s = Sum(:L2, include_self = false),
        L3 ~ Beta(3, 2),
        L4 ~ Poisson(3),
        L4s = Sum(:L4, include_self = false),
        A ~ (@. Normal(0.5 * (:L + :L2) + :L3 + 0.05 * (:L4 + :L2s) + 0.01 * :L4s + 1, 1)),
        As = Sum(:A, include_self = false),
        Y ~ (@. Normal(:A + 0.5 * :As + 0.5 * (:L + :L2) + :L3 + 0.05 * (:L4 + :L2s) + 0.01 * :L4s + 100, 1))
    );

    dgp_net =  DataGeneratingProcess(
        #n -> Graphs.random_regular_graph(n, 4),
        n -> Graphs.erdos_renyi(n, 3/n),
        distseqnet;
        treatment = :A,
        response = :Y,
        controls = [:L, :L2, :L3, :L4]
        )

    n_large = 10000
    data_vlarge = rand(dgp_net, 10^6)
    data_large = rand(dgp_net, n_large)
    intervention = AdditiveShift(0.1)

    truth = compute_true_MTP(dgp_net, data_vlarge, intervention)
    mean_estimator = LinearRegressor()
    #density_ratio_estimator = DensityRatioKLIEP([10.0], [10])
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
    @test !all(within.(values(σ2_est)[4:5] .* n_large, truth.eff_bound, moe))
    
    σ2net_est  = values(σ2net(output))
    @test isnothing(σ2_est[1])
    @test all(within.(values(σ2net_est)[4:5] .* n_large, truth.eff_bound, moe))

    data_large.graph
    A = adjacency_matrix(getgraph(data_large))
    Anew = ((A .+ (A * A)) .> 0)

    data = data_large
    Ysymb = getresponsesymbol(data)
    
    # Organize and transform the data
    Y = getresponse(data)
    intmach = machine(InterventionModel(), data) |> fit!
    LAδs, _ = transform(intmach, intervention)
    LAδsinv, dAδsinv = inverse_transform(intmach, intervention)

    # Compute conditional means of response
    dgp = dgp_net
    Q0bar_noshift = conmean(dgp, data, Ysymb)
    Q0bar_shift = conmean(dgp, CausalTables.replace(LAδs; tbl = merge(gettable(LAδs), (Y = Y,))), Ysymb)

    # Compute conditional density ratio of treatment
    Hn_aux = ones(length(Y))
    for col in keys(dAδsinv)
        g0_Anat= pdf.(condensity(dgp, data, col), Tables.getcolumn(data, col))
        g0_Ainv = pdf.(condensity(dgp, LAδsinv, col), Tables.getcolumn(LAδsinv, col))
        Hn_aux = Hn_aux .* (g0_Ainv ./ g0_Anat)
    end
    Hn_aux = Hn_aux .* prod(dAδsinv)
    

    # Compute the EIF and get g-computation result
    ψnew = mean(Q0bar_shift)
    D = Hn_aux .* (Y .- Q0bar_noshift) .+ (Q0bar_shift .- ψnew)

    if nv(getgraph(data)) == 0 # if graph is empty, just use empirical variance
        eff_bound = var(D)
    else # if graph exists, use estimator from Ogburn 2022
        G = ModifiedTreatment.get_dependency_neighborhood(getgraph(data))
        eff_bound = ModifiedTreatment.cov_unscaled(D, G) / length(D)
    end

    values(σ2_est)[4:5] .* n_large
    values(σ2net_est)[4:5] .* n_large
    truth
    eff_bound
    
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


