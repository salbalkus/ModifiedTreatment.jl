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

using Random


# function for testing approximate equality of statistical estimators
within(x, truth, ϵ) = abs(x - truth) < ϵ

distseqiid = Vector{Pair{Symbol, CausalTables.ValidDGPTypes}}([
    :L1 => (; O...) -> DiscreteUniform(1, 5),
    :A => (; O...) -> (@. Normal(O[:L1], 1)),
    :Y => (; O...) -> (@. Normal(O[:A] + 0.5 * O[:L1] + 10, 0.5))
])
dgp_iid = DataGeneratingProcess(distseqiid, :A, :Y, [:L1]);
data_iid = rand(dgp_iid, 100)


distseqnet = Vector{Pair{Symbol, CausalTables.ValidDGPTypes}}([
        :L1 => (; O...) -> DiscreteUniform(1, 5),
        :L1_s => NeighborSum(:L1),
        :A => (; O...) -> (@. Normal(O[:L1] + 0.1 * O[:L1_s], 0.5)),
        :A_s => NeighborSum(:A),
        :Y => (; O...) -> (@. Normal(O[:A] + 0.1 * O[:A_s] + 0.2 * O[:L1] + 10, 1))
    ])

dgp = DataGeneratingProcess(n -> erdos_renyi(n, 3/n), distseqnet; 
                            treatment = :A, response = :Y, controls = [:L1]);
data_net = rand(dgp, 100)


@testset "Intervention" begin
    int1 = AdditiveShift(0.5)
    int2 = MultiplicativeShift(1.5)
    int3 = LinearShift(1.5, 0.5)

    L = CausalTables.getcontrols(data_net)
    A = CausalTables.gettreatment(data_net)

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

    @test apply_intervention(inducedint, A, L) == A .* 1.5 .+ adjacency_matrix(data_net.graph) * (ones(nv(data_net.graph)) .* 0.5)
    @test apply_inverse_intervention(inducedint, A, L) == (A .- adjacency_matrix(data_net.graph) * (ones(nv(data_net.graph)) .* 0.5)) ./ 1.5
    @test all(differentiate_intervention(inducedint, A, L) .== 1.5)
    @test all(differentiate_inverse_intervention(inducedint, A, L) .== 1/1.5)
end

@testset "InterventionModel" begin
    intervention = LinearShift(1.5, 0.5)
    intmach = machine(InterventionModel(), data_net) |> fit!
    LAs, Ls, As = predict(intmach, intervention)  

    @test Ls.tbl == (L1 = data_net.tbl.L1, L1_s = data_net.tbl.L1_s)
    @test As == (A_s = data_net.tbl.A_s, A = data_net.tbl.A)
    @test LAs.tbl == (L1 = data_net.tbl.L1, L1_s = data_net.tbl.L1_s, A = data_net.tbl.A, A_s = data_net.tbl.A_s)

    LAδs, dAδs = transform(intmach, intervention)
    @test LAδs.tbl.A ≈ data_net.tbl.A .* 1.5 .+ 0.5
    @test LAδs.tbl.A_s ≈ adjacency_matrix(data_net.graph) * ((data_net.tbl.A .* 1.5) .+ 0.5)
    @test dAδs.A == 1.5

    LAδsinv, dAδsinv = inverse_transform(intmach, intervention)
    @test LAδsinv.tbl.A ≈ (data_net.tbl.A .- 0.5) ./ 1.5
    @test LAδsinv.tbl.A_s ≈ adjacency_matrix(data_net.graph) * ((data_net.tbl.A .- 0.5) ./ 1.5)
    @test dAδsinv.A == 1/1.5
end

@testset "CrossFitModel" begin
    
    # Test a regression model
    LA = replacetable(data_iid, TableOperations.select(data_iid, :L1, :A) |> Tables.columntable)    
    Y = Tables.getcolumn(LA, :A) .+ 0.5 .* Tables.getcolumn(LA, :L1) .+ 10
    mean_estimator = MLJLinearModels.LinearRegressor()
    mean_crossfit = CrossFitModel(mean_estimator, CV())
    mach_mean = machine(mean_crossfit, LA, Y) |> fit!
    pred_mean = MLJBase.predict(mach_mean, LA)
    @test cor(Y, pred_mean) == 1.0

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
    @test MLJBase.predict(mach_ratio, LA, LAδ) ==  g0 ./ g0shift
    
end


#@testset "MTP IID" begin
    Random.seed!(1)
    
    data_large = rand(dgp_iid, 10^5)

    intervention = LinearShift(1.1, 0.5)
    truth = compute_true_MTP(dgp_iid, data_large, intervention)

    moe = 0.1

    mean_estimator = LinearRegressor()
    density_ratio_estimator = DensityRatioPropensity(OracleDensityEstimator(dgp_iid))
    boot_sampler = BasicSampler(10)
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

estimators = (IPW(), OneStep(), TMLE())
δ = source(IdentityIntervention())
O = data_iid
Os = source(O)

    Y = getresponse(Os)
    model_intervention = InterventionModel()
    LAs, Ls, As, LAδs, dAδs, LAδsinv, dAδsinv = intervene_on_data(model_intervention, Os, δ)
    mach_mean, mach_density = crossfit_nuisance_estimators(mtp, Y, LAs, Ls, As)

    Qn, Qδn, Hn, Hshiftn = estimate_nuisances(mach_mean, mach_density, LAs, LAδs, LAδsinv, dAδs, dAδsinv)
    outcome_regression_est, ipw_est, onestep_est, tmle_est = estimate_causal_parameters(estimators, Y, Qn, Qδn, Hn, Hshiftn)


    # Conservative EIF variance estimate
    #mach_consvar = machine(MTP.EIFConservative(), Y, Qn, gdf_source) |> fit!
    #consvar = MMI.transform(mach_consvar, Qδn, Hn)
    #onestep = merge(estimates[3], consvar)
    #tmle = merge(estimates[4], consvar)

    Os_b = bootstrap_samples(mtp.boot_sampler, Os)
    Y = node(Os_b -> Iterators.map(getresponse, Os_b), Os_b)
    model_intervention = ResampledModel(InterventionModel())
    foo = Os_b()
    mach_intervention = machine(model_intervention, foo) |> fit!
    bar = δ()

    tmp1 = predict(mach_intervention, δ)
    tmp2 = transform(mach_intervention, δ)
    tmp3 = inverse_transform(mach_intervention, δ)

    LAs, Ls, As = tmp1[1], tmp1[2], tmp1[3]
    LAδs, dAδs = tmp2[1], tmp2[2]
    LAδsinv, dAδsinv = tmp3[1], tmp3[2]
    
    mach_mean = machine(mtp.mean_estimator, first(LAs), first(Y)) |> fit!
    mach_density = machine(mtp.density_ratio_estimator, first(Ls), first(As)) |> fit!

    Qn, Qδn, Hn, Hshiftn = resample_nuisances(mach_mean, mach_density, LAs, LAδs, LAδsinv, dAδs, dAδsinv)
    mach_ipw = machine(ResampledModel(estimators[1]), Y) |> fit!
    foo = transform(mach_ipw, Hn)
    collect(foo())
    mach = first(mach_ipw.fitresult.machines)
    transform(mach, Hn)()
    foo = Iterators.map(x -> transform(x...), zip(mach_ipw.fitresult.machines, Hn()))
    collect(zip(mach_ipw.fitresult.machines, Hn()))
    
    
    first(foo)
    ipw_est = transform(mach_ipw, Hn)
    ψ = [sum(h .* y) / sum(h) for (h, y) in zip(Hn(), Y())]
    
