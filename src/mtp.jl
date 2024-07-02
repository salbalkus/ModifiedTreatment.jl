mutable struct MTP <: UnsupervisedNetworkComposite
    mean_estimator::MMI.Supervised
    density_ratio_estimator::Condensity.ConDensityRatioEstimator
    cv_splitter
    confidence::Float64
end


MTP(mean_estimator, density_ratio_estimator, cv_splitter) = MTP(mean_estimator, density_ratio_estimator, cv_splitter, 0.95)

function MLJBase.prefit(mtp::MTP, verbosity, O::CausalTable, Δ::Intervention)

    δ = source(Δ)
    Os = source(O)
    Y = _get_response_vector(Os)
    G = CausalTables.dependency_matrix(Os)

    model_intervention = InterventionModel()
    LAs, Ls, As, LAδs, dAδs, LAδsinv, dAδsinv = intervene_on_data(model_intervention, Os, δ)
    
    # Fit and estimate nuisance parameters
    mach_mean, mach_density = crossfit_nuisance_estimators(mtp, Y, LAs, LAδsinv, Ls, As)
    Qn, Qδn, Hn, Hshiftn = estimate_nuisances(mach_mean, mach_density, LAs, LAδs, LAδsinv, dAδs, dAδsinv)

    # Get causal estimates
    plugin_est, ipw_est, sipw_est, onestep_est, tmle_est = estimate_causal_parameters(Y, G, Qn, Qδn, Hn, Hshiftn)

    return (; 
        plugin = plugin_est,
        ipw = ipw_est,
        sipw = sipw_est,
        onestep = onestep_est,
        tmle = tmle_est,
        report = (; 
            LAs = LAs,
            LAδs = LAδs,
            LAδsinv = LAδsinv,
            dAδs = dAδs,
            dAδsinv = dAδsinv,
            Qn = Qn,
            Qδn = Qδn,
            Hn = Hn,
            Hshiftn = Hshiftn,
        ),
        nuisance_machines = (;
            machine_mean = mach_mean,
            machine_density = mach_density,
        )
    )    

end

# Define custom functions akin to `predict` that yield the result of each estimation strategy from a learning network machine
plugin(machine, δnew::Intervention) = MTPResult(MLJBase.unwrap(machine.fitresult).plugin(δnew), machine, δnew)
ipw(machine, δnew::Intervention) = MTPResult(MLJBase.unwrap(machine.fitresult).ipw(δnew), machine, δnew)
onestep(machine, δnew::Intervention) = MTPResult(MLJBase.unwrap(machine.fitresult).onestep(δnew), machine, δnew)
tmle(machine, δnew::Intervention) = MTPResult(MLJBase.unwrap(machine.fitresult).tmle(δnew), machine, δnew)

estimate(machine::Machine, δnew::Intervention) = MTPResult(
    (plugin = MLJBase.unwrap(machine.fitresult).plugin(δnew),
     ipw = MLJBase.unwrap(machine.fitresult).ipw(δnew),
     sipw = MLJBase.unwrap(machine.fitresult).sipw(δnew),
     onestep = MLJBase.unwrap(machine.fitresult).onestep(δnew),
     tmle = MLJBase.unwrap(machine.fitresult).tmle(δnew)
    ),
    machine,
    δnew
)

# Define custom function to extract the nuisance estimators from the learning network machine
nuisance_machines(machine::Machine{MTP}) = MLJBase.unwrap(machine.fitresult).nuisance_machines

function _get_response_vector(Os::CausalTable)
    Y_table = CausalTables.response(Os)
    Y = DataAPI.ncol(Y_table) == 1 ? Tables.getcolumn(Y_table, 1) : throw(ArgumentError("Provided table with columns $(Tables.columnnames(Y_table)) has more than one column, when only a single column is supported."))
    return Y
end
_get_response_vector(Os::AbstractNode) = node(_get_response_vector, Os)

function get_dependency_neighborhood(g::AbstractGraphOrNothing)
    # Only compute if a graph is passed in
    if isnothing(g)
        return nothing
    end
    
    A = adjacency_matrix(g)

    # get the nodes within two-hops of the adjacency matrix,
    # add them to the original,
    # and return edge weights to 1
    Anew = ((A .+ (A * A)) .> 0)
    Anew[diagind(Anew)] .= 1

    # directly return the adjacency matrix
    # WARNING: If a graph is constructed from this output, the 1-diagonal will be converted to a 0-diagonal
    # and subsequently matrix multiplications may be incorrect if a 1-diagonal was assumed
    return Anew
end
get_dependency_neighborhood(g::Node) = node(g -> get_dependency_neighborhood(g), g)

function intervene_on_data(model_intervention, Os, δ)
    mach_intervention = machine(model_intervention, Os)

    tmp1 = MMI.predict(mach_intervention, δ)
    tmp2 = MMI.transform(mach_intervention, δ)
    tmp3 = MMI.inverse_transform(mach_intervention, δ)

    LAs, Ls, As = tmp1[1], tmp1[2], tmp1[3]
    LAδs, dAδs = tmp2[1], tmp2[2]
    LAδsinv, dAδsinv = tmp3[1], tmp3[2]

    return LAs, Ls, As, LAδs, dAδs, LAδsinv, dAδsinv
end

function crossfit_nuisance_estimators(mtp, Y, LAs, LAδsinv, Ls, As)

    # If the density ratio estimator is adaptive, we need to ensure multiple estimators are fit for each factorized component
    # (Otherwise, this is handled automatically by fixed density ratio estimators)
    ratio_model_type = typeof(mtp.density_ratio_estimator)
    if ratio_model_type <: Condensity.ConDensityRatioEstimatorAdaptive
        dprmodel = DecomposedPropensityRatio(mtp.density_ratio_estimator)
    elseif ratio_model_type <: Condensity.ConDensityRatioEstimatorFixed
        dprmodel = mtp.density_ratio_estimator
    else
        throw(ArgumentError("Unrecognized density ratio estimator type. Density ratio estimator must be of type CondensityRatioEstimatorAdaptive or CondensityRatioEstimatorFixed."))
    end

    # Decide whether to cross-fit the models
    if isnothing(mtp.cv_splitter)
        mean_model = mtp.mean_estimator
        dr_model = dprmodel
    else
        mean_model = CrossFitModel(mtp.mean_estimator, mtp.cv_splitter)
        dr_model = CrossFitModel(dprmodel, mtp.cv_splitter)
    end

    # Construct machines bound to appropriate data
    mach_mean = machine(mtp.mean_estimator, LAs, Y)
    if ratio_model_type <: Condensity.ConDensityRatioEstimatorAdaptive
        mach_density = machine(dr_model, Ls, As)
    else # if ratio_model_type <: Condensity.ConDensityRatioEstimatorFixed
        mach_density = machine(dr_model, LAδsinv, LAs)
    end

    return mach_mean, mach_density
end

function estimate_nuisances(mach_mean, mach_density, LAs, LAδs, LAδsinv, dAδs, dAδsinv)
    # Get Conditional Mean
    Qn = MMI.predict(mach_mean, LAs)
    Qδn = MMI.predict(mach_mean, LAδs)

    # Get Density Ratio
    Hn = MMI.predict(mach_density, LAδsinv, LAs) * prod(dAδsinv)

    # TODO: Check if we actually don't need to multiply by a derivative here
    Hshiftn = MMI.predict(mach_density, LAs, LAδs)

    return Qn, Qδn, Hn, Hshiftn
end

function estimate_causal_parameters(Y, G, Qn, Qδn, Hn, Hshiftn)
    mach_ipw = machine(IPW(), Y, G) |> fit!
    mach_onestep = machine(OneStep(), Y, Qn, G) |> fit!
    mach_tmle = machine(TMLE(), Y, Qn, G) |> fit!

    plugin_est = plugin_transform(Qδn)
    ipw_est = node(Hn -> MMI.transform(mach_ipw, Hn, false), Hn)
    sipw_est =  node(Hn -> MMI.transform(mach_ipw, Hn, true), Hn)
    onestep_est = MMI.transform(mach_onestep, Qδn, Hn)
    tmle_est = MMI.transform(mach_tmle, Qδn, Hn, Hshiftn)

    return plugin_est, ipw_est, sipw_est, onestep_est, tmle_est
end

