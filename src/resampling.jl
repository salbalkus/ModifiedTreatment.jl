
# Model to handle bootstrapped lazy outputs
mutable struct ResampledModel{T <: MMI.Model} <: MMI.Model
    model::T
end

function MMI.fit(rse::ResampledModel, verbosity, X...)

    # Here X should be a tuple of equal-length vectors
    # Each vector entry is one "sample" of the data for that particular parameter

    # Fit each sample
    machines = Iterators.map(x -> fit!(machine(rse.model, x...)), zip(X...))

    fitresult = (; machines = machines)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

# Predict on Generators
MMI.transform(rse::ResampledModel, fitresult, X::Base.Generator...) = Iterators.map(x -> MMI.transform(x...), zip(fitresult.machines, X...))

# Special case of InterventionModel
function MMI.predict(rse::ResampledModel{InterventionModel}, fitresult, δ::Intervention)
    result = Iterators.map(mach -> MMI.predict(mach, δ), fitresult.machines)
    return Iterators.map(r -> getindex(r, 1), result), Iterators.map(r -> getindex(r, 2), result), Iterators.map(r -> getindex(r, 3), result)
end
function MMI.transform(rse::ResampledModel{InterventionModel}, fitresult, δ::Intervention)
    result = Iterators.map(mach -> MMI.transform(mach, δ), fitresult.machines)
    return Iterators.map(r -> getindex(r, 1), result), Iterators.map(r -> getindex(r, 2), result)
end
function MMI.inverse_transform(rse::ResampledModel{InterventionModel}, fitresult, δ::Intervention)
    result = Iterators.map(mach -> MMI.inverse_transform(mach, δ), fitresult.machines)
    return Iterators.map(r -> getindex(r, 1), result), Iterators.map(r -> getindex(r, 2), result)
end


# Bootstrapping
abstract type BootstrapSampler end

bootstrap_samples(sampler::BootstrapSampler, O::AbstractNode) = node(O -> bootstrap_samples(sampler, O), O)
bootstrap_samples(sampler::BootstrapSampler, O::CausalTable) = Iterators.map(i -> bootstrap_sample(sampler, O), 1:sampler.B)

mutable struct BasicSampler <: BootstrapSampler
    B::Int64
end
bootstrap_sample(sampler::BasicSampler, O::CausalTable) = Tables.subset(O, rand(1:DataAPI.nrow(O), DataAPI.nrow(O)))

######

mutable struct ClusterSampler <: BootstrapSampler
    B
    K # size of each cluster. Must be identical across clusters
end
function bootstrap_sample(sampler::ClusterSampler, O::CausalTable)
    n_clusters = nrow(gdf.graph_data) ÷ sampler.K
    samples = [vcat(c, neighbors(getgraph(O), c)) for c in sample(1:DataAPI.nrow(O), n_clusters)]
    return cluster_index(O, samples)
end

# TODO: How do we know the dataframe row number matches up with the vertex number?
function cluster_index(O, s)
    
    # Determine which nodes are duplicated in the sampling
    Ic = countmap(s)
    Ivalues = collect(values(Ic))
    Ikeys = collect(keys(Ic))

    # track the order in which indices are added to the new graph
    I_new = copy(Ikeys)

    # Construct a graph of the nodes that were actually sampled
    g_new = getgraph(O)[I_new]
   
    # Add the duplicated nodes
    for val in 2:maximum(Ivalues)
        for _ in 1:(val-1)
            Ikeys_next = Ikeys[Ivalues .== val]
            I_new = vcat(I_new, Ikeys_next)
            g_new = blockdiag(g_new, gdf.graph[Ikeys_next])
        end
    end
    tbl_new = Tables.subset(O.tbl, I_new)
    return CausalTables.replace(O; tbl = tbl_new, graph = g_new)
end

### Vertex samplers

mutable struct VertexMooNSampler <: BootstrapSampler
    B
    frac
end
function bootstrap_sample(sampler::VertexMooNSampler, O::CausalTable)
    s = sample(1:nv(gdf), Int(DataAPI.nrow(O) * sampler.frac), replace = false)
    return CausalTables.replace(O; tbl = Tables.subset(O.tbl, s), graph = getgraph(O)[s])
end

mutable struct VertexSampler <: BootstrapSampler 
    B
end
bootstrap_sample(sampler::VertexSampler, O::CausalTable) = vertex_index(O, rand(1:DataAPI.nrow(O), DataAPI.nrow(O)))

function vertex_index(O::CausalTable, samp::Vector{Int})
    # First, filter the table
    g = getgraph(O)
    tbl_new = Tables.subset(gettable(O), rand(1:DataAPI.nrow(O), DataAPI.nrow(O)))

    # Compute the graph density for adding random edges
    g_density = (ne(g) / nv(g)^2 )

    # Construct a transformation of the adjacency matrix
    S = sparse(I, nv(g), nv(g))[:, samp]

    # Perform Snijders-Borgatti sampling sans random edges between duplicates
    A = transpose(S) * adjacency_matrix(g) * S

    # Add the random edges between duplicates
    duplicates = unique([findall(samp .== s) for s in samp])
    potential_edges = reduce(vcat, collect.(combinations.(duplicates[length.(duplicates) .> 1], 2)))
    for pair in potential_edges
        # random add an edge between duplicated vertices based on the number of edges in the original graph
        # TODO: Enable edge weights
        if rand() < g_density
            A[pair[1], pair[2]] = 1
            A[pair[2], pair[1]] = 1
        end
    end 
        
    return CausalTables.replace(O; tbl = tbl_new, graph = Graph(A))
end

    