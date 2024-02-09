# Bootstrapping
abstract type BootstrapSampler end

bootstrap_samples(sampler::BootstrapSampler, O::AbstractNode) = node(O -> bootstrap_samples(sampler, O), O)
bootstrap_samples(sampler::BootstrapSampler, O::CausalTable) = Iterators.map(i -> bootstrap(sampler, O), 1:sampler.B)

mutable struct BasicSampler <: BootstrapSampler end
bootstrap(::BasicSampler, O::CausalTable) = Tables.subset(O, rand(1:DataAPI.nrow(O), DataAPI.nrow(O)))

######

mutable struct ClusterSampler <: BootstrapSampler
    K::Int # size of each cluster. Must be identical across clusters
    n_clusters::Union{Int, Nothing}
    graph::Union{Graph, Nothing}
    ClusterSampler(K) = new(K, nothing, nothing)
end

# Function to update the graph in the ClusterSampler to fit a new sample size
function updategraph!(cs::ClusterSampler, O)
    cs.n_clusters = DataAPI.nrow(O) ÷ cs.K
    # Since we're indexing connected observations adjacently,
    # we know that the adjacency matrix will be block diagonal
    block = ones(cs.K, cs.K)
    block[diagind(block)] .= 0
    block = sparse(block)
    cs.graph = Graph(blockdiag((block for i in 1:cs.n_clusters)...))
end

function bootstrap(bootstrapsampler::ClusterSampler, O::CausalTable)

    # Get the underlying graph to construct the resample
    if isnothing(bootstrapsampler.graph) || nv(bootstrapsampler.graph) != DataAPI.nrow(O)
        updategraph!(bootstrapsampler, O)
    end

    # Sample the clusters
    g = getgraph(O)
    samples = reduce(vcat, [vcat(c, neighbors(g, c)) for c in sample(1:DataAPI.nrow(O), bootstrapsampler.n_clusters)])

    # subset the table and combine with the graph
    return CausalTables.replace(O; tbl = Tables.subset(O.tbl, samples), graph = bootstrapsampler.graph)
end

### Vertex samplers

mutable struct VertexMooNSampler <: BootstrapSampler
    frac
end
function bootstrap(bootstrapsampler::VertexMooNSampler, O::CausalTable)
    s = sample(1:nv(gdf), Int(DataAPI.nrow(O) * bootstrapsampler.frac), replace = false)
    return CausalTables.replace(O; tbl = Tables.subset(O.tbl, s), graph = getgraph(O)[s])
end

mutable struct VertexSampler <: BootstrapSampler end
bootstrap(::VertexSampler, O::CausalTable) = vertex_index(O, rand(1:DataAPI.nrow(O), DataAPI.nrow(O)))

function vertex_index(O::CausalTable, samp::Vector{Int})
    # First, filter the table
    g = getgraph(O)
    tbl_new = Tables.subset(gettable(O), samp)

    # Compute the graph density for adding random edges
    g_density = (ne(g) / nv(g)^2 )

    # Perform Snijders-Borgatti sampling sans random edges between duplicates
    A = adjacency_matrix(g)[samp, samp]

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

    