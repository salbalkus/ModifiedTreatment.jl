
# Overload CausalTable functions for nodes
CausalTables.treatment(O::AbstractNode) = node(CausalTables.treatment, O)
CausalTables.response(O::AbstractNode) = node(CausalTables.response, O)
CausalTables.confounders(O::AbstractNode) = node(CausalTables.confounders, O)
CausalTables.summarize(O::AbstractNode) = node(CausalTables.summarize, O)
CausalTables.replace(O::AbstractNode; data::AbstractNode = nothing) = node((x, data) -> CausalTables.replacetable(x; data = data), O, data)
CausalTables.data(O::AbstractNode) = node(CausalTables.data, O)
CausalTables.adjacency_matrix(O::AbstractNode) = node(CausalTables.adjacency_matrix, O)
CausalTables.dependency_matrix(O::AbstractNode) = node(CausalTables.dependency_matrix, O)

# Overload the merge function for nodes
Base.merge(nt1::AbstractNode, nt2::AbstractNode) = node((nt1, nt2) -> merge(nt1, nt2), nt1, nt2)

# Overload getindex function for nodes
Base.getindex(nt::AbstractNode, i::Int64) = node(nt -> getindex(nt, i), nt)
Base.iterate(nt::AbstractNode) = node(nt -> iterate(nt), nt)
Base.prod(nt::AbstractNode) = node(nt -> prod(nt), nt)
#Base.length(nt::AbstractNode) = node(nt -> length(nt), nt)





