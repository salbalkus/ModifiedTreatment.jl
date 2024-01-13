
# Overload CausalTable functions for nodes
CausalTables.gettreatment(O::AbstractNode) = node(CausalTables.gettreatment, O)
CausalTables.getresponse(O::AbstractNode) = node(CausalTables.getresponse, O)
CausalTables.getgraph(O::AbstractNode) = node(CausalTables.getgraph, O)
CausalTables.getsummaries(O::AbstractNode) = node(CausalTables.getsummaries, O)
CausalTables.summarize(O::AbstractNode) = node(CausalTables.summarize, O)

# Overload the merge function for nodes
Base.merge(nt1::AbstractNode, nt2::AbstractNode) = node((nt1, nt2) -> merge(nt1, nt2), nt1, nt2)

# Overload getindex function for nodes
Base.getindex(nt::AbstractNode, i::Int64) = node(nt -> getindex(nt, i), nt)
Base.iterate(nt::AbstractNode) = node(nt -> iterate(nt), nt)
Base.prod(nt::AbstractNode) = node(nt -> prod(nt), nt)
#Base.length(nt::AbstractNode) = node(nt -> length(nt), nt)


