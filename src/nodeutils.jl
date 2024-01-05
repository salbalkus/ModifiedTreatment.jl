
# Overload CausalTable functions for nodes
gettreatment(O::AbstractNode) = node(CausalTables.gettreatment, O)
getresponse(O::AbstractNode) = node(CausalTables.gettarget, O)
getgraph(O::AbstractNode) = node(CausalTables.getgraph, O)
getsummaries(O::AbstractNode) = node(CausalTables.getsummaries, O)
summarize(O::AbstractNode) = node(CausalTables.summarize, O)

# Overload the merge function for nodes
merge(nt1::AbstractNode, nt2::AbstractNode) = node((nt1, nt2) -> merge(nt1, nt2), nt1, nt2)



