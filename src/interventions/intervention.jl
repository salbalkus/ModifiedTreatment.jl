
InterventionParam = Union{Float64, Function}

abstract type Intervention end

apply_intervention(intervention::Intervention, A, L) = error("Intervention for this combination of intervention and summary is unknown or not implemented.")
apply_inverse_intervention(intervention::Intervention, A, L) = error("Inverse intervention for this combination of intervention and summary is unknown or not implemented.")
differentiate_intervention(intervention::Intervention, A, L) = error("Differentiated intervention for this combination of intervention and summary is unknown or not implemented.")
get_induced_intervention(intervention::Intervention, summary::NetworkSummary) = error("Induced intervention for this combination of intervention and summary is unknown or not implemented.")

mutable struct IdentityIntervention <: Intervention end

apply_intervention(intervention::IdentityIntervention, A, L) =  A
apply_inverse_intervention(intervention::IdentityIntervention, A, L) = A
differentiate_intervention(intervention::IdentityIntervention, A, L) = 1
differentiate_inverse_intervention(intervention::IdentityIntervention, A, L) = 1
get_induced_intervention(intervention::IdentityIntervention, summary::NeighborSum) = IdentityIntervention()







