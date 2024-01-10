
InterventionParam = Union{Float64, Function}

abstract type Intervention end

apply_intervention(intervention::Intervention, A, L) = error("Intervention for this combination of intervention and summary is unknown or not implemented.")
apply_inverse_intervention(intervention::Intervention, A, L) = error("Inverse intervention for this combination of intervention and summary is unknown or not implemented.")
differentiate_intervention(intervention::Intervention, A, L) = error("Differentiated intervention for this combination of intervention and summary is unknown or not implemented.")
get_induced_intervention(intervention::Intervention, summary::NetworkSummary) = error("Induced intervention for this combination of intervention and summary is unknown or not implemented.")

mutable struct LinearShift <: Intervention
    δa::Function
    δb::Function
    function LinearShift(δa::InterventionParam, δb::InterventionParam)
        δa_input = δa
        δb_input = δb
        if δa isa Float64
            δa_input = L -> δa
        end
        if δb isa Float64
            δb_input = L -> δb
        end
        new(δa_input, δb_input)
    end
end

AdditiveShift(δ::InterventionParam) = LinearShift(1.0, δ)
MultiplicativeShift(δ::InterventionParam) = LinearShift(δ, 0.0)

apply_intervention(intervention::LinearShift, A, L) =  A .* intervention.δa(L) .+ intervention.δb(L)
apply_inverse_intervention(intervention::LinearShift, A, L) = (A .- intervention.δb(L)) ./ intervention.δa(L)
differentiate_intervention(intervention::LinearShift, A, L) = intervention.δa(L)
differentiate_inverse_intervention(intervention::LinearShift, A, L) = 1 ./ intervention.δa(L)


function get_induced_intervention(intervention::LinearShift, summary::NeighborSum)
    return LinearShift(intervention.δa, L -> Graphs.adjacency_matrix(L.graph) * (ones(nv(L.graph)) .* intervention.δb(L)))
end







