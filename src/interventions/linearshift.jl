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
differentiate_intervention(intervention::LinearShift, A, L) = intervention.δa(L)
inverse(intervention::LinearShift) = LinearShift(L -> 1.0 ./ intervention.δa(L), L -> -intervention.δb(L))

function get_induced_intervention(intervention::LinearShift, summary::NeighborSum)
    return LinearShift(intervention.δa, L -> Graphs.adjacency_matrix(L.graph) * (ones(nv(L.graph)) .* intervention.δb(L)))
end