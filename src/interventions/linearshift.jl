
# TODO: Allow constructor for constant intervention to take data types other than Float64 (e.g. Int)
mutable struct LinearShift <: Intervention
    δa::Function
    δa_is_constant::Bool
    δb::Function
    δb_is_constant::Bool
    function LinearShift(δa::InterventionParam, δb::InterventionParam)
        δa_constant = δa isa Float64
        δb_constant = δb isa Float64
        δa_input = δa
        δb_input = δb
        if δa_constant
            δa_input = L -> δa
        end
        if δb_constant
            δb_input = L -> δb
        end
        new(δa_input, δa_constant, δb_input, δb_constant)
    end
end

AdditiveShift(δ::InterventionParam) = LinearShift(1.0, δ)
MultiplicativeShift(δ::InterventionParam) = LinearShift(δ, 0.0)

apply_intervention(intervention::LinearShift, A, L) =  A .* intervention.δa(L) .+ intervention.δb(L)
differentiate_intervention(intervention::LinearShift, A, L) = intervention.δa(L)

function inverse(intervention::LinearShift)
    δa_new = intervention.δa_is_constant ? 1.0 ./ intervention.δa(0) : L -> 1.0 ./ intervention.δa(L)
    δb_new = intervention.δa_is_constant && intervention.δb_is_constant ? -intervention.δb(0) ./ intervention.δa(0) : L -> -intervention.δb(L) ./ intervention.δa(L)
    return LinearShift(δa_new, δb_new)
end

function get_induced_intervention(intervention::LinearShift, summary::NeighborSum)
    if intervention.δa_is_constant
        return LinearShift(intervention.δa, L -> Graphs.adjacency_matrix(L.graph) * (ones(nv(L.graph)) .* intervention.δb(L)))
    else
        error("A dynamic multiplicative intervention with a NeighorSum is not invertible, and is therefore not a valid MTP.")
    end
end