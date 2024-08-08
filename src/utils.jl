const UNIT_LOWER_BOUND = 0.005
const UNIT_UPPER_BOUND = 0.995

# The logistic function
qlogis(p) = log(p / (1 - p))

function bound(X::Vector; lower = -Inf, upper = Inf)
    X = copy(X)
    X[X .> upper] .= upper
    X[X .< lower] .= lower
    return X
end

function bound!(X::Vector; lower = -Inf, upper = Inf)
    X[X .> upper] .= upper
    X[X .< lower] .= lower
end

cov_unscaled(x::AbstractArray, G::AbstractMatrix) = (transpose(x) * G * x)

function neighbor_center(X::AbstractArray, G::AbstractMatrix)
    k_neighbors = Int.(G * ones(length(X)))
    k_neighbors_unique = unique(k_neighbors)
    ψ_by_k = Dict(k_neighbors_unique .=> [mean(X[k_neighbors .== k]) for k in k_neighbors_unique])
    return([ψ_by_k[k] for k in k_neighbors])
end

function network_variance(D::AbstractArray, G::AbstractMatrix)
    Dc = D .- neighbor_center(D, G)
    return(cov_unscaled(Dc, G) / (length(D)))
end


