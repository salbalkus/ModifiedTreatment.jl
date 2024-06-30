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

cov_unscaled(x::Vector, G::AbstractMatrix) = (transpose(x) * G * x)

table_to_vector(X) = DataAPI.ncol(X) == 1 ? Tables.getcolumn(X, 1) : throw(ArgumentError("Provided table has more than one column. This function is only used to convert a single column table to a vector."))



