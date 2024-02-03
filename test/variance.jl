using Distributions
using LinearAlgebra
using Graphs


# Generate a graph to simulate a correlation matrix
n_samples = 500
k = 5
g = Graphs.stochastic_block_model(50.0, 0.01, fill(n ÷ k, k))
G = adjacency_matrix(g)

# Draw samples
n_simulations = 10000
L = rand(Normal(1, 1), n_samples, n_simulations)
Ls = G * L

# Suppose we want to estimate the sample mean of Ls.
# They are correlated, and we don't know what the correlation is.
# However, we know how that correlation arises, and we can simulate it.

# Simulate the true mean
μ = mean(Ls, dims = 1)
mean(μ)

# compared to resampled estimate which is much closer...
var(μ)

# ...the sample variance is much lower than expected!
σ2 = var(Ls, dims = 1) ./ n_samples
mean(σ2)

# But the problem is we can't resample many datasets... unless we bootstrap
# What if we use multiplier bootstrap?
N_bootstraps = 10000
w = rand(Normal(1, 1), n_samples, N_bootstraps)
μb = mean(G * (L[:, 1] .* w), dims = 1)
var(μb)

# Compute bootstrapped means


# Compute bootstrapped variances
σ2b = var(μb, dims = 1)





# but our multiplier bootstrap estimate gets much closer




