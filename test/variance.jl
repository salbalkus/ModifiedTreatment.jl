using Distributions
using LinearAlgebra

n = 1000

v = rand(Exponential(0.5), n, n)
V = v'v ./ (n)

μ = 1
ε = rand(MvNormal(fill(μ, n), V))

X = rand(Normal(5, 1), n)

Y = X .+ 5 .+ ε

using GLM

data = (X = X, Y = Y)
model = lm(@formula(Y ~ X), data)
adjr2(model)