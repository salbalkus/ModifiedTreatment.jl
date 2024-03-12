using MLJ
using TableOperations
using Distributions
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree


A = rand(DiscreteUniform(1,4), 100)
B = rand(Normal(), 100)
C = A .+ B .+ rand(Normal(), 100)
data = (A = A, B = B, C = C)

X = TableOperations.select(data, :A, :B) |> Tables.columntable
y = data.C
typeof(X)
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree

mach = machine(DecisionTreeRegressor(), X, y) |> fit!
evaluate(DecisionTreeRegressor(), X, y, resampling=CV(nfolds = 5), measure=rms)
