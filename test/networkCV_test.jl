using SparseArrays
CausalTables.dependency_matrix(data_net)

X = responseparents(data_net)
y = response(data_net).data.Y

pairs = MLJBase.train_test_pairs(CV(nfolds = 5), 1:DataAPI.nrow(X))

pairs[1]

D = CausalTables.dependency_matrix(X)


fold = D[pairs[1][1], pairs[1][2]]
nonzero_rows = SparseArrays.rowvals(fold)
new_train = pairs[1][1][Not(nonzero_rows)]

foo = map(p -> (p[1][Not(SparseArrays.rowvals(D[p[1], p[2]]))], p[2]), pairs)

length.(getindex.(foo, 1))