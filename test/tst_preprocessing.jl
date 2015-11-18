x = collect(-5:.1:5)
X = [x x.^2 x.^3]'

msg("Test FeatureNormalizer model")

cs = fit(FeatureNormalizer, X)
@test_approx_eq vec(mean(X, 2)) cs.offset
@test_approx_eq vec(std(X, 2)) cs.scale

X4 = predict(cs, X)
@test X4 != X
@test sum(mean(X4, 2)) <= 10e-10
@test_approx_eq std(X4, 2) [1, 1, 1]
