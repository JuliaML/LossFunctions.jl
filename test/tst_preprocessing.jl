X, y = RegressionData.polynome(-10:.2:20, [4, 3, 2, 1], noise=.5)

# ==========================================================================

msg("Test center, rescale and center_rescale")

X1 = copy(X)
center!(X1)
@test sum(mean(X1, 2)) <= 10e-10

X2 = copy(X)
rescale!(X2)
@test_approx_eq std(X2, 2) [1, 1, 1]

X3 = copy(X)
center_rescale!(X3)
@test sum(mean(X3, 2)) <= 10e-10
@test_approx_eq std(X3, 2) [1, 1, 1]

# ==========================================================================

msg("Test CenterScale model")

cs = fit(CenterRescale, X)
@test_approx_eq vec(mean(X, 2)) cs.mean
@test_approx_eq vec(std(X, 2)) cs.std

X4 = predict(cs, X)
@test X4 != X
@test sum(mean(X4, 2)) <= 10e-10
@test_approx_eq std(X4, 2) [1, 1, 1]
