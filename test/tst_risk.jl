
# ==========================================================================

msg("Test center, rescale and center_rescale")

X, y = RegressionData.polynome(-10:.2:20, [4, 3, 2, 1], noise=.5)

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

msg("Test linear regression on noisy line")

w = [1, 10]
X, y = RegressionData.polynome(-10:.2:10, w, noise=.5)
x = vec(X[end,:])

# Set hyper parameters
θ = randn(length(w))
α = 0.005
maxIter = 1000

loss = LossFunctions.L2DistLoss()
pred = LinearPredictor(bias = 1)
risk = RiskModel(pred, loss)
ŷ = pred(X, θ)
▽ = zeros(length(w), 1)

for i = 1:maxIter
    value!(ŷ, risk, X, θ, y)
    grad!(▽, risk, X, θ, y, ŷ)
    θ = θ - α .* vec(▽)
end

@test sumabs(w - θ) < .1
mp = scatterplot(x, y, color = :blue, height = 5)
lineplot!(mp, x, vec(value(pred, X, θ)), color = :red)
print(mp)
