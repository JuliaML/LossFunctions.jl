# ==========================================================================

msg("Test linear regression on noisy line")

w = [1, 10]
X, y = RegressionData.polynome(w, -10:.2:10, noise=.5)
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

@test sumabs(w - θ) < .15
mp = scatterplot(x, y, color = :blue, height = 5, margin = 5)
lineplot!(mp, x, vec(value(pred, X, θ)), color = :red)
print(mp)

# ==========================================================================

msg("Test linear regression on sin using poly basis expansion")

k = 5
x, y = RegressionData.example_sin(0:.1:2π; noise = .1)
X, _ = RegressionData.polynome(ones(k), x, noise = 0.)
center_rescale!(X)

# Set hyper parameters
θ = zeros(k)
α = 0.2
maxIter = 5000

loss = LossFunctions.L2DistLoss()
pred = LinearPredictor(bias = 1)
risk = RiskModel(pred, loss)
ŷ = pred(X, θ)
▽ = zeros(k, 1)

for i = 1:maxIter
    value!(ŷ, risk, X, θ, y)
    grad!(▽, risk, X, θ, y, ŷ)
    θ = θ - α .* vec(▽)
end

J = value!(ŷ, risk, X, θ, y)
@test J < 0.05
mp = scatterplot(x, y, color = :blue, height = 5)
lineplot!(mp, x, vec(value(pred, X, θ)), color = :red)
print(mp)
