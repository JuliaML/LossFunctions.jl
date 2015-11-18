# ==========================================================================

msg("Test linear regression on noisy line")

w = [1, 10]
x, y = DataGenerators.noisy_poly(w, -10:.2:10, noise = .5)
X = x'

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
x, y = DataGenerators.noisy_sin(100; noise = .1)
X = DataUtils.expand_poly(x, degree = 4)

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
