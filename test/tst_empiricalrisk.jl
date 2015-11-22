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
risk = EmpiricalRisk(pred, loss)
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
DataUtils.normalize!(X)

# Set hyper parameters
θ = zeros(k)
α = 0.2
maxIter = 5000

loss = LossFunctions.L2DistLoss()
pred = LinearPredictor(bias = 1)
risk = EmpiricalRisk(pred, loss)
ŷ = pred(X, θ)
▽ = zeros(k, 1)

for i = 1:maxIter
    value!(ŷ, risk, X, θ, y)
    grad!(▽, risk, X, θ, y, ŷ)
    θ = θ - α .* vec(▽)
end

J = value!(ŷ, risk, X, θ, y)
@test J < 0.04
mp = scatterplot(x, y, color = :blue, height = 5)
lineplot!(mp, x, vec(value(pred, X, θ)), color = :red)
print(mp)

# ==========================================================================

msg("Test linear regression with L2 pen on sin using poly basis expansion")

k = 14
x, y = load_sin()
X = DataUtils.expand_poly(x, degree = k)
o, s = DataUtils.normalize!(X)

x1 = collect(0:.01:2π)
X1 = DataUtils.expand_poly(x1, degree = k)
DataUtils.normalize!(X1, o, s)

θ = zeros(k)
α = 0.05
maxIter = 5000

loss = LossFunctions.L2DistLoss()
pred = LinearPredictor(bias = 0)
pen = Penalties.L2Penalty(0.05)
risk = EmpiricalRisk(pred, loss, pen)
ŷ = pred(X, θ)
▽ = zeros(k, 1)

for i = 1:maxIter
    value!(ŷ, risk, X, θ, y)
    grad!(▽, risk, X, θ, y, ŷ)
    θ = θ - α .* vec(▽)
end

J = value!(ŷ, risk, X, θ, y)
@test 0.09 < J < 0.1
mp = scatterplot(x, y, color = :blue, height = 5, ylim=[-1.1, 1.1])
lineplot!(mp, x1, vec(value(pred, X1, θ)), color = :red)
print(mp)
