# ==========================================================================

msg("Test linear regression with value_grad_fun")

w = [1, 10]
x, y = DataGenerators.noisy_poly(w, -10:.2:10, noise = .5)
X = x'

# Set hyper parameters
θ = zeros(length(w))
α = 0.005
maxIter = 1000

loss = LossFunctions.L2DistLoss()
pred = LinearPredictor(bias = 1)
risk = RiskModel(pred, loss)
riskfunc = RiskFunctional(risk, X, y)

fg! = value_grad_fun(riskfunc, θ)

J = 0.
▽ = zeros(θ)
for i = 1:maxIter
    J = fg!(θ, ▽)
    θ = θ - α .* ▽
end

@test J < 0.3
@test sumabs(w - θ) < .15
mp = scatterplot(x, y, color = :blue, height = 5, margin = 5)
lineplot!(mp, x, vec(value(pred, X, θ)), color = :red)
print(mp)

# ==========================================================================

msg("Test linear regression with value_fun and grad_fun")

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
risk = RiskModel(pred, loss)
riskfunc = RiskFunctional(risk, X, y)

f = value_fun(riskfunc, θ)
g! = grad_fun(riskfunc, θ)

J = 0.
▽ = zeros(θ)
for i = 1:maxIter
    J = f(θ)
    g!(θ, ▽)
    θ = θ - α .* ▽
end

@test J < 0.04
mp = scatterplot(x, y, color = :blue, height = 5)
lineplot!(mp, x, vec(value(pred, X, θ)), color = :red)
print(mp)
