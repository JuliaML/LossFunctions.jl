
using UnicodePlots
using LearnBase

# create target and design matrix
X, y = RegressionData.polynome(-10:.1:10, [-.1, 1, 10], noise=1.)
x = vec(X[end,:])

# Set hyper parameters
θ = randn(3)
α = 0.00005
maxIter = 100000

loss = LossFunctions.L2DistLoss()
reg = Penalties.L2Penalty(.001)
pred = LinearPredictor(bias = 1)
risk = RiskModel(pred, loss, reg)
ŷ = pred(X, θ)

# function muh(risk, X, w, y)
#     ŷ = zeros(1, size(X,2))
#     buffer = zeros(1, length(w))
#     for i = 1:1000
#         grad!(buffer, risk, X, w, y, ŷ)
#     end
# end
# @time muh(risk, X, θ, y)

# Perform gradient descent
J = zeros(maxIter)
print("Starting gradient descent ... ")
for i = 1:maxIter
    J[i] = value!(ŷ, risk, X, θ, y)
    ▽ = grad(risk, X, θ, y, ŷ)
    θ = θ - α .* vec(▽)
end
println("DONE")

println(θ)
# Plot results
mp = scatterplot(x, y, color = :blue)
lineplot!(mp, x, vec(value(pred, X, θ)), color = :red)
println(mp)
println(lineplot(1:maxIter, J))
