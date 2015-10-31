
using RDatasets
using UnicodePlots
using LearnBase

# Load data of interest from the Rdatasets package
myData = dataset("datasets", "cars")

# create target and design matrix
x = convert(Array,myData[1])
x = (x - mean(x)) / sqrt(var(x))
y = convert(Array,myData[2])
m = length(y)
X = [ones(m) x]'

# Set hyper parameters
θ = [0, 0]
α = 0.05
maxIter = 100

loss = L2DistLoss()
pred = LinearPredictor(bias = 0)

# Perform gradient descent
J = zeros(maxIter)
print("Starting gradient descent ... ")
for i = 1:maxIter
  ŷ = pred(X, θ)
  J[i] = mean(loss(y, ŷ))
  ▽ = mean(loss'(y, ŷ) .* pred'(X, θ), 2)
  θ = θ + α .* vec(▽)
end
println("DONE")

# Plot results
mp=scatterplot(x, y, color=:blue)
lineplot!(mp, x, vec(value(pred, X, θ)), color = :red)
println(mp)
println(lineplot(1:maxIter, J))
