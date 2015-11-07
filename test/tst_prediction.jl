
using RDatasets
using UnicodePlots
using LearnBase

# Load data of interest from the Rdatasets package
myData = dataset("datasets", "cars")

cs(x) = (x - mean(x)) / sqrt(var(x))

# create target and design matrix
x = cs(convert(Array,myData[1]))
y = convert(Array,myData[2])
y = y .+ sin(x.*2) * 20
#y[2] = 100
#y[10] = 100
m = length(y)
X = [cs(sqrt(abs(x))) x cs(x.^2) cs(x.^3) cs(x.^4)]'

# Set hyper parameters
θ = [0., 0, 0, 0, 0, 0]
α = 0.05
maxIter = 300

loss = L2DistLoss()
reg = L2Penalty(.1)
pred = LinearPredictor(bias = 1)
risk = RiskModel(pred, loss)
ŷ = pred(X, θ)

# Perform gradient descent
J = zeros(maxIter)
print("Starting gradient descent ... ")
for i = 1:maxIter
  J[i] = value!(ŷ, risk, X, θ, y)
  grd = grad(risk, X, θ, y)
  ▽ = mean(deriv(loss, y, ŷ) .* pred'(X, θ), 2)
  addgrad!(view(▽, 1:(length(θ)-1)), reg, view(θ, 1:(length(θ)-1)))
  θ = θ - α .* vec(▽)
end
println("DONE")

println(θ)
# Plot results
mp=scatterplot(x, y, color=:blue)
lineplot!(mp, x, vec(value(pred, X, θ)), color = :red)
println(mp)
println(lineplot(1:maxIter, J))
