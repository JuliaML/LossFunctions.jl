module LossFunctionsTests

using LossFunctions
using LossFunctions.Traits
using CategoricalArrays
using DualNumbers
using Statistics
using Random
using Test

tests = ["core.jl", "props.jl", "agg.jl"]

# for deterministic testing
Random.seed!(1234)

distance_losses = [
  L2DistLoss(),
  LPDistLoss(2.0),
  L1DistLoss(),
  LPDistLoss(1.0),
  LPDistLoss(0.5),
  LPDistLoss(1.5),
  LPDistLoss(3),
  LogitDistLoss(),
  L1EpsilonInsLoss(0.5),
  EpsilonInsLoss(1.5),
  L2EpsilonInsLoss(0.5),
  L2EpsilonInsLoss(1.5),
  PeriodicLoss(1),
  PeriodicLoss(1.5),
  HuberLoss(1),
  HuberLoss(1.5),
  QuantileLoss(0.2),
  QuantileLoss(0.5),
  QuantileLoss(0.8),
  LogCoshLoss()
]

margin_losses = [
  LogitMarginLoss(),
  L1HingeLoss(),
  L2HingeLoss(),
  PerceptronLoss(),
  SmoothedL1HingeLoss(0.5),
  SmoothedL1HingeLoss(1),
  SmoothedL1HingeLoss(2),
  ModifiedHuberLoss(),
  ZeroOneLoss(),
  L2MarginLoss(),
  ExpLoss(),
  SigmoidLoss(),
  DWDMarginLoss(0.5),
  DWDMarginLoss(1),
  DWDMarginLoss(2)
]

other_losses = [MisclassLoss(), PoissonLoss(), CrossEntropyLoss()]

for t in tests
  @testset "$t" begin
    include(t)
  end
end

end # module
