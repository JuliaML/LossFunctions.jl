# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

module LossFunctions

using Markdown

import Base: sum
import Statistics: mean
import Requires: @init, @require

# trait functions
include("traits.jl")

# loss functions
include("losses.jl")

# IO methods
include("io.jl")

# Extensions
if !isdefined(Base, :get_extension)
  @init @require CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597" include(
    "../ext/LossFunctionsCategoricalArraysExt.jl"
  )
end

export
  # margin-based losses
  ZeroOneLoss,
  LogitMarginLoss,
  PerceptronLoss,
  HingeLoss,
  L1HingeLoss,
  L2HingeLoss,
  SmoothedL1HingeLoss,
  ModifiedHuberLoss,
  L2MarginLoss,
  ExpLoss,
  SigmoidLoss,
  DWDMarginLoss,

  # distance-based losses
  LPDistLoss,
  L1DistLoss,
  L2DistLoss,
  PeriodicLoss,
  HuberLoss,
  EpsilonInsLoss,
  L1EpsilonInsLoss,
  L2EpsilonInsLoss,
  LogitDistLoss,
  QuantileLoss,
  LogCoshLoss,

  # other losses
  MisclassLoss,
  PoissonLoss,
  CrossEntropyLoss,

  # meta losses
  ScaledLoss,
  WeightedMarginLoss,

  # reexport mean
  mean

end # module
