# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

# fallback to unary evaluation
(loss::DistanceLoss)(output::Number, target::Number) = loss(output - target)
deriv(loss::DistanceLoss, output::Number, target::Number) = deriv(loss, output - target)
deriv2(loss::DistanceLoss, output::Number, target::Number) = deriv2(loss, output - target)
(loss::MarginLoss)(output::Number, target::Number) = loss(target * output)
deriv(loss::MarginLoss, output::Number, target::Number) = target * deriv(loss, target * output)
deriv2(loss::MarginLoss, output::Number, target::Number) = deriv2(loss, target * output)

# broadcasting behavior
Broadcast.broadcastable(loss::SupervisedLoss) = Ref(loss)

# ------------------
# AVAILABLE LOSSES
# ------------------

include("losses/distance.jl")
include("losses/margin.jl")
include("losses/other.jl")

# meta-losses
include("losses/scaled.jl")
include("losses/weighted.jl")

# ----------------------
# AGGREGATION BEHAVIOR
# ----------------------

"""
    sum(loss, outputs, targets)

Return sum of `loss` values over the iterables `outputs` and `targets`.
"""
function sum(loss::SupervisedLoss, outputs, targets)
  sum(i -> loss(outputs[i], targets[i]), eachindex(outputs, targets))
end

"""
    sum(loss, outputs, targets, weights; normalize=true)

Return sum of `loss` values over the iterables `outputs` and `targets`.
The `weights` determine the importance of each observation. The option
`normalize` divides the result by the sum of the weights.
"""
function sum(loss::SupervisedLoss, outputs, targets, weights; normalize=true)
  s = sum(i -> weights[i] * loss(outputs[i], targets[i]), eachindex(outputs, targets, weights))
  n = normalize ? sum(weights) : one(first(weights))
  s / n
end

"""
    mean(loss, outputs, targets)

Return mean of `loss` values over the iterables `outputs` and `targets`.
"""
function mean(loss::SupervisedLoss, outputs, targets)
  mean(i -> loss(outputs[i], targets[i]), eachindex(outputs, targets))
end

"""
    mean(loss, outputs, targets, weights; normalize=true)

Return mean of `loss` values over the iterables `outputs` and `targets`.
The `weights` determine the importance of each observation. The option
`normalize` divides the result by the sum of the weights.
"""
function mean(loss::SupervisedLoss, outputs, targets, weights; normalize=true)
  m = mean(i -> weights[i] * loss(outputs[i], targets[i]), eachindex(outputs, targets, weights))
  n = normalize ? sum(weights) : one(first(weights))
  m / n
end
