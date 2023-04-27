# type alias to make code more readable
Scalar = Union{Number,CategoricalValue}

# convenient functor interface
(loss::SupervisedLoss)(output::Scalar, target::Scalar) = value(loss, output, target)

# fallback to unary evaluation
value(loss::DistanceLoss, output::Number, target::Number)  = value(loss, output - target)
deriv(loss::DistanceLoss, output::Number, target::Number)  = deriv(loss, output - target)
deriv2(loss::DistanceLoss, output::Number, target::Number) = deriv2(loss, output - target)

value(loss::MarginLoss, output::Number, target::Number)  = value(loss, target * output)
deriv(loss::MarginLoss, output::Number, target::Number)  = target * deriv(loss, target * output)
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
    sum(loss(ŷ, y) for (ŷ, y) in zip(outputs, targets))
end

"""
    sum(loss, outputs, targets, weights; normalize=true)

Return sum of `loss` values over the iterables `outputs` and `targets`.
The `weights` determine the importance of each observation. The option
`normalize` divides the result by the sum of the weights.
"""
function sum(loss::SupervisedLoss, outputs, targets, weights; normalize=true)
    s = sum(w * loss(ŷ, y) for (ŷ, y, w) in zip(outputs, targets, weights))
    n = normalize ? sum(weights) : one(first(weights))
    s / n
end

"""
    mean(loss, outputs, targets)

Return mean of `loss` values over the iterables `outputs` and `targets`.
"""
function mean(loss::SupervisedLoss, outputs, targets)
    mean(loss(ŷ, y) for (ŷ, y) in zip(outputs, targets))
end

"""
    mean(loss, outputs, targets, weights; normalize=true)

Return mean of `loss` values over the iterables `outputs` and `targets`.
The `weights` determine the importance of each observation. The option
`normalize` divides the result by the sum of the weights.
"""
function mean(loss::SupervisedLoss, outputs, targets, weights; normalize=true)
    m = mean(w * loss(ŷ, y) for (ŷ, y, w) in zip(outputs, targets, weights))
    n = normalize ? sum(weights) : one(first(weights))
    m / n
end