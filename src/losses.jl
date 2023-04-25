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
for FUN in (:value, :deriv, :deriv2)
    @eval begin
        function ($FUN)(loss::SupervisedLoss, outputs, targets, ::AggMode.Sum)
            sum(loss(ŷ, y) for (ŷ, y) in zip(outputs, targets))
        end

        function ($FUN)(loss::SupervisedLoss, outputs, targets, ::AggMode.Mean)
            T = typeof(loss(first(outputs), first(targets)))
            l = zero(T)
            n = 0
            for (ŷ, y) in zip(outputs, targets)
                l += loss(ŷ, y)
                n += 1
            end
            l / n
        end

        function ($FUN)(loss::SupervisedLoss, outputs, targets, agg::AggMode.WeightedSum)
            l = sum(w * loss(ŷ, y) for (ŷ, y, w) in zip(outputs, targets, agg.weights))
            w = sum(agg.weights)
            d = agg.normalize ? w : one(w)
            l / d
        end

        function ($FUN)(loss::SupervisedLoss, outputs, targets, agg::AggMode.WeightedMean)
            T = typeof(loss(first(outputs), first(targets)))
            l = zero(T)
            n = 0
            for (ŷ, y, w) in zip(outputs, targets, agg.weights)
                l += w * loss(ŷ, y)
                n += 1
            end
            w = sum(agg.weights)
            d = agg.normalize ? n * w : n * one(w)
            l / d
        end
    end
end